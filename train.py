import argparse
import os
import random

os.environ["TOKENIZERS_PARALLELISM"]='false'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from mmengine.config import Config
from codes.models import *
from codes.datasets import *
from codes.loss import *
from utils import *

import torch
import torch.multiprocessing as mp
torch.autograd.set_detect_anomaly(True)
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from codes.engines.engine_clip_vl_ssl import train_vl_ssl
from codes.engines.engine_phase_video_hierarchy import train as train_hierarchy
from codes.engines.engine_val import val, evaluate_retrieval_Bert, evaluate_retrieval_CLIP, evaluate_triplet, evaluate_zero_frame 

from codes.datasets.utils import bert_token_collate_fn
from torch.utils.tensorboard import SummaryWriter


def get_args(description='SurgVLP'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--work_dir', default='', type=str, help='dir to save ')
    parser.add_argument('--resume', action='store_true', help='')
    args = parser.parse_args()
    return args, parser

def main():
    register_all_modules(init_default_scope=False)
    args, _ = get_args()

    config_name = os.path.join(os.path.dirname(__file__), args.work_dir, 'config.py')
    configs = Config.fromfile(config_name)['config']

    if args.resume:
        configs['resume'] = args.resume
        configs['work_dir'] = args.work_dir

    if configs.seed is not None:
        random.seed(configs.seed)
        torch.manual_seed(configs.seed)
 
    if configs.world_size == -1 and "SLURM_JOB_NUM_NODES" in os.environ:
        configs.world_size = int(os.environ["SLURM_JOB_NUM_NODES"])
        configs.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        configs.dist_url = "file://{}.{}".format(os.path.realpath(configs.dist_file), jobid)
        print(
            "dist-url:{} at PROCID {} / {}".format(
                configs.dist_url, configs.rank, configs.world_size
            )
        )
    else:
        print('SLURM not supported')

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if configs.multiprocessing_distributed:
        configs.world_size = ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, configs))
    else:
        main_worker(configs.gpu, ngpus_per_node, configs)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    if args.cudnn_benchmark:
        cudnn.benchmark = True
    log(
        "Starting training loop for rank: {}".format(
            args.rank
        ), args
    )
    log(str(args), args)

    # tensorboard init
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(os.path.join(args.work_dir, 'tensorlog'))

    # Model Configuration
    model = build_algorithm(args.model_config)

    # amp scaler
    scaler = None

    if args.distributed:
        # distributed training
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size_train = [int(i / ngpus_per_node) for i in args.batch_size_train]

            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)

            args.batch_size_test = int(args.batch_size_test / ngpus_per_node)

            args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        # single gpu training
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Loading pre-trained weights for evaluation
    if args.pretrain_cnn_path:
        print('loading: ',args.pretrain_cnn_path)
        net_data = torch.load(args.pretrain_cnn_path)['state_dict']
        a, b = model.load_state_dict(net_data, strict=False)
        log("=> missing keys '{}'".format(a), args)
        log("=> unexpected keys '{}'".format(b), args)

    # Dataset Configuration
    print('Building Dataset...')
    train_datasets = [build_dataset(i) for i in args.trainset_config]
    val_dataset = build_dataset(args.valset_config)


    # Downstream dataset for zero-shot evaluation during pretraining
    zero_datasets = []
    for c_list in args.cholec_autolaparo_testset_config:
        zero_datasets.append([build_dataset(c) for c in c_list])
    
    # Dataset sampler
    print(args.distributed, 'distributed')
    if args.distributed:
        train_samplers = [torch.utils.data.distributed.DistributedSampler(train_dataset) for train_dataset in train_datasets]

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        
        zero_samplers = []
        for zero_dataset_list in zero_datasets:
            zero_samplers.append([torch.utils.data.distributed.DistributedSampler(zero_dataset_vid) for zero_dataset_vid in zero_dataset_list])
    else:
        train_samplers = [None for train_dataset in train_datasets]
        test_sampler = None
        val_sampler = None
        zero_samplers = []
        for zero_dataset_list in zero_datasets:
            zero_samplers.append([None for zero_dataset_vid in zero_dataset_list])


    # Train dataloader
    print('Training Dataloader...')
    train_loaders = []
    for idx, dataset in enumerate(train_datasets):
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size_train[idx],
            shuffle=(train_samplers[idx] is None),
            drop_last=False,
            num_workers=args.num_thread_reader,
            pin_memory=args.pin_memory,
            sampler=train_samplers[idx],
        )
        train_loaders.append(train_loader)
    
    print('Val Dataloader...')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_val,
        shuffle=(val_sampler is None),
        drop_last=False,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory,
        sampler=val_sampler,
    )

    print('Zeroshot Phase Recognition Dataloader...')
    # Phase recognition test dataloaders
    # For each dataset, we have one loader for one video to perform video-wise phase recognition, which is same to SOTA benchmarks
    # like TeCNO <https://github.com/tobiascz/TeCNO>
    zero_loaders = []
    for idx, dataset in enumerate(zero_datasets):
        sampler_dataset = zero_samplers[idx]
        vid_loaders = []
        for vid_idx, vid_dataset in enumerate(dataset):
            sampler = sampler_dataset[vid_idx]
            zero_loader = torch.utils.data.DataLoader(
                vid_dataset,
                batch_size=args.batch_size_zero[idx],
                shuffle=False,
                drop_last=False,
                num_workers=args.num_thread_reader,
                pin_memory=args.pin_memory,
                sampler=sampler,
            )
            vid_loaders.append(zero_loader)
        zero_loaders.append(vid_loaders)

    # Triplet test dataloaders, one loader per vid
    print('Triplet Evaluation Dataloader...')
    if args.triplet_mode == "test": videos = [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
    elif args.triplet_mode == "val": videos = [8, 12, 29, 50, 78]
    records  = ['VID{}'.format(str(v).zfill(2)) for v in videos]
    triplet_loaders = []
    triplet_config = args.cholect50_testset_config
    for vid in records:
        triplet_config['list_video'] = vid
        triplet_loaders.append(build_dataset(triplet_config)(batch_size=args.batch_size_test, shuffle=False))
    ############

    # Loss Function Configuration
    criterions = [build_loss(i) for i in args.loss_config_hierarchy]

    # Optimizer Configuration
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momemtum)

    # Scheduler Configuration
    len_iter = [len(i) for i in train_loaders]
    len_iter = sum(len_iter)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, (len_iter) * args.epochs)
    print('Warmup steps: ', args.warmup_steps, 'Total_steps: ', len(train_loader) * args.epochs)


    # optionally resume from a checkpoint
    checkpoint_dir = args.work_dir
    if args.resume:
        checkpoint_path = get_last_checkpoint(checkpoint_dir)
        if checkpoint_path:
            log("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint["epoch"]
            log("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint["epoch"]), args)

            model.load_state_dict(checkpoint["state_dict"])
            if 'optimizer' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])
            if 'scheduler' in checkpoint.keys():
                scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            log("=> no checkpoint found at '{}'".format(args.resume), args)

    # Evaluate
    if args.evaluate:
        model.eval()

        # Evaluate Triplet
        evaluate_triplet(writer, triplet_loaders, model, args.start_epoch, args)

        # Evaluate Zero
        evaluate_zero_frame(writer, zero_loaders, model, args.start_epoch, args)

        model.train()

    # Epoch based training iteration
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for train_sampler in train_samplers:
                train_sampler.set_epoch(epoch)

        # train each dataloader for n epoch
        for idx, train_loader in enumerate(train_loaders):
            train_dataset = train_datasets[idx]
            train_vl_ssl(writer, train_loader, model, criterions[idx], optimizer, scheduler, epoch, train_dataset, args, scaler, idx)
            
            val(writer, val_loader, model, criterions[0], epoch, val_dataset, args)

        if epoch % args.eval_epoch == 0 and epoch != 0:
            model.eval()

            # Test Triplet
            print('eval triplet')
            evaluate_triplet(writer, triplet_loaders, model, epoch, args)

            # Test Zero
            print('eval recognition')
            evaluate_zero_frame(writer, zero_loaders, model, epoch, args)

            model.train()
            
            if args.rank == 0:
                save_checkpoint_eval(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, checkpoint_dir, epoch + 1
                )


        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, checkpoint_dir, epoch + 1
            )


if __name__ == '__main__':
    main()
