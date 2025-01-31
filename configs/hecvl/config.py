import torch
import torchvision.transforms as transforms
from utils import Gaussian

config = dict(
    multiprocessing_distributed=False, # Use distributed gpu training or not
    gpu=0,
    seed=1,
    dist_backend='nccl',
    dist_url='tcp://224.66.41.62:23456',
    cudnn_benchmark=0,
    dist_file='dist/dist-file',
    rank=0,
    world_size=-1,
    pin_memory=False,
    evaluate=False, # Directly evaluate the model
    momentum=0.9,
    lr=0.00008,
    start_epoch=0,
    epochs=150,
    eval_epoch=5,
    pretrain_cnn_path='', # load pre-trained weights
    # max_words=77,
    warmup_steps=0,
    batch_size_train=[2, 1, 1],
    batch_size_val=40,
    batch_size_test=40,
    num_thread_reader=8,
    optimizer='adam',
    source_num=2,
    n_display=1,
    verbose=True,

    # cholec triplet downstream dataset
    triplet_mode='test',
    triplet_prompt='./data/prompts/cholec_triplet.txt',
    set_chlg_eval=False,

    # zero-shot phase recognition downstream dataset
    batch_size_zero=[60, 60, 60, 60],
    zero_prompt=['./data/prompts/class_prompt_cholec80_manual.txt',
                 './data/prompts/class_prompt_autolapa_manual.txt'], 
    log_prefix=['cholec80 recognition',
                'autolapa recognition'],

    # model construction
    model_config = dict(
        type='MVNet',
        backbone_img = dict(
            type='img_backbones/ImageEncoder',
            num_classes=768,
            pretrained='random', #'imagenet',
            backbone_name='resnet_50',
            img_norm=False
        ),
        backbone_text= dict(
            type='text_backbones/BertEncoder',
            text_bert_type='/home2020/home/icube/kunyuan/release/offline_src/biobert_pretrain_output_all_notes_150000', # e
            text_last_n_layers=4,
            text_aggregate_method='sum',
            text_norm=False,
            text_embedding_dim=768,
            text_freeze_bert=False,
            text_agg_tokens=True
        )
    ),
    trainset_config = [
        # action level dataset
        dict(
            type='video_text_ssl',
            csv='./data/splits/train_narration.csv',
            video_root='./data/videos',
            caption_roots=[
              './data/aws_narration_texts',
              './data/whisper_narration_texts'],
            candidates=torch.IntTensor([1, 2]), # support [1,num[0-n]] [num[0-n],1]
            min_time=4.0,
            fps=2,
            num_frames=4,
            size=336,
            crop_only=False,
            center_crop=False,
            random_flip=True,
            bert_type='/home2020/home/icube/kunyuan/release/offline_src/biobert_pretrain_output_all_notes_150000',
            clip_min=3,
            clip_max=5,
            transforms=transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            aug_transforms= transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(336, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([Gaussian([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ),
        # keystep level dataset
        dict(
            type='video_text_phase',
            csv='./data/splits/train_keystep.csv',
            video_root='./data/videos',
            phase_root='./data/keystep_texts',
            caption_roots=[
              './data/aws_narration_texts',
              './data/whisper_narration_texts'],
            min_time=4.0,
            candidates=torch.IntTensor([3]), # support [1,num[0-n]] [num[0-n],1]
            num_frames_clip=16, # frames from each action clip
            size=336,
            crop_only=False,
            center_crop=False,
            random_flip=True,
            bert_type='/home2020/home/icube/kunyuan/release/offline_src/biobert_pretrain_output_all_notes_150000',
            transforms=transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        ),
        # video level dataset
        dict(
            type='video_text_phase_aug_phase_abstract',
            csv='./data/splits/train_abstract.csv',
            video_root='./data/videos',
            phase_root='./data/abstract_texts',
            caption_roots=['./data/keystep_texts'],
            min_time=4.0,
            candidates=torch.IntTensor([8]), # support [1,num[0-n]] [num[0-n],1]
            num_frames_clip=64,
            size=336,
            crop_only=False,
            center_crop=False,
            random_flip=True,
            bert_type='/home2020/home/icube/kunyuan/release/offline_src/biobert_pretrain_output_all_notes_150000', 
            transforms=transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        )    
    ],
    valset_config = 
        # action level dataset
        dict(
        type='video_text_ssl',
        csv='./data/splits/val.csv',
        video_root='./data/videos',
        caption_roots=[
          './data/aws_narration_texts',
          './data/whisper_narration_texts'
        ],
        candidates=torch.IntTensor([1, 2]), # support [1,num[0-n]] [num[0-n],1]
        min_time=4.0,
        fps=2,
        num_frames=4,
        size=336,
        crop_only=False,
        center_crop=False,
        random_flip=True,
        bert_type='/home2020/home/icube/kunyuan/release/offline_src/biobert_pretrain_output_all_notes_150000',
        clip_min=3,
        clip_max=5,
        transforms=transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        aug_transforms= transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomResizedCrop(336, scale=(0.08, 1.)),
          transforms.RandomApply([
              transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
          ], p=0.8),
          transforms.RandomGrayscale(p=0.2),
          transforms.RandomApply([Gaussian([.1, 2.])], p=0.5),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])),
    cholect50_testset_config = dict(
        type='CholecT50',
        dataset_dir='./data/downstream_datasets/CholecT50-challenge-train',
        split='test',
        preprocess=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ),
    cholec_autolaparo_testset_config = [
       [
            dict(
            type='Recognition_frame',
            csv_root='./data/downstream_datasets/cholec80/csvs',
            vid='video%02d.csv'%i,
            video_root='./data/downstream_datasets/cholec80/frames',
            transforms=transforms.Compose(
                [
                transforms.Resize((360, 640)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
                ),
            ) for i in range(49, 56)
        ],
        [
            dict(
            type='Recognition_frame',
            csv_root='./data/downstream_datasets/autolaparo/csvs',
            vid='%02d.csv'%i,
            video_root='./data/downstream_datasets/autolaparo/frames',
            transforms=transforms.Compose(
                [
                transforms.Resize((360, 640)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
                ),
            ) for i in range(15, 22)
        ]
    ],
    loss_config_hierarchy = [
        dict(
            type='SSL_VL_Loss_new',
            alpha_weight=0.75
        ),
        dict(
            type='hier_infonce',
            alpha_weight=0.75
        ),
        dict(
            type='hier_infonce',
            alpha_weight=0.75
        ),
    ]
)