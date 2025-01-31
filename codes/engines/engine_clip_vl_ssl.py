# Copyright (c) University of Strasbourg, All Rights Reserved.
import time
import torch
from torch.cuda.amp import autocast
from utils import gpu_mem_usage, cumulative_sum
from utils import AllGather, log
import utils
allgather = AllGather.apply


def TrainOneBatch_vl_ssl(model, opt, scheduler, data, loss_fun, args, scaler, dataset_idx):
    # video: bs, t, c, h, w
    # video_aug1: bs, t, c, h, w
    # video_aug1: bs, t, c, h, w
    # sents: sents_0, sent_1....
    # -- # sents_0: bs, num_candidates, token_len
    video = data["video"].float().cuda(args.gpu, non_blocking=args.pin_memory)
    video_aug1 = data["video_aug1"].float().cuda(args.gpu, non_blocking=args.pin_memory)
    video_aug2 = data["video_aug2"].float().cuda(args.gpu, non_blocking=args.pin_memory)

    assert len(video.shape) == 5, "video is not a seq!"
    bs, t, c, h, w = video.shape
    video = video.view(-1, c, h, w)
    video_aug1 = video_aug1.view(-1, c, h, w)
    video_aug2 = video_aug2.view(-1, c, h, w)

    bs_t = video_aug2.shape[0]
    assert video_aug1.shape[0] == bs_t
    assert video.shape[0] == bs_t

    video_input = torch.cat([video, video_aug1, video_aug2], 0) # (bs_t + bs_t + bs_t, c, h, w)

    assert dataset_idx==0

    splits = [i * bs for i in args.trainset_config[dataset_idx].candidates.tolist()] # bs = 20 -> [20, 60]
    splits = cumulative_sum(splits) # [20, 80]

    combined_text = {}
    combined_text['input_ids'] = torch.cat([
        data['sents_'+str(i)]['input_ids'].view(-1, 77) for i in range(args.source_num)
        ], 0).cuda(args.gpu, non_blocking=args.pin_memory)
    
    combined_text['token_type_ids'] = torch.cat([
        data['sents_'+str(i)]['token_type_ids'].view(-1, 77) for i in range(args.source_num)
        ], 0).cuda(args.gpu, non_blocking=args.pin_memory)
    combined_text['attention_mask'] = torch.cat([
        data['sents_'+str(i)]['attention_mask'].view(-1, 77) for i in range(args.source_num)
        ], 0).cuda(args.gpu, non_blocking=args.pin_memory)

    opt.zero_grad()
    with torch.set_grad_enabled(True):
        with autocast():

            output = model(video_input, combined_text, mode='action')

            text_embd_combined = output['text_emb']

            video_embd = output['img_emb']  # (bs_t+bs_t+bs_t, d)
            video_ori_embd = video_embd[:bs_t,...]
            video_embd_aug1 = video_embd[bs_t:2*bs_t,...]
            video_embd_aug2 = video_embd[2*bs_t:3*bs_t,...]


            video_ori_embd = video_ori_embd.view(bs, t, -1).mean(dim=1, keepdim=False)
            video_embd_aug1 = video_embd_aug1.view(bs, t, -1).mean(dim=1, keepdim=False)
            video_embd_aug2 = video_embd_aug2.view(bs, t, -1).mean(dim=1, keepdim=False)

            if args.distributed:
                video_ori_embd = allgather(video_ori_embd, args)
                video_embd_aug1 = allgather(video_embd_aug1, args)
                video_embd_aug2 = allgather(video_embd_aug2, args)


            total_text_embd = []
            for idx, sp in enumerate(splits):
                if idx == 0:
                    curr_text_embd = text_embd_combined[0:sp, :]
                else:
                    curr_text_embd = text_embd_combined[splits[idx-1]:sp, :]

                if args.distributed:
                    curr_text_embd = allgather(curr_text_embd, args)
                total_text_embd.append(curr_text_embd)
                
                del curr_text_embd


            if 'logit_scale' in output.keys():
                utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
                logit_scale = output['logit_scale']
            else:
                logit_scale = None
            
            # total_text_embd [asr_center, asr_other] asr_center:(bs, n_candi, 768)
            loss = loss_fun(video_ori_embd, video_embd_aug1, video_embd_aug2, total_text_embd, logit_scale)
    
    loss.backward()
    opt.step()
    scheduler.step()
    
    
    return loss.item()

def train_vl_ssl(writer, train_loader, model, criterion, optimizer, scheduler, epoch, dataset, args, scaler, dataset_idx):
    running_loss = 0.0
    s = time.time()
    for i_batch, sample_batch in enumerate(train_loader):
        batch_loss = TrainOneBatch_vl_ssl(model, optimizer, scheduler, sample_batch, criterion, args, scaler, dataset_idx)

        writer.add_scalar('train batch loss', batch_loss, epoch)

        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose and args.rank == 0:
            d = time.time() - s
            log(
                "Training Action level: Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f GPU Usage %.4f G"
                % (
                    epoch + 1,
                    d,
                    args.batch_size_train[dataset_idx] * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.n_display,
                    optimizer.param_groups[0]['lr'],
                    gpu_mem_usage()
                ), args
            )
            running_loss = 0.0
            s = time.time()
