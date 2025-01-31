# Copyright (c) University of Strasbourg, All Rights Reserved.
import time
import torch
from torch.cuda.amp import autocast
import utils
from utils import gpu_mem_usage, cumulative_sum
from utils import AllGather, log
allgather = AllGather.apply

def train(writer, train_loader, model, criterion, optimizer, scheduler, epoch, dataset, args, scaler, dataset_idx):
    running_loss = 0.0
    s = time.time()
    for i_batch, sample_batch in enumerate(train_loader):
        batch_loss = TrainOneBatch_phase_abstract(model, optimizer, scheduler, sample_batch, criterion, args, scaler, dataset_idx)

        writer.add_scalar('train batch loss '+str(dataset_idx), batch_loss, epoch)

        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose and args.rank == 0:
            d = time.time() - s
            log(
                "Training Phase/Abstract level: Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f, GPU Usage %.4f G"
                % (
                    epoch,
                    d,
                    args.batch_size_train[dataset_idx] * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.n_display,
                    optimizer.param_groups[0]['lr'],
                    gpu_mem_usage()
                ), args
            )
            running_loss = 0.0
            s = time.time()


def TrainOneBatch_phase_abstract(model, opt, scheduler, data, loss_fun, args, scaler, dataset_idx):
    # video: bs, t, c, h, w
    # sents: sents_0
    # -- # sents_0: bs, num_candidates, token_len

    video = data["video"].float().cuda(args.gpu, non_blocking=args.pin_memory)

    if 'pos_step' in data.keys():
        pos_step = data['pos_step'].cuda(args.gpu, non_blocking=args.pin_memory)
    else:
        pos_step = None


    assert len(video.shape) == 5, "video is not a seq!"
    bs, t, c, h, w = video.shape
    video = video.view(-1, c, h, w)

    splits = [i * bs for i in [1, args.trainset_config[dataset_idx].candidates[0].item()]] # bs = 20 -> [20, 60]

    splits = cumulative_sum(splits) # [20, 80]
    combined_text = {}
    combined_text['input_ids'] = torch.cat(
        [
            data['sents_summarize']['input_ids'].view(-1, 200), 
            data['sents_0']['input_ids'].view(-1, 200)
            ]
    ).cuda(args.gpu, non_blocking=args.pin_memory)


    combined_text['token_type_ids'] = torch.cat(
        [
            data['sents_summarize']['token_type_ids'].view(-1, 200), 
            data['sents_0']['token_type_ids'].view(-1, 200)
            ]
    ).cuda(args.gpu, non_blocking=args.pin_memory)


    combined_text['attention_mask'] = torch.cat(
        [
            data['sents_summarize']['attention_mask'].view(-1, 200), 
            data['sents_0']['attention_mask'].view(-1, 200)
            ]
    ).cuda(args.gpu, non_blocking=args.pin_memory)

    opt.zero_grad()
    with torch.set_grad_enabled(True):
        with autocast():

            output = model(video, combined_text, mode='all')
        
            video_embd, text_embd_combined = output['img_emb'], output['text_emb']
            video_embd_frame = video_embd.view(bs, t, -1)
            video_embd = video_embd_frame.mean(dim=1) # (bs, d)
            bs, d = video_embd.shape

            total_text_embd = []
            for idx, sp in enumerate(splits):
                if idx == 0:
                    curr_text_embd = text_embd_combined[0:sp, :]

                else:
                    curr_text_embd = text_embd_combined[splits[idx-1]:sp, :]
                    curr_text_embd = curr_text_embd.view(-1, args.trainset_config[dataset_idx].candidates[0], d)

                if args.distributed:
                    curr_text_embd = allgather(curr_text_embd, args)
                total_text_embd.append(curr_text_embd)
                
                del curr_text_embd

            # total_text_embd [asr_center, asr_other] asr_center:(bs, n_candi, 768)
            if args.distributed:
                video_embd = allgather(video_embd, args)
                video_embd_frame = allgather(video_embd_frame, args)
                if pos_step is not None:
                    pos_step = allgather(pos_step, args)

            if 'logit_scale' in output.keys():
                utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
                logit_scale = output['logit_scale']
            else:
                logit_scale = None

            loss = loss_fun(video_embd, total_text_embd, video_embd_frame, pos_step=pos_step, logit_scale=logit_scale)
    

    loss.backward()
    opt.step()
    scheduler.step()
    
    
    return loss.item()