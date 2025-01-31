# Copyright (c) University of Strasbourg, All Rights Reserved.
import time
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer
from torch.cuda.amp import autocast
import ivtmetrics
from metrics import compute_metrics, calc_accuracy, calc_f1
from utils import gpu_mem_usage, cumulative_sum
from utils import AllGather, log
allgather = AllGather.apply


def ValidateOneBatch(model, data, loss_fun, args):
    # video: bs, t, c, h, w
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


    splits = [i * bs for i in args.valset_config.candidates.tolist()] # bs = 20 -> [20, 60]
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

    with torch.set_grad_enabled(False):
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

            if args.distributed:
                video_ori_embd = allgather(video_ori_embd, args)
                video_embd_aug1 = allgather(video_embd_aug1, args)
                video_embd_aug2 = allgather(video_embd_aug2, args)

            # total_text_embd [asr_center, asr_other] asr_center:(bs, n_candi, 768)s
            loss = loss_fun(video_ori_embd, video_embd_aug1, video_embd_aug2, total_text_embd)

    
    return loss.item()


def val(writer, val_loader, model, criterion, epoch, dataset, args):
    running_loss = 0.0
    s = time.time()
    for i_batch, sample_batch in enumerate(val_loader):
        batch_loss = ValidateOneBatch(model, sample_batch, criterion, args)

        writer.add_scalar('val batch loss', batch_loss, epoch)

        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose and args.rank == 0:
            d = time.time() - s
            log(
                "Validation Action level: Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, GPU Usage %.4f G"
                % (
                    epoch + 1,
                    d,
                    args.batch_size_val * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.n_display,
                    gpu_mem_usage()
                ), args
            )
            running_loss = 0.0
            s = time.time()



def evaluate_retrieval_Bert(writer, test_loader, model, epoch, args, dataset_name):
    all_txt_embd = []
    all_video_embd = []
    model.eval()
    if args.rank == 0:  
        log('Evaluating on {}'.format(dataset_name), args)
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            text = data['text']
            text['input_ids'] = text['input_ids'].view(-1, text['input_ids'].shape[-1]).cuda()
            text['token_type_ids'] = text['token_type_ids'].view(-1, text['token_type_ids'].shape[-1]).cuda()
            text['attention_mask'] = text['attention_mask'].view(-1, text['attention_mask'].shape[-1]).cuda()

            video = data['video'].float().cuda()

            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4]) # (bs*nc, c, h, w)
            output = model(video, text)
            video_embd, text_embd = output['img_emb'], output['text_emb']
            video_embd = video_embd.view(text_embd.shape[0], args.testset_config.num_frames_clip, text_embd.shape[1])
            video_embd = video_embd.mean(dim=1)

            # Normalize
            video_embd /= video_embd.norm(dim=-1, keepdim=True)
            text_embd /= text_embd.norm(dim=-1, keepdim=True)

            if args.distributed:
                video_embd = allgather(video_embd, args)
                text_embd = allgather(text_embd, args)
            if args.rank == 0:
                text_embd = text_embd.cpu().numpy()
                video_embd = video_embd.cpu().numpy()
                all_txt_embd.append(text_embd)
                all_video_embd.append(video_embd)
            
            # Delete variables to free up memory
            del text, video, video_embd, text_embd
            torch.cuda.empty_cache() # Free up unused memory

    model.train()
    if args.rank == 0:
        all_txt_embd = np.concatenate(all_txt_embd, axis=0)
        all_video_embd = np.concatenate(all_video_embd, axis=0)

        metrics_t2i = compute_metrics(np.dot(all_txt_embd, all_video_embd.T))
        log('Epoch {} Text-to-Image results: {}'.format(epoch, metrics_t2i), args)

        metrics_i2t = compute_metrics(np.dot(all_video_embd, all_txt_embd.T))
        log('Epoch {} Image-to-Text results: {}'.format(epoch, metrics_i2t), args)

        for k, v in metrics_t2i.items():
            writer.add_scalar('t2i '+k, v, epoch)

        for k, v in metrics_i2t.items():
            writer.add_scalar('i2t '+k, v, epoch)

def evaluate_retrieval_CLIP(writer, test_loader, model, epoch, args, dataset_name):
    all_txt_embd = []
    all_video_embd = []
    model.eval()
    if args.rank == 0:  
        log('Evaluating on {}'.format(dataset_name), args)
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            text = data['text'].cuda()
            text = text.view(-1, text.shape[-1])

            video = data['video'].float().cuda()

            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4]) # (bs*nc, c, h, w)

            video_embd = model.module.encode_image(video)
            text_embd = model.module.encode_text(text)
            video_embd = video_embd.view(text_embd.shape[0], args.num_frames_clip, text_embd.shape[1])
            video_embd = video_embd.mean(dim=1)

            # Normalize
            video_embd /= video_embd.norm(dim=-1, keepdim=True)
            text_embd /= text_embd.norm(dim=-1, keepdim=True)
            
            if args.distributed:
                video_embd = allgather(video_embd, args)
                text_embd = allgather(text_embd, args)
            if args.rank == 0:
                text_embd = text_embd.cpu().numpy()
                video_embd = video_embd.cpu().numpy()
                all_txt_embd.append(text_embd)
                all_video_embd.append(video_embd)
    model.train()
    if args.rank == 0:
        all_txt_embd = np.concatenate(all_txt_embd, axis=0)
        all_video_embd = np.concatenate(all_video_embd, axis=0)

        metrics_t2i = compute_metrics(np.dot(all_txt_embd, all_video_embd.T))
        log('Epoch {} Text-to-Image results: {}'.format(epoch, metrics_t2i), args)

        metrics_i2t = compute_metrics(np.dot(all_video_embd, all_txt_embd.T))
        log('Epoch {} Image-to-Text results: {}'.format(epoch, metrics_i2t), args)

        for k, v in metrics_t2i.items():
            writer.add_scalar('t2i '+k, v, epoch)

        for k, v in metrics_i2t.items():
            writer.add_scalar('i2t '+k, v, epoch)


def process_text(bert_type, text):
    tokenizer_clinical = AutoTokenizer.from_pretrained(bert_type)
    ixtoword = {v: k for k, v in tokenizer_clinical.get_vocab().items()}
    if type(text) == str:
        text = [text]

    processed_text_tensors = []
    for t in text:

        text_tensors = tokenizer_clinical(
            t,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=35,
        )
        text_tensors["sent"] = [
            ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
        ]
        processed_text_tensors.append(text_tensors)

    caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
    attention_mask = torch.stack(
        [x["attention_mask"] for x in processed_text_tensors]
    )
    token_type_ids = torch.stack(
        [x["token_type_ids"] for x in processed_text_tensors]
    )

    if len(text) == 1:
        caption_ids = caption_ids.squeeze(0).cuda()
        attention_mask = attention_mask.squeeze(0).cuda()#.to(device)
        token_type_ids = token_type_ids.squeeze(0).cuda()
    else:
        caption_ids = caption_ids.squeeze().cuda()
        attention_mask = attention_mask.squeeze().cuda()
        token_type_ids = token_type_ids.squeeze().cuda()

    cap_lens = []
    for txt in text:
        cap_lens.append(len([w for w in txt if not w.startswith("[")]))

    return {
        "input_ids": caption_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "cap_lens": cap_lens,
    }


def evaluate_triplet(writer, test_loaders, model, epoch, args):
    triplet_prompt=args.triplet_prompt
    set_chlg_eval=args.set_chlg_eval #False
    bert_type = args.valset_config.bert_type

    model.eval()
    with open(triplet_prompt) as f:
        lines = f.readlines()
    f.close()

    ivt_texts = [i.split(':')[-1].replace('\n','').replace(',',' ').replace('_',' ') for i in lines]
    
    i_texts = ['grapser', 'bipolar', 'hook', 'scissor', 'clipper', 'irrigator']

    v_texts = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack', 'null verb']

    t_texts = ['gallbladder', 'cystic plate', 'cystic duct', 'cystic artery', 'cystic pedicle', 'blood vessel', 'fluid', 'abdominal wall cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen bag', 'null target']
        
    # assert args.model_config.type == 'MVNet'
    i_texts = process_text(bert_type, i_texts)
    v_texts = process_text(bert_type, v_texts)
    t_texts = process_text(bert_type, t_texts)
    ivt_texts = process_text(bert_type, ivt_texts)
    with torch.no_grad():
        classifier_i = model(None, i_texts, mode='text')['text_emb'].cuda()
        classifier_v = model(None, v_texts, mode='text')['text_emb'].cuda()
        classifier_t = model(None, t_texts, mode='text')['text_emb'].cuda()
        classifier_ivt = model(None, ivt_texts, mode='text')['text_emb'].cuda()

    classifier_i /= classifier_i.norm(dim=-1, keepdim=True).cuda()
    classifier_v /= classifier_v.norm(dim=-1, keepdim=True).cuda()
    classifier_t /= classifier_t.norm(dim=-1, keepdim=True).cuda()
    classifier_ivt /= classifier_ivt.norm(dim=-1, keepdim=True).cuda()

    mAP = ivtmetrics.Recognition(100)
    mAPi = ivtmetrics.Recognition(6)
    mAPv = ivtmetrics.Recognition(10)
    mAPt = ivtmetrics.Recognition(15)
    mAP.reset_global()
    mAPi.reset_global()
    mAPv.reset_global()
    mAPt.reset_global()

    mAP.reset()  
    mAPv.reset() 
    mAPt.reset() 
    mAPi.reset() 

    
    activation = nn.Sigmoid()

    for dataloader in test_loaders:
        with torch.no_grad():
          with autocast():
            for i, (frames, (y_i, y_v, y_t, y_ivt)) in enumerate(dataloader): 
                y_i    = y_i.squeeze(1).cuda()
                y_v    = y_v.squeeze(1).cuda()
                y_t    = y_t.squeeze(1).cuda()
                y_ivt  = y_ivt.squeeze(1).cuda()
                frames = frames.cuda()


                frames_embd = model(frames, None, mode='video')['img_emb']
                frames_embd /= frames_embd.norm(dim=-1, keepdim=True)
                logit_i = torch.matmul(frames_embd, classifier_i.T) # (len(dataset), Class_i)
                logit_v = torch.matmul(frames_embd, classifier_v.T) # (len(dataset), Class_v)
                logit_t = torch.matmul(frames_embd, classifier_t.T) # (len(dataset), Class_t)
                logit_ivt = torch.matmul(frames_embd, classifier_ivt.T) # (len(dataset), Class_ivt)

                if args.distributed:

                    logit_i = allgather(logit_i, args)
                    logit_v = allgather(logit_v, args)
                    logit_t = allgather(logit_t, args)
                    logit_ivt = allgather(logit_ivt, args)

                    y_i = allgather(y_i, args)
                    y_v = allgather(y_v, args)
                    y_t = allgather(y_t, args)
                    y_ivt = allgather(y_ivt, args)


                if args.rank == 0:

                    mAPi.update(y_i.float().detach().cpu(), activation(logit_i).detach().cpu()) # Log metrics 
                    mAPv.update(y_v.float().detach().cpu(), activation(logit_v).detach().cpu()) # Log metrics 
                    mAPt.update(y_t.float().detach().cpu(), activation(logit_t).detach().cpu()) # Log metrics 
                    mAP.update(y_ivt.float().detach().cpu(), activation(logit_ivt).detach().cpu()) # Log metrics 
          
            if args.rank == 0:
                mAP.video_end() 
                mAPv.video_end()
                mAPt.video_end()
                mAPi.video_end()
    # compute the final mAP for all the test videos
    if args.rank == 0:
        if set_chlg_eval:
            mAP_i = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
            mAP_v = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
            mAP_t = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)
        else:
            mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
            mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
            mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
        mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
        mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
        mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval) 
        log('Test Results\nPer-category AP: ', args)
        log(f'I   : {mAP_i["AP"]}', args)
        log(f'V   : {mAP_v["AP"]}', args)
        log(f'T   : {mAP_t["AP"]}', args)
        log(f'IV  : {mAP_iv["AP"]}', args)
        log(f'IT  : {mAP_it["AP"]}', args)
        log(f'IVT : {mAP_ivt["AP"]}', args)
        log('-'*50, args)
        log(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', args)
        log(f':::::: : {mAP_i["mAP"]:.4f} | {mAP_v["mAP"]:.4f} | {mAP_t["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ', args)
        log('='*50, args)

        writer.add_scalar('MAP_i', mAP_i["mAP"], epoch)
        writer.add_scalar('MAP_v', mAP_v["mAP"], epoch)
        writer.add_scalar('MAP_t', mAP_t["mAP"], epoch)
        writer.add_scalar('MAP_iv', mAP_iv["mAP"], epoch)
        writer.add_scalar('MAP_it', mAP_it["mAP"], epoch)
        writer.add_scalar('MAP_ivt', mAP_ivt["mAP"], epoch)


    model.train()


def evaluate_zero_frame(writer, test_loaders_datasets, model, epoch, args):
    bert_type = args.valset_config.bert_type

    model.eval()

    with torch.no_grad():      
        for dataset_idx, vid_dataloaders in enumerate(test_loaders_datasets): 

            total_acc = []
            total_f1_phase = []
            total_f1_phase_class = []

            for test_loader in vid_dataloaders:
                preds_list = []
                labels_list = []
                prompt=args.zero_prompt[dataset_idx]
                log_prefix = args.log_prefix[dataset_idx]
                with open(prompt) as f:
                    lines = f.readlines()
                f.close()

                class_texts = [i.replace('\n', '') for i in lines]
                class_texts = process_text(bert_type, class_texts)

                text_features = model(None, class_texts, mode='text')['text_emb'].cuda()
                text_features /= text_features.norm(dim=-1, keepdim=True)

                for i, data in enumerate(test_loader): 
                    frames = data['video'].cuda() # (1, M, T, C, H, W)
                    B, C, H, W = frames.shape
                    frames = frames.view(-1, C, H, W)
                    image_features = model(frames, None, mode='video')['img_emb'] # (B*M*T, D)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) # (B, classes)
                    labels = data['label'].cuda()

                    preds_list.append(probs)
                    labels_list.append(labels)
                
                preds_list = torch.cat(preds_list, 0) # (bs, d)
                labels_list = torch.cat(labels_list, 0) # (bs, )

                if args.distributed:
                    pred_gather = allgather(preds_list, args)
                    label_gather = allgather(labels_list, args)
                else:
                    pred_gather = preds_list
                    label_gather = labels_list

                pred_gather = pred_gather[label_gather!=11]
                label_gather = label_gather[label_gather!=11]

                if args.rank == 0: # and args.distributed:

                    accuracy_vid = calc_accuracy(pred_gather, label_gather)
                    f1_class_vid, f1_average_vid = calc_f1(pred_gather, label_gather)

                    total_acc.append(accuracy_vid)
                    total_f1_phase.append(f1_average_vid)

            if args.rank == 0:
                f1_average = np.mean(np.asarray(total_f1_phase))
                accuracy = np.mean(np.asarray(total_acc))

                log(f'Top 1 Accuracy on zero-shot on {log_prefix}: {accuracy} {gpu_mem_usage()}', args)
                log('='*50, args)
                writer.add_scalar(f'Acc_a{log_prefix}', accuracy, epoch)

                log(f'F1 Macro Score on zero-shot on {log_prefix}: {f1_average} {gpu_mem_usage()}', args)
                writer.add_scalar(f'F1_{log_prefix}', f1_average, epoch)

        model.train()