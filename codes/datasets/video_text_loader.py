# Copyright (c) University of Strasbourg, All Rights Reserved.
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import torch
from transformers import AutoTokenizer
from codes.datasets.utils import *
from codes.registry import DATASETS

@DATASETS.register_module(name='video_text')
class Video_Text_Loader(Dataset):
    """Video-Text loader outputs one video clip/frame to one/multi-source of ASR texts"""

    def __init__(
            self,
            csv, # name of the training samples
            video_root='', # path to videos folder
            caption_roots=[], # path of the different ASRs
            candidates=None, #Tensor (n,) inscending order [1, 3,...]
            min_time=4.0,
            fps=1,
            num_frames=16,
            size=224,
            crop_only=False,
            center_crop=True,
            random_flip=False,
            bert_type="",
            transforms=None,
            clip_min=4,
            clip_max=20
    ):
        assert isinstance(size, int)
        assert len(candidates) == len(caption_roots)
        self.csv = pd.read_csv(csv)
        self.video_root = video_root
        self.caption_roots = caption_roots
        self.candidates = candidates
        
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.crop_only = crop_only
        self.center_crop = center_crop
        self.random_flip = random_flip

        # the ASR with lower number setting are set to anchor
        assert (self.candidates > 0).any()
        self.candidates = self.candidates.tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)

        self.transforms = transforms 

    def __len__(self):
        return len(self.csv)

    def _get_video(self, video_path, start, end):
        # randomly select video clip duration upto clip_max seconds, no less than clip_min seconds
        clip_len = random.randint(self.clip_min, self.clip_max)


        # mid_seek = random.randint(start, end)
        # start_seek = max(0, mid_seek - (clip_len / 2))

        start_seek = random.randint(start, int(max(start, end - clip_len)))


        metadata = ffmpeg.probe(video_path)

        duration = metadata['streams'][0]['duration']
        if end > float(duration): start_seek = float(duration) - clip_len
        if start_seek < 0: start_seek = 0

        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=clip_len)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video.copy())
        video = video.permute(0, 3, 1, 2) # (T, C, H, W)
        assert video.shape[0] > self.num_frames

        ''' random sampling '''
        #indices = torch.randperm(video.shape[0])[:self.num_frames]
        #video = video.index_select(0, indices)


        ''' select from videos uniformly '''
        step = video.shape[0]  // self.num_frames
        video = video[::step,...]
        video = video[:self.num_frames,...] # (t, c, h, w)

        video = video / 255.0
        video = self.transforms(video)

        return video
    
    def _find_nearest_candidates(self, start, end, caption, num_texts):

        df1 = caption.loc[(caption['start']<=start)&(caption['end']>=start)]
        df2 = caption.loc[(caption['start']>=start)&(caption['end']<=end)]
        df3 = caption.loc[(caption['start']<=end)&(caption['end']>=end)]

        captions = pd.concat([df1, df2, df3], axis=0)['text'].values[:].tolist()

        pad = num_texts - len(captions)
        # pad > 0: need to pad
        # < 0: exclude captions
        # == 0: perfect
        return captions, pad


    def _get_text(self, caption, num_texts):
        num_texts -= 1
        cap = pd.read_csv(caption)
        ind = random.randint(0, len(cap) - num_texts - 1)
        start, end = cap['start'].values[ind], cap['end'].values[ind+num_texts]

        # One caption from the aws caption
        sents = cap['text'].values[ind:ind+num_texts+1].tolist()

        #TODO: May need to be improved for edge cases. 
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time 
        return sents, int(start), int(end)


    def __getitem__(self, idx):
        video_file = self.csv['video_path'][idx]
        video_id = video_file.split('.')[0]
        video_path = os.path.join(self.video_root, video_file)

        sents = []
        for idx, asr_root in enumerate(self.caption_roots):
            caption_path = os.path.join(asr_root, video_id + '.csv')
            if idx == 0:
                # return pairs from the first asr
                sents_center, start, end = self._get_text(caption_path, self.candidates[idx])
                sents.append(sents_center)

                # get video
                video = self._get_video(video_path, start, end)
            else:
                other_caption = pd.read_csv(caption_path)
                sents_other, pad = self._find_nearest_candidates(start, end, other_caption, self.candidates[idx])

                if len(other_caption) == 0:
                    sents_other = sents_center*pad
                else:
                    sents_other += sents_center*abs(pad)
                
                sents_other = sents_other[:self.candidates[idx]]
                sents.append(sents_other)

        final_dict = {'video':video}

        for idx, i in enumerate(sents):
            tokens =  self.tokenizer(i,             
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=77)
            final_dict['sents_'+str(idx)] = tokens
        return final_dict
