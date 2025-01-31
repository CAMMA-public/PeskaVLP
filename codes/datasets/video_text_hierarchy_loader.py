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
import math

@DATASETS.register_module(name='video_text_phase')
class Video_Text_Loader(Dataset):
    """Video-Text loader outputs one video clip/frame to one/multi-source of ASR texts"""

    def __init__(
            self,
            csv, # name of the training samples
            video_root='', # path to videos folder
            phase_root='', # path of the different ASRs
            caption_roots=[''], # aws for now
            candidates=10, #Tensor (n,) inscending order [1, 3,...]
            num_frames_clip=4,
            size=224,
            crop_only=False,
            center_crop=True,
            random_flip=False,
            bert_type="",
            transforms=None,
            min_time=20,
            **kwargs,
    ):
        assert isinstance(size, int)
        self.csv = pd.read_csv(csv)
        self.video_root = video_root
        self.phase_root = phase_root
        self.caption_roots = caption_roots

        self.size = size
        self.num_frames_clip = num_frames_clip
        self.min_time = min_time
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        self.candidates = candidates
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)

        self.transforms = transforms 

    def __len__(self):
        return len(self.csv)

    def _get_video(self, video_path, start, end):

        #TODO: May need to be improved for edge cases. 
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time 

        start_seek = start
        clip_len = end - start

        metadata = ffmpeg.probe(video_path)

        duration = metadata['streams'][0]['duration']
        if end > float(duration): start_seek = float(duration) - clip_len
        if start_seek < 0: start_seek = 0

        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=clip_len)
            .filter('fps', fps=math.ceil(self.num_frames_clip/clip_len))
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
        video = temporal_sampling(video, 0, video.shape[0], self.num_frames_clip)

        video = video.permute(0, 3, 1, 2) # (T, C, H, W)
        assert video.shape[0] >= self.num_frames_clip

        video = video / 255.0
        video = self.transforms(video)

        return video

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

    
    def _find_nearest_candidates(self, start, end, caption):

        df1 = caption.loc[(caption['start']<=start)&(caption['end']>=start)]
        df2 = caption.loc[(caption['start']>=start)&(caption['end']<=end)]
        df3 = caption.loc[(caption['start']<=end)&(caption['end']>=end)]

        captions = pd.concat([df1, df2, df3], axis=0)['text'].values[:].tolist()
        starts = pd.concat([df1, df2, df3], axis=0)['start'].values[:].tolist()
        ends = pd.concat([df1, df2, df3], axis=0)['end'].values[:].tolist()

        assert len(captions) == len(starts)

        sample = [[captions[i], starts[i], ends[i]] for i in range(len(captions))]

        return sample # [[cap, start, end]]

    def __getitem__(self, idx):
        video_file = self.csv['video_path'][idx]
        video_id = video_file.split('.')[0]
        video_path = os.path.join(self.video_root, video_file)
        phase_caption_path = os.path.join(self.phase_root, video_id + '.csv')
        # get phase text
        sents_phase, start, end = self._get_text(phase_caption_path, 1)

        # get asr sentences
        samples = []
        for idx, asr_root in enumerate(self.caption_roots):
            caption_path = os.path.join(asr_root, video_id + '.csv')
            other_caption = pd.read_csv(caption_path)
            samples += self._find_nearest_candidates(start, end, other_caption) # [[cap, start, end], ...]
        
        pad = len(samples) - self.candidates[0]
        if pad > 0:
            samples = random.sample(samples, self.candidates[0])
        elif pad < 0:
            samples += [[sents_phase[0], start, end]]*abs(pad)
        elif pad==0:
            samples = samples
        else:
            raise NotImplementedError
        assert len(samples) == self.candidates[0], len(samples)


        sents = [i[0] for i in samples]
        starts = [i[1] for i in samples]
        ends = [i[2] for i in samples]
        assert len(sents) == len(starts)

        # evenly select frames from phase video clip
        videos = self._get_video(video_path, starts[0], ends[-1]) # (num_frame_clip, c, h, w)

        # store video
        final_dict = {'video':videos}

        assert len(sents_phase) == 1, sents_phase
        assert sents_phase[0] != '', sents_phase

        # tokenize abstract text
        tokens = self.tokenizer(sents_phase,             
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=200)
        final_dict['sents_summarize'] = tokens


        tokens = self.tokenizer(sents,             
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=200)
        final_dict['sents_0'] = tokens

        return final_dict
    
def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames