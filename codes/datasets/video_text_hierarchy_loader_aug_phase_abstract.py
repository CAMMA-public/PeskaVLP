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

@DATASETS.register_module(name='video_text_phase_aug_phase_abstract')
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
        assert len(self.caption_roots) == 1

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

        if 'step' in cap.columns:  # Check if 'step' column exists
            selected_column = 'text'# random.choice(['step', 'text'])  # Randomly select 'step' or 'text'
        else:
            selected_column = 'text'  # Default to 'text' column if 'step' doesn't exist
        
        sents = cap[selected_column].values[ind:ind+num_texts+1].tolist()
        # Randomly select a starting index for the text

        # sents = cap['text'].values[ind:ind+num_texts+1].tolist()

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
        phase_caption_path = os.path.join(self.phase_root, video_id + '.csv')
        # get abstract sentences
        sents_abstract, _, _ = self._get_text(phase_caption_path, 1)

        # get phase sentences
        caption_path = os.path.join(self.caption_roots[0], video_id + '.csv')
        phase_caption = pd.read_csv(caption_path)

        if 'step' in phase_caption.columns:  # Check if 'step' column exists
            selected_column = 'text' # random.choice(['step', 'text'])  # Randomly select 'step' or 'text'
        else:
            selected_column = 'text'  # Default to 'text' column if 'step' doesn't exist

        phase_captions = phase_caption[selected_column].values[:].tolist()
        phase_starts = phase_caption['start'].values[:].tolist()
        phase_ends = phase_caption['end'].values[:].tolist()

        assert len(phase_captions) == len(phase_starts)

        samples = [[phase_captions[i], phase_starts[i], phase_ends[i]] for i in range(len(phase_caption))]


        step = len(samples) // self.candidates[0]
        if step == 0: 
            # padding
            pad_len = self.candidates[0] - len(samples)
            mask = list(range(0, len(samples))) # mask is unpadded seq
            samples += [samples[-1]]*abs(pad_len)  # pad the last caption to tail of the sequence
            mask += [-1]*abs(pad_len) # pad -1 to the mask seq
            mask = torch.tensor(mask)
        else:
            indices = torch.tensor(list(range(0, len(samples), step))[:self.candidates[0]])
            samples = [samples[i] for i in indices]
            mask = torch.tensor(list(range(0, len(samples)))) # mask is truncated seq

        assert len(samples) == self.candidates[0], len(samples)

        sents = [i[0] for i in samples]
        starts = [i[1] for i in samples]
        ends = [i[2] for i in samples]
        assert len(sents) == len(starts)

        # evenly select frames from phase video clip
        videos = self._get_video(video_path, starts[0], ends[-1]) # (num_frame_clip, c, h, w)

        # store video
        final_dict = {'video':videos}

        # store position mask that use to mask out padded steps
        final_dict['pos_step'] = mask

        assert len(sents_abstract) == 1, sents_abstract
        assert sents_abstract[0] != '', sents_abstract

        # tokenize abstract text
        tokens = self.tokenizer(sents_abstract,             
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