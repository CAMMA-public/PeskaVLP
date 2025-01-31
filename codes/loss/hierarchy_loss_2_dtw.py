from codes.loss.clip_infonce import ClipInfoCELoss
from codes.loss.ntxent import NTXentLoss
from codes.loss.milnce import MILNCELoss
from codes.loss.hard_contrastive import HardNegativeContrastiveLoss
import torch.nn as nn
from codes.registry import LOSSES
import torch
from codes.loss.dtw import compute_dtw_loss as dtw
import torch.nn.functional as F

@LOSSES.register_module(name='hier_infonce_dtw')
class Abstract_Loss(nn.Module):
    def __init__(self, dtw_beta, dtw_ratio, dtw_scale_factor, temperature=0.1, alpha_weight=0.5, reg_weight=0.1):
        super(Abstract_Loss, self).__init__()

        self.clip_info_loss = ClipInfoCELoss(temperature)
        self.ntx_loss = NTXentLoss(temperature, alpha_weight)
        self.mil_nce_loss = MILNCELoss()

        self.dtw_beta = dtw_beta
        self.dtw_ratio = dtw_ratio
        self.dtw_scale_factor = dtw_scale_factor

        self.reg_weight = reg_weight

    def forward(self, video_embd, text_embds, video_embd_frame=None, pos_step=None, logit_scale=None):
        '''
        video_embd: (bs, d)
        video_embd_frame: (bs, n_frames, d)
        text_embds: [embs_1, embs_2]
        embs_1 (bs, d)
        embs_2 (bs, n_candidate, d)
        '''
        loss_1 = self.clip_info_loss(video_embd, text_embds[0])
        loss_2 = self.ntx_loss(video_embd, text_embds[0])

        aggregate_text = torch.mean(text_embds[1], axis=1, keepdim=False)

        loss_3_infonce = self.clip_info_loss(video_embd, aggregate_text)

        loss_4_infonce = self.clip_info_loss(text_embds[0], aggregate_text)

        # DTW with normal and reversed text seq
        embs_v = video_embd_frame
        embs_t = text_embds[1]
        embs_v = F.normalize(embs_v, p=2, dim=-1)
        embs_t = F.normalize(embs_t, p=2, dim=-1)

        loss_dtw, _ = dtw(
            embs_v, embs_t, pos_indices=pos_step, dtw_beta=self.dtw_beta, dtw_ratio=self.dtw_ratio, dtw_scale_factor=self.dtw_scale_factor, alignment_type='dtw_contrastive', cyclic_action=False)

        return (loss_1 + loss_2 + loss_3_infonce + loss_4_infonce + self.reg_weight*loss_dtw) / 5