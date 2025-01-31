from codes.loss.clip_infonce import ClipInfoCELoss
from codes.loss.ntxent import NTXentLoss
from codes.loss.milnce import MILNCELoss
from codes.loss.hard_contrastive import HardNegativeContrastiveLoss
import torch.nn as nn
from codes.registry import LOSSES
import torch
from codes.loss.dtw import compute_dtw_loss as dtw
import torch.nn.functional as F

@LOSSES.register_module(name='dtw')
class Abstract_Loss(nn.Module):
    def __init__(self, dtw_beta, dtw_ratio, dtw_scale_factor, temperature=0.1, alpha_weight=0.5):
        super(Abstract_Loss, self).__init__()

        # self.clip_info_loss = ClipInfoCELoss(temperature)
        # self.ntx_loss = NTXentLoss(temperature, alpha_weight)
        # self.mil_nce_loss = MILNCELoss()

        self.dtw_beta = dtw_beta
        self.dtw_ratio = dtw_ratio
        self.dtw_scale_factor = dtw_scale_factor

    def forward(self, video_embd, text_embds, video_embd_frame=None, pos_step=None, logit_scale=None):
        '''
        video_embd: (bs, d)
        video_embd_frame: (bs, n_frames, d)
        text_embds: [embs_1, embs_2]
        embs_1 (bs, d)
        embs_2 (bs, n_candidate, d)
        '''
        # DTW with normal and reversed text seq
        embs_v = video_embd_frame
        embs_t = text_embds[1]
        embs_v = F.normalize(embs_v, p=2, dim=-1)
        embs_t = F.normalize(embs_t, p=2, dim=-1)

        loss_dtw, _ = dtw(
            embs_v, embs_t, pos_indices=pos_step, dtw_beta=self.dtw_beta, dtw_ratio=self.dtw_ratio, dtw_scale_factor=self.dtw_scale_factor, alignment_type='dtw_contrastive', cyclic_action=False)
        return loss_dtw