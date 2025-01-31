from codes.loss.clip_infonce import ClipInfoCELoss
from codes.loss.ntxent import NTXentLoss
from codes.loss.milnce import MILNCELoss
from codes.loss.simclr import SIMCLRLoss
import torch.nn as nn
import torch
import numpy as np
from codes.registry import LOSSES

@LOSSES.register_module()
class SSL_VL_Loss_new_learn(nn.Module):
    def __init__(self, temperature=0.1, alpha_weight=0.75):
        super(SSL_VL_Loss_new_learn, self).__init__()

        self.temperature_clip = nn.Parameter(torch.ones([]) * temperature)
        self.temperature_nxten = nn.Parameter(torch.ones([]) *temperature)
        self.temperature_simclr = nn.Parameter(torch.ones([]) *temperature)

        self.clip_info_loss = ClipInfoCELoss(self.temperature_clip)
        self.ntx_loss = NTXentLoss(self.temperature_nxten, alpha_weight)
        self.mil_nce_loss = MILNCELoss()

        self.ssl_loss = SIMCLRLoss(self.temperature_simclr)


    def forward(self, video_embd, video_embd_aug1, video_embd_aug2, text_embds, logit_scale=None):
        '''
        text_embds: [embs_1, embs_2]
            embs_1 (bs, d)
            embs_2 (bs, n_candidate, d)
        video_embd_aug1 (bs, d)
        video_embd_aug2 (bs, d)
        '''

        # Image - Text Supervision
        loss_1_vl = self.clip_info_loss(video_embd, text_embds[0], logit_scale)
        loss_2_vl = self.ntx_loss(video_embd, text_embds[0], logit_scale)
        loss_3_vl = self.mil_nce_loss(video_embd, text_embds[1], logit_scale)
        loss_1 = (loss_1_vl + loss_2_vl + loss_3_vl) / 3

        # Augmented Image_1 - Text Supervision
        loss_1_vl = self.clip_info_loss(video_embd_aug1, text_embds[0], logit_scale)
        loss_2_vl = self.ntx_loss(video_embd_aug1, text_embds[0], logit_scale)
        loss_3_vl = self.mil_nce_loss(video_embd_aug1, text_embds[1], logit_scale)
        loss_2 = (loss_1_vl + loss_2_vl + loss_3_vl) / 3

        # Augmented Image_2 - Text Supervision
        loss_1_vl = self.clip_info_loss(video_embd_aug2, text_embds[0], logit_scale)
        loss_2_vl = self.ntx_loss(video_embd_aug2, text_embds[0], logit_scale)
        loss_3_vl = self.mil_nce_loss(video_embd_aug2, text_embds[1], logit_scale)
        loss_3 = (loss_1_vl + loss_2_vl + loss_3_vl) / 3

        # SSL loss between Augmented Image_1 & Image_2
        loss_ssl = self.ssl_loss(video_embd_aug1, video_embd_aug2)
        loss_4 = loss_ssl['loss']

        return (loss_1 + loss_2 + loss_3 + loss_4) / 4
