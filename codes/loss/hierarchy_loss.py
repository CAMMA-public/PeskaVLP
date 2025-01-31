from codes.loss.clip_infonce import ClipInfoCELoss
from codes.loss.ntxent import NTXentLoss
from codes.loss.milnce import MILNCELoss
from codes.loss.hard_contrastive import HardNegativeContrastiveLoss
import torch.nn as nn
from codes.registry import LOSSES
import torch

@LOSSES.register_module()
class Abstract_Loss(nn.Module):
    def __init__(self, temperature=0.1, alpha_weight=0.75):
        super(Abstract_Loss, self).__init__()

        self.clip_info_loss = ClipInfoCELoss(temperature)
        self.ntx_loss = NTXentLoss(temperature, alpha_weight)
        self.mil_nce_loss = MILNCELoss()

    def forward(self, video_embd, text_embds):
        '''
        video_embd: (bs, d)
        text_embds: [embs_1, embs_2]
        embs_1 (bs, d)
        embs_2 (bs, n_candidate, d)
        '''
        # 
        loss_1 = self.clip_info_loss(video_embd, text_embds[0])
        loss_2 = self.ntx_loss(video_embd, text_embds[0])

        aggregate_text = torch.mean(text_embds[1], axis=1, keepdim=False)

        loss_3_infonce = self.clip_info_loss(video_embd, aggregate_text)

        return (loss_1 + loss_2 + loss_3_infonce) / 3