from codes.loss.clip_infonce import ClipInfoCELoss
from codes.loss.ntxent import NTXentLoss
from codes.loss.milnce import MILNCELoss
from codes.loss.hard_contrastive import HardNegativeContrastiveLoss
import torch.nn as nn
from codes.registry import LOSSES

@LOSSES.register_module()
class Combined_Loss(nn.Module):
    def __init__(self, temperature=0.1, alpha_weight=0.75, hard_margin=0.2, hard_nmax=1):
        super(Combined_Loss, self).__init__()

        self.clip_info_loss = ClipInfoCELoss(temperature)
        self.ntx_loss = NTXentLoss(temperature, alpha_weight)
        self.mil_nce_loss = MILNCELoss()
        # self.negative_contrastive = HardNegativeContrastiveLoss(nmax=hard_nmax, margin=hard_margin)

    def forward(self, video_embd, text_embds, logit_scale=None):
        '''
        text_embds: [embs_1, embs_2]
        embs_1 (bs*n_candidate_1, d)
        embs_2 (bs*n_candidate_1, d)
        '''
        # 
        loss_1 = self.clip_info_loss(video_embd, text_embds[0], logit_scale)
        loss_2 = self.ntx_loss(video_embd, text_embds[0], logit_scale)
        # negative
        # loss_4 = self.negative_contrastive(video_embd, text_embds[0])

        # 
        loss_3 = self.mil_nce_loss(video_embd, text_embds[1], logit_scale)

        return (loss_1 + loss_2 + loss_3) / 3 # + 0.01*loss_4
