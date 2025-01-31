from codes.loss.clip_infonce import ClipInfoCELoss
from codes.loss.ntxent import NTXentLoss
from codes.loss.milnce import MILNCELoss
from codes.loss.simclr import SIMCLRLoss
import torch.nn as nn
from codes.registry import LOSSES

@LOSSES.register_module()
class SSL_VL_Loss(nn.Module):
    def __init__(self, temperature=0.1, alpha_weight=0.75):
        super(SSL_VL_Loss, self).__init__()

        self.clip_info_loss = ClipInfoCELoss(temperature)
        self.ntx_loss = NTXentLoss(temperature, alpha_weight)
        self.mil_nce_loss = MILNCELoss()

        self.ssl_loss = SIMCLRLoss(temperature)

    def forward(self, video_embd, video_embd_aug1, video_embd_aug2, text_embds):
        '''
        text_embds: [embs_1, embs_2]
        video_embd_aug1 (bs, d)
        video_embd_aug2 (bs, d)
        '''
        
        loss_1 = self.clip_info_loss(video_embd, text_embds[0])
        loss_2 = self.ntx_loss(video_embd, text_embds[0])
        # negative
        # loss_4 = self.negative_contrastive(video_embd, text_embds[0])

        
        loss_3 = self.mil_nce_loss(video_embd, text_embds[1])

        # SSL loss within video frames
        loss_ssl = self.ssl_loss(video_embd_aug1, video_embd_aug2)
        loss_4 = loss_ssl['loss']

        return (loss_1 + loss_2 + loss_3 + loss_4) / 4
