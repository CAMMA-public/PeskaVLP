from codes.loss.clip_infonce import ClipInfoCELoss
from codes.loss.ntxent import NTXentLoss
from codes.loss.milnce import MILNCELoss
from codes.loss.hard_contrastive import HardNegativeContrastiveLoss
import torch.nn as nn
from codes.registry import LOSSES
import torch.nn.functional as F
import torch

@LOSSES.register_module(name='hier_infonce_hardng')
class Abstract_Loss(nn.Module):
    def __init__(self, temperature=0.1, alpha_weight=0.5, anchor_beta=0.5, adjecent_beta=0.5, margin_anchor=0.1, margin_adjecant=0.1, negative_beta=1):
        super(Abstract_Loss, self).__init__()

        self.clip_info_loss = ClipInfoCELoss(temperature)
        self.alpha_weight = alpha_weight
        self.temperature = temperature

        self.anchor_beta = anchor_beta
        self.adjecent_beta = adjecent_beta
        self.margin_anchor = margin_anchor
        self.margin_adjecant = margin_adjecant

        self.negative_beta = negative_beta

    def forward(self, video_embd, text_embds, video_embd_frame=None, pos_step=None):
        '''
        video_embd: (bs, d)
        text_embds: (bs, 3, d)

        text_embds[:,0,:] (bs, d) -> anchor textual embedding
        text_embds[:,1,:] (bs, d) -> adjecent aggregated textual embedding
        text_embds[:,2,:] (bs, d) -> adjecent aggregated textual embedding in shuffled order (hard negative)
        '''

        emb_v = video_embd
        emb_t_positive_anchor = text_embds[:, 0, :]
        emb_t_positive_adjecant = text_embds[:, 1, :]
        emb_t_negative_adjecant = text_embds[:, 2, :]

        bs, d = emb_v.shape
        labels = torch.arange(bs).cuda()

        # Normalize Vectors
        emb_v = F.normalize(emb_v, p=2, dim=1)
        emb_t_positive_anchor = F.normalize(emb_t_positive_anchor, p=2, dim=1)
        emb_t_positive_adjecant = F.normalize(emb_t_positive_adjecant, p=2, dim=1)
        emb_t_negative_adjecant = F.normalize(emb_t_negative_adjecant, p=2, dim=1)

        # vision-to-positive anchor textual embedding
        logits_per_image = torch.matmul(emb_v, torch.transpose(emb_t_positive_anchor, 0, 1)) # (bs, bs)
        scores_positive_anchor = logits_per_image.diag()

        logits_per_text = torch.matmul(emb_t_positive_anchor, torch.transpose(emb_v,0, 1)) # (bs, bs)
        loss_i = F.cross_entropy(logits_per_image / self.temperature, labels)
        loss_t = F.cross_entropy(logits_per_text / self.temperature, labels)
        loss_positive_anchor = self.alpha_weight*loss_i + (1-self.alpha_weight)*loss_t


        # vision-to-positive adjecant textual embedding
        logits_per_image = torch.matmul(emb_v, torch.transpose(emb_t_positive_adjecant,0, 1)) # (bs, bs)
        scores_positive_adjecant = logits_per_image.diag()

        logits_per_text = torch.matmul(emb_t_positive_adjecant, torch.transpose(emb_v,0, 1))  # (bs, bs)
        loss_i = F.cross_entropy(logits_per_image / self.temperature, labels)
        loss_t = F.cross_entropy(logits_per_text / self.temperature, labels)
        loss_positive_adjecant = self.alpha_weight*loss_i + (1-self.alpha_weight)*loss_t


        # vision-to-negative adjecant textual embedding
        logits_per_image = torch.matmul(emb_v, torch.transpose(emb_t_negative_adjecant, 0, 1)) # (bs, bs)
        scores_negative_adjecant = logits_per_image.diag() # scores of negative video-text pairs



        loss_negative_anchor = torch.clamp((scores_negative_adjecant.sum() - scores_positive_anchor.sum()) / (scores_negative_adjecant.size(0) + scores_positive_anchor.size(0)), min=0)

        loss_negative_adjecant = torch.clamp(
            (scores_negative_adjecant.sum() - scores_positive_adjecant.sum()) / (scores_negative_adjecant.size(0) + scores_positive_adjecant.size(0)), min=0)

        # loss_negative_anchor = torch.sum(
        #   torch.clamp(
        #       scores_negative_adjecant + self.margin_anchor - scores_positive_anchor, min=0
        #   ) # minize the 
        # )
        # loss_negative_adjecant = torch.sum(
        #   torch.clamp(
        #       scores_negative_adjecant + self.margin_adjecant - scores_positive_adjecant, min=0
        #   )
        # )

        loss_negative = self.anchor_beta*loss_negative_anchor + self.adjecent_beta*loss_negative_adjecant

        return (loss_positive_anchor + loss_positive_adjecant + self.negative_beta*loss_negative) / 3