import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from codes.registry import LOSSES

@LOSSES.register_module()
class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None

    def forward(self, aug1_embed, aug2_embed, logit_scale=None):

        q_a = aug1_embed
        q_b = aug2_embed

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        batch_size = q_a.size(0)

        k_a, k_b = q_a, q_b


        labels = torch.arange(
                batch_size, device=q_a.device
            )
        masks = F.one_hot(labels, batch_size) * 1e9

        if logit_scale is None:
            logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
            logits_aa = logits_aa - masks
            logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
            logits_bb = logits_bb - masks
            logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
            logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau
        else:
            logits_aa = logit_scale * torch.matmul(q_a, k_a.transpose(0, 1))
            logits_aa = logits_aa - masks
            logits_bb = logit_scale * torch.matmul(q_b, k_b.transpose(0, 1))
            logits_bb = logits_bb - masks
            logits_ab = logit_scale * torch.matmul(q_a, k_b.transpose(0, 1))
            logits_ba = logit_scale * torch.matmul(q_b, k_a.transpose(0, 1))

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100 * correct / batch_size

        return {'loss': loss, 'ssl_loss': loss, 'ssl_acc': acc}