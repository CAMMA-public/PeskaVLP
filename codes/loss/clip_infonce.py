import torch.nn as nn
import torch as th
import torch.nn.functional as F
from codes.registry import LOSSES

@LOSSES.register_module(name='clip_infonce')
class ClipInfoCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ClipInfoCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb_v, emb_t, logit_scale=None):
        '''
        emb_v: (bs, d)
        emb_t: (bs, d)
        '''

        bs, d = emb_v.shape
        labels = th.arange(bs).cuda()

        emb_v = F.normalize(emb_v, p=2, dim=1)
        emb_t = F.normalize(emb_t, p=2, dim=1)

        if logit_scale is None:

            logits_per_image = th.matmul(emb_v, th.transpose(emb_t,0, 1)) / self.temperature # (bs, bs)
            logits_per_text = th.matmul(emb_t, th.transpose(emb_v,0, 1)) / self.temperature # (bs, bs)

        else:
            logits_per_image = logit_scale * th.matmul(emb_v, th.transpose(emb_t,0, 1))
            logits_per_text = logit_scale * th.matmul(emb_t, th.transpose(emb_v,0, 1))


        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i+loss_t)/2
        return loss
    