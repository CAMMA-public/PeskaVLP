import torch as th
import torch.nn as nn
from codes.registry import LOSSES

@LOSSES.register_module()
class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax

    def forward(self, imgs, caps):
        scores = th.mm(imgs, caps.t()) # (bs, bs)
        diag = scores.diag()

        # Reducing the score on diagonal so there are not selected as hard negative
        scores = scores - 2 * th.diag(scores.diag())

        sorted_cap, _ = th.sort(scores, 0, descending=True)
        sorted_img, _ = th.sort(scores, 1, descending=True)

        # Selecting the nmax hardest negative examples
        max_c = sorted_cap[: self.nmax, :]
        max_i = sorted_img[:, : self.nmax]

        # Margin based loss with hard negative instead of random negative
        neg_cap = th.sum(
            th.clamp(
                max_c + (self.margin - diag).view(1, -1).expand_as(max_c), min=0
            )
        )
        neg_img = th.sum(
            th.clamp(
                max_i + (self.margin - diag).view(-1, 1).expand_as(max_i), min=0
            )
        )

        loss = neg_cap + neg_img

        return loss