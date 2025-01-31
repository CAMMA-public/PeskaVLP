import torch.nn as nn
import torch as th
import torch.nn.functional as F
from codes.registry import LOSSES

@LOSSES.register_module()
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = th.tensor(temperature).cuda()

    def forward(self, z_i, z_j):

        bs = z_i.shape[0]
        self.negatives_mask = (~th.eye(bs * 2, bs * 2, dtype=bool).cuda()).float()

        representations = th.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = th.diag(similarity_matrix, bs)
        sim_ji = th.diag(similarity_matrix, -bs)


        positives = th.cat([sim_ij, sim_ji], dim=0)

        nominator = th.exp(positives / self.temperature)
        denominator = self.negatives_mask * th.exp(similarity_matrix / self.temperature)

        loss_partial = -th.log(nominator / th.sum(denominator, dim=1))
        loss = th.sum(loss_partial) / (2 * bs)
        return loss