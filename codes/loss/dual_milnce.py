import torch as th
import torch.nn as nn
from milnce import MILNCELoss
from codes.registry import LOSSES

@LOSSES.register_module()
class MILNCE_Dual(nn.Module):
    # 1 video --> multiple ASR 1
    # 1 video --> multiple ASR 2
    def __init__(self, ):
        super(MILNCE_Dual, self).__init__()

        self.mil_nce_loss_1 = MILNCELoss()
        self.mil_nce_loss_2 = MILNCELoss()

    def forward(self, video_embd, text_embd_aws, text_embd_whisper):
        loss_1 = self.mil_nce_loss_1(video_embd, text_embd_whisper)
        loss_2 = self.mil_nce_loss_1(video_embd, text_embd_aws)
    
        loss = (loss_1 + loss_2) / 2

        return loss


class Dual_MILNCELoss(th.nn.Module):
    # multi video --> multiple ASR
    def __init__(self, bs):
        super(Dual_MILNCELoss, self).__init__()
        self.bs = bs

    def forward(self, video_embd, text_embd):
        '''
        video_embd: (bs*n_f, d)
        text_embd: (bs*n_c, d)
        '''
        n_f = video_embd.shape[0] // self.bs
        n_c = text_embd.shape[0] // self.bs
        x = th.matmul(video_embd, text_embd.t())
        # x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        x = x.view(self.bs, n_f, self.bs, n_c).permute(0, 2, 1, 3).reshape(self.bs, self.bs, -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)