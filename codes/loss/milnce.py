import torch as th
from codes.registry import LOSSES

@LOSSES.register_module(name='milnce')
class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd, logit_scale=None):
        '''
        video_embd: (bs, d)
        text_embd: (bs*n_c, d)
        '''
        d = video_embd.shape[-1]
        text_embd = text_embd.view(-1, d)
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)