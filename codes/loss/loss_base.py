import torch as th
import torch.nn as nn
from torch.autograd import Variable

def safe_div(a, b):
    out = a / b
    out[th.isnan(out)] = 0
    return out


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim.
    x1: (N, D)
    x2: (N, D)
    """
    w12 = th.sum(x1 * x2, dim)
    w1 = th.norm(x1, 2, dim)
    w2 = th.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()



def global_loss(cnn_code, rnn_code, eps=1e-8, temp3=10.0):
    '''
    cnn_code: (N, D)
    rnn_code: (N, D)
    '''

    batch_size = cnn_code.shape[0]
    labels = Variable(th.LongTensor(range(batch_size))).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = th.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = th.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = th.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = th.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1








# def build(args):
#     if args.loss == 'milnce':
#         criterion = MILNCELoss()
#     elif args.loss == 'infonce':
#         criterion = InfoNCE(args.temperature)
#     elif args.loss == 'ntx':
#         criterion = NTXentLoss(args.temperature)
#     elif args.loss == 'combine':
#         # InfoNCE + MILNCE
#         criterion = Combined_Loss(args.temperature, args.alpha_weight)
#     elif args.loss == 'milnce_milnce':
#         criterion = MILNCE_Dual()
#     else:
#         raise NotImplementedError
#     return criterion