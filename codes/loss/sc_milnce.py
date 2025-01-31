import torch as th
import torch.nn as nn
import torch.nn.functional as F
from loss_base import safe_div
from codes.registry import LOSSES

@LOSSES.register_module()
class SC_MILNCELoss(nn.Module):
    def __init__(self, temperature=0.1, label_varience=10.0, embedding_size=512):
        super(SC_MILNCELoss, self).__init__()
        self.positive_type = "gauss"
        self.negative_type = "single_noself"
        self.temperature = temperature
        self.label_varience = label_varience
        self.embedding_size = embedding_size

    def forward(self, v_embs, t_embs, seq_lens, steps_v, steps_t):
        # v_embs (bs, n_f, d)
        # t_embs (bs, n_c, d)
        # seq_lens (bs, ), range (0, n_a) n_a >=n_f
        # steps_v (bs, n_f) --> timestamps from seq, max is len(vid)
        # steps_t (bs, n_c) --> timestamps from seq, max is len(vid)
        # masks (bs, n_f) 
        batch_size, num_frames, channels = v_embs.shape
        _, num_can, _ = t_embs.shape
        
        input_masks = th.ones((batch_size*num_frames, batch_size*num_can)).cuda() # (bs*n_f, bs*n_c)

        v_embs = v_embs.view(-1, channels) # (bs*n_f, d)
        t_embs = t_embs.view(-1, channels) # (bs*n_f, d)
        steps_v = steps_v.view(-1) # (bs*n_f, 1)
        steps_t = steps_t.view(-1) # (bs*n_c, 1)


        seq_lens_v = seq_lens.unsqueeze(-1).expand(batch_size, num_frames).contiguous().view(-1).float() # (bs*n_f, 1)
        seq_lens_t = seq_lens.unsqueeze(-1).expand(batch_size, num_can).contiguous().view(-1).float() # (bs*n_c, 1)

        logits = th.matmul(v_embs, t_embs.transpose(0,1)) / self.temperature # (bs*n_f, bs*n_c)
        distence = th.abs((steps_v.view(-1,1)/seq_lens_v.view(-1,1)*seq_lens_t.view(1,-1))-steps_t.view(1,-1)).cuda() # (bs*n_f, bs*n_c) - (1, bs*n_c)
        
        # distence = distence/seq_lens_t.cuda() #####????

        distence.masked_fill_((input_masks==0), 1e6) # (bs*n_f, bs*n_c)
        weight = th.ones_like(logits).cuda() # (bs*n_f, bs*n_c)
        # nn = torch.zeros_like(steps).long() # (bs*n_v*n_f, 1)

        # negative weight
        for b in range(batch_size):
            start_v = b*num_frames
            start_t = b*num_can
            end_v = (b+1)*num_frames
            end_t = (b+1)*num_can
            # nn[start:mid] = mid+torch.argmin(distence[start:mid,mid:end], dim=1) # first view of seq
            # nn[mid:end] = start+torch.argmin(distence[mid:end,start:mid], dim=1) # second view of seq
            # if "single" in self.negative_type:
            #     weight[start:end,:start].fill_(0)
            #     weight[start:end,end:].fill_(0)
            if "noself" in self.negative_type:
                weight[start_v:end_v,start_t:end_t] = 0
        weight.masked_fill_((input_masks==0), 1e-6)

        # positive weight
        label = th.zeros_like(logits) # (bs*n_f, bs*n_c)
        if self.positive_type == "gauss":
            pos_weight = th.exp(-th.square(distence)/(2*self.label_varience)).type_as(logits) # (bs*n_f, bs*n_c)
            # according to three sigma law, we can ignore the distence further than three sigma.
            # it may avoid the numerical unstablity and keep the performance.
            # pos_weight[(distence>3*np.sqrt(self.label_varience))] = 0
            for b in range(batch_size):
                start_v = b*num_frames
                start_t = b*num_can
                end_v = (b+1)*num_frames
                end_t = (b+1)*num_can
                cur_pos_weight = pos_weight[start_v:end_v,start_t:end_t] # (n_f, n_c)
                label[start_v:end_v,start_t:end_t] = safe_div(cur_pos_weight, cur_pos_weight.sum(dim=1, keepdim=True))

        exp_logits = th.exp(logits)
        sum_negative = th.sum(weight*exp_logits, dim=1, keepdim=True)

        loss = F.kl_div(th.log(safe_div(exp_logits, sum_negative) + 1e-6), label, reduction="none")
        loss = th.sum(loss*input_masks)
        loss = loss / th.sum(input_masks)
        
        return loss