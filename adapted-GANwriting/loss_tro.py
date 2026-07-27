"""
loss_tro.py — unchanged from original GANwriting; works with any vocab.
"""

import torch
import Levenshtein as Lev
from load_data import vocab_size, tokens, num_tokens, index2letter


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class LabelSmoothing(torch.nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion  = torch.nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing  = smoothing
        self.size       = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.detach().unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.detach() == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)


log_softmax = torch.nn.LogSoftmax(dim=-1)
crit        = LabelSmoothing(vocab_size, tokens['PAD_TOKEN'], smoothing=0.4)


class CER:
    def __init__(self):
        self.ed  = 0
        self.len = 0

    def add(self, pred, gt):
        pred_label = torch.topk(pred, 1, dim=-1)[1].squeeze(-1).cpu().numpy()
        for i in range(pred_label.shape[0]):
            p = [x for x in pred_label[i].tolist() if x >= num_tokens]
            g = [x for x in gt[i].cpu().numpy().tolist() if x >= num_tokens]
            p_str = ''.join(index2letter.get(c - num_tokens, '') for c in p)
            g_str = ''.join(index2letter.get(c - num_tokens, '') for c in g)
            self.ed  += Lev.distance(p_str, g_str)
            self.len += max(len(g_str), 1)

    def fin(self):
        return 100.0 * self.ed / max(self.len, 1)
