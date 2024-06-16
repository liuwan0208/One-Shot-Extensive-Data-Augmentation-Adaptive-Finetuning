import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())##---logai+Gi
    return F.softmax(y / temperature, dim=-1)##---Yi

def gumbel_softmax(logits, temperature = 5):
    """
    input:  [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)##---Yi
    shape = y.size()##[batch, layer_num, 2]
    _, ind = y.max(dim=-1)##[batch, layer_num, 1]~dim=-1:0/1
    y_hard = torch.zeros_like(y).view(-1, shape[-1])##[batchsize*layer_num,1]
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y
