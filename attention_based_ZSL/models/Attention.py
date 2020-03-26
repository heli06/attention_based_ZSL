import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models
import copy
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        '''
        接收模型size和注意力头的数量
        '''
        super(MultiHeadAttention, self).__init__()
        self.num_heads = args.num_heads
        self.attention_type = args.attention_type
        self.num_units = args.out_DIM
        self.normalize = args.normalize

        self.conv1d = nn.Conv1d(args.out_DIM // args.num_heads, args.out_DIM, 1)
        self.qsLinear = nn.Linear(args.out_DIM // args.num_heads, self.num_units)
        self.ksLinear = nn.Linear(args.out_DIM // args.num_heads, self.num_units)
        self.vsLinear = nn.Linear(args.out_DIM // args.num_heads, self.num_units)
        self.addLinear = nn.Linear(args.out_DIM // args.num_heads, self.num_units)

        nn.init.xavier_uniform_(self.conv1d.weight.data)
        nn.init.xavier_uniform_(self.qsLinear.weight.data)
        nn.init.xavier_uniform_(self.ksLinear.weight.data)
        nn.init.xavier_uniform_(self.vsLinear.weight.data)
        nn.init.xavier_uniform_(self.addLinear.weight.data)

    def forward(self, query, value):
        if self.num_units % self.num_heads != 0:
            raise ValueError('Multi head attention requires that num_units '
                             'is a multiple of num_heads')

        q = query
        k = self.conv1d(value.permute(0, 2, 1)).permute(0, 2, 1)
        v = value
        qs, ks, vs = self._split_heads(q, k, v)
        if self.attention_type == 'mlp_attention':
            style_embeddings = self._mlp_attention(qs, ks, vs)
        elif self.attention_type == 'dot_attention':
            style_embeddings = self._dot_product(qs, ks, vs)
        else:
            raise ValueError('ERROR')
        return self._combine_heads(style_embeddings)

    def _dot_product(self, qs, ks, vs):
        qk = torch.matmul(qs, ks.permute(0,1,3,2))
        scale_factor = (self.num_units // self.num_heads) ** -0.5
        if self.normalize:
            qk *= scale_factor
        weights = F.softmax(qk)
        context = torch.matmul(weights, vs)
        return context

    def _mlp_attention(self, qs, ks, vs):
        num_units = qs.size()[-1]
        dtype = qs.dtype

        qs = self.qsLinear(qs)
        ks = self.ksLinear(ks)
        vs = self.vsLinear(vs)

        v = Variable(torch.randn(num_units))
        if self.normalize:
            g = Variable(torch.FloatTensor(math.sqrt(1. / num_units)))
            b = Variable(torch.zeros(num_units))
            normed_v = g * v * torch.rsqrt(torch.sum(torch.from_numpy(
                np.square(v.data.numpy()))))

            add = torch.sum(normed_v * F.tanh(ks + qs + b), -1, keepdim=True)
        else:
            add = torch.sum(v * F.tanh(ks + qs), -1, keepdim=True)

        add = self.addLinear(add)
        weights = F.softmax(add.permute(0, 1, 3, 2))
        context = torch.matmul(weights, vs)
        return context

    def _split_heads(self, q, k, v):
        qs = self._split_last_dimension(q, self.num_heads).permute(0, 2, 1, 3)
        ks = self._split_last_dimension(k, self.num_heads).permute(0, 2, 1, 3)
        v_shape = v.size()
        vs = v.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        return qs, ks, vs

    def _split_last_dimension(self, x, num_heads):
        x_shape = x.size()
        dim = x_shape[-1]
        assert dim % num_heads == 0
        return x.reshape(list(x_shape[:-1]) + [num_heads, dim // num_heads])

    def _combine_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x_shape = x.size()
        return x.reshape(list(x_shape[:-3]) + [self.num_heads * x_shape[-1]])