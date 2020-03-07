import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models
import copy
import math

def clones(module, N):
    '''
    产生 N 个独立的层
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, dropout=None):
    '''
    缩放后的点乘注意力
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        '''
        接收模型size和注意力头的数量
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 假设 d_v 等于 d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        '''
        参考图2
        '''
        nbatches = query.size(0)

        # 作 d_model => h * d_k 的线性映射
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(
                1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 将注意力机制分批次应用到所有映射后的向量
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)

        # 用view连接，作最后的线性映射
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears(-1)(x)