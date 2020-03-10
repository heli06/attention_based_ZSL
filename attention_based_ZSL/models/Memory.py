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
    def __init__(self, num_heads=4, attention_type='mlp_attention',
                num_units=None, normalize=True, args):
        '''
        接收模型size和注意力头的数量
        '''
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.num_units = num_units
        self.normalize = normalize
        self.conv1d = nn.Conv1d(args.num_gst, args.num_gst, 2048)

    def forward(self, query, value):
        if self.num_units % self.num_heads != 0:
            raise ValueError('ERROR')

        q = query
        k = self.conv1d(value)
        v = value
        qs, ks, vs = self._split_heads(q, k, v)
        if self.attention_type == 'mlp_attention':
            style_embeddings = self._mlp_attention(qs, ks, vs)

        return self._combine_heads(style_embeddings)

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
        return x.reshape(x_shape[:-1] + [num_heads, dim // num_heads])

class MemoryFusion(nn.Module):
    def __init__(self, args):
        super(MemoryFusion, self).__init__()
        self.tokens = torch.normal(mean=0., std=0.5, size=(
            args.num_gst, args.style_embed_depth // args.num_heads))
        self.image_attn = MultiHeadedAttention(
            args.num_heads, args.style_attn_dim, args.style_att_type)
        self.attr_attn = MultiHeadedAttention(
            args.num_heads, args.style_attn_dim, args.style_att_type)

    def forward(self, z_image, z_attr, key, value):
        batch_size = z_image.size()[0]
        output_image = self.image_attn(
            z_image.unsqueeze(1), self.tokens.repeat(batch_size, 1, 1))
        output_attr = self.attr_attn(
            z_attr.unsqueeze(1), self.tokens.repeat(batch_size, 1, 1))