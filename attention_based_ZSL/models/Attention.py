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

class ProjNet(nn.Module):
    def __init__(self, args):
        super(ProjNet, self).__init__()
        self.localFc = nn.Linear(args.img_outDIM, args.attn_embedDIM)
        self.globalFc = nn.Linear(args.img_outDIM, args.attn_embedDIM)

    def forward(self, local_img, global_img):
        local_img = local_img.view(local_img.size(0), local_img.size(1), -1).permute(0, 2, 1)
        local_img = self.localFc(local_img).permute(0, 2, 1)
        global_img = self.globalFc(global_img)
        return local_img, global_img

# att_features batch_size*150, 312, 128
# image_features batch_size*150, 128, 256
# d 128
def func_attention(query, context, args):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    context = context.transpose(1, 2)
    smooth = args.lambda_softmax

    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    # --> (batch*sourceL, queryL)
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)
    # --> (batch, sourceL, queryL)
    attn = attn.view(batch_size, sourceL, queryL)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def score_att2img(images, attributes, args):
    """
    Images: (n_image, n_regions, d) matrix of images
    attributes: (n_caption, max_n_word, d) matrix of attributes
    """
    weiContext, attn = func_attention(attributes, images, args)
    attributes = attributes.contiguous()
    weiContext = weiContext.contiguous()
    sim = cosine_similarity(attributes, weiContext, dim=2)
    if args.agg_func == 'LogSumExp':
        sim = torch.exp(args.lambda_lse * sim)
        sim = sim.sum(dim=1, keepdim=True)
        sim = torch.log(sim) / args.lambda_lse
    elif args.agg_func == 'Max':
        sim = sim.max(dim=1, keepdim=True)[0]
    elif args.agg_func == 'Sum':
        sim = sim.sum(dim=1, keepdim=True)
    elif args.agg_func == 'Mean':
        sim = sim.mean(dim=1, keepdim=True)
    else:
        raise ValueError("unknown aggfunc: {}".format(args.agg_func))

    return sim

def attention_loss(sim, labels, args):
    # equation 11
    gamma3 = args.gamma3
    sim = gamma3 * sim
    pred = sim.view(args.batch_size, -1)
    loss = F.cross_entropy(pred, labels)
    return loss