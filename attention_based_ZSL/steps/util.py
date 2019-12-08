import math
import pickle
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def normalizeFeature(x):	
    
    x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide\
    feature_norm = torch.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x / feature_norm.unsqueeze(-1)
    return feat

def triplet_loss(image_output, att_output, neg_samples,args):

    p_dist = 1 - torch.cosine_similarity(image_output,att_output,dim=1)
    n_dist = 1 - torch.cosine_similarity(image_output,neg_samples,dim=1)

    loss = (args.margin + p_dist - n_dist).sum()/(p_dist.shape[0])    
    loss = torch.max(torch.tensor(0).float().cuda(),loss)  

    return loss

# top k% negative mining
def negative_samples_mining(image_output,att_output,cls_id):
    img_f = normalizeFeature(image_output)
    att_f = normalizeFeature(att_output)
    sim_mat = img_f.mm(att_f.t())

    clss_m1 = cls_id.unsqueeze(-1).repeat(1,cls_id.shape[0])
    clss_m2 = clss_m1.t()

    mask = (clss_m1==clss_m2).bool().cuda()
    n_mask = (clss_m1!=clss_m2)
    sim_mat.data.masked_fill_(mask,-1.0)

    # sim_mat = sim_mat*mask
    sim, index = sim_mat.sort(1,descending=True)
    statistic = n_mask.int()
    num = statistic.sum(1)
    min_num = num.min()
    number = (min_num.float()*0.1).int()
    idx_i = np.random.randint(0,number,size=(image_output.shape[0]))
 
    for i in range(index.shape[0]):
        idx = index[i][idx_i[i]].cpu().numpy()
        if i==0:
            idxes = idx
        else:
            idxes = np.hstack((idxes,idx))   
    
    
    neg_samples = att_output[idxes]
    return neg_samples

# batch hardest negative mining
def hardest_negative_mining_pair(image_output,att_output,cls_id):
    img_f = normalizeFeature(image_output)
    att_f = normalizeFeature(att_output)
    sim_mat = img_f.mm(att_f.t())

    clss_m1 = cls_id.unsqueeze(-1).repeat(1,cls_id.shape[0])
    clss_m2 = clss_m1.t()

    mask = (clss_m1==clss_m2).bool().cuda()
    n_mask = (clss_m1!=clss_m2)
    sim_mat.data.masked_fill_(mask,-1.0)
    sim_mat_T = sim_mat.t()
    # sim_mat = sim_mat*mask
    sim, index = sim_mat.sort(1,descending=True)
    sim_t,index_t = sim_mat_T.sort(1,descending=True)
 
    for i in range(index.shape[0]):
        idx = index[i][0].cpu().numpy()
        idxt = index_t[i][0].cpu().numpy()
        if i==0:
            idxes = idx
            idxes_t = idxt
        else:
            idxes = np.hstack((idxes,idx))   
            idxes_t = np.hstack((idxes_t,idxt))
    
    
    neg_audio = att_output[idxes]
    neg_img = image_output[idxes_t]
    return neg_audio, neg_img

def hardest_negative_mining_single(image_output,cls_id):
    img_f = normalizeFeature(image_output)
    att_f = img_f
    sim_mat = img_f.mm(att_f.t())

    clss_m1 = cls_id.unsqueeze(-1).repeat(1,cls_id.shape[0])
    clss_m2 = clss_m1.t()

    mask = (clss_m1==clss_m2).bool().cuda()
    n_mask = (clss_m1!=clss_m2)
    sim_mat.data.masked_fill_(mask,-1.0)   
    # sim_mat = sim_mat*mask
    sim, index = sim_mat.sort(1,descending=True)   
 
    for i in range(index.shape[0]):
        idx = index[i][0].cpu().numpy()
        if i==0:
            idxes = idx
        else:
            idxes = np.hstack((idxes,idx))               
    
    
    neg_samples = image_output[idxes]   
    return neg_samples



# batch loss

def batch_loss(cnn_code, rnn_code, class_ids,args,eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    batch_size = args.batch_size
    labels = Variable(torch.LongTensor(range(batch_size)))
    labels = labels.cuda()   
    
    masks = []
    if class_ids is not None:
        class_ids =  class_ids.data.cpu().numpy()
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        masks = masks.to(torch.bool)
        if args.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * args.smooth_gamma

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def distribute_loss(img,audio):
    soft_image = F.softmax(img)
    soft_audio = F.softmax(audio)
    log_soft_image = F.log_softmax(img)
    log_soft_audio = F.log_softmax(audio)
    loss1 = soft_image.mul(log_soft_audio).sum(1).mean()*(-1.0)
    loss2 = soft_audio.mul(log_soft_image).sum(1).mean()*(-1.0)
    return loss1, loss2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10
