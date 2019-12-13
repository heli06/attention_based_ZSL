import argparse
import os
import pickle
import sys
import time
import torch
import random
import datetime
import pprint
import dateutil.tz
import numpy as np
from PIL import Image
from dataloaders.dataset import ZSLDataset
from models import  ImageModels, AttModels
from steps import train
import torchvision.transforms as transforms 
import scipy.io as sio

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#'/media/shawn/data/Data/birds'
#'/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/birds'
parser.add_argument('--data_path', type = str, default='../Bird_for_ZSL') #
parser.add_argument('--class_num',type = int, default= 200)
parser.add_argument('--exp_dir', type = str, default= 'outputs/baseline')
parser.add_argument("--resume", action="store_true", default=True,
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="adam",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--batch_size', '--batch_size', default=20, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--workers',default=0,type=int,help='number of worker in the dataloader')
parser.add_argument('--lr_A', '--learning-rate-attribute', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_I', '--learning-rate-image', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=100, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
    metavar='W', help='weight decay (default: 1e-4)')     #5e-7
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument('--CUDA',default=True, help='whether use GPU')
parser.add_argument('--gpu_id',type = int, default= 0)
parser.add_argument('--manualSeed',type=int,default= 200, help='manual seed')
parser.add_argument('--img_size',type=int,default = 244,help='image size')

parser.add_argument('--att_DIM',type=int,default = 312)
parser.add_argument('--att_hidDIM',type=int,default = 1600)

parser.add_argument('--Loss_cont',default = False)
parser.add_argument('--gamma_cont',default = 1.0)
parser.add_argument('--Loss_batch',default = True)
parser.add_argument('--gamma_batch',default = 1.0)
parser.add_argument('--Loss_dist',default = False)
parser.add_argument('--gamma_dist',default = 1.0)
parser.add_argument('--Loss_hinge',default = False)
parser.add_argument('--gamma_hinge',default = 1.0)

parser.add_argument('--smooth_gamma',type=float,default=10)

# 新增加的载入ZSL和GZSL文件的参数
parser.add_argument('--test_class_id',default = 'test_class_id.txt')
parser.add_argument('--test_class_attr',default = 'test_class_attribute_labels_continuous.txt')
parser.add_argument('--all_class_id',default = 'all_class_id.txt')
parser.add_argument('--all_class_attr',default = 'class_attribute_labels_continuous.txt')


args = parser.parse_args()

resume = args.resume

print(args)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.CUDA:    
    torch.cuda.manual_seed(args.manualSeed)  
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):   # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(args.manualSeed + worker_id)

# Get data loader
imsize = args.img_size
image_transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])


dataset = ZSLDataset(args.data_path, args,'train',
                        transform=image_transform)
dataset_test = ZSLDataset(args.data_path, args,'test',                        
                        transform=image_transform)

parser.add_argument('testset_len',type=int,default = dataset_test.__len__())

print('dataset.__len__(): ', dataset.__len__()) # 8821
print('dataset_test.__len__(): ', dataset_test.__len__()) # 2967
print('len(dataset_test.attributes)', len(dataset_test.attributes))
print('type(dataset_test.attributes[0]), len', type(dataset_test.attributes[0]), len(dataset_test.attributes[0]))
print('attributes[0].shape', dataset_test.attributes[0].shape)
print('class_id.__len__()', dataset_test.class_id.__len__())
# print('class_id', dataset_test.class_id)

# 本来想用参考代码里的划分，但出现list out of range错误，不用了
# split_file = os.path.join(args.data_path, 'att_splits.mat')
# matcontent = sio.loadmat(split_file)
# numpy array index starts from 0, matlab starts from 1
# trainval_loc = matcontent['trainval_loc'].squeeze() - 1
# np.random.shuffle(trainval_loc)
# test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
# np.random.shuffle(test_seen_loc)
# test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
# np.random.shuffle(test_unseen_loc)


# 使用pytorch的sampler，从原本的训练集里切出一部分作为x_test_seen
# train: (8821,)
# trainval: (7057,)
# test_seen: (1764,)
# test_unseen: (2967,)
n_train = 8821
split = 7058
indices = list(range(n_train))
random.shuffle(indices)

trainval_loc = np.asarray(indices[:split])
test_seen_loc = np.asarray(indices[split:])

indices_trainval = torch.from_numpy(trainval_loc.astype(np.int32))
indices_test_seen = torch.from_numpy(test_seen_loc.astype(np.int32))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_trainval)
test_seen_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_test_seen)
print(indices_trainval.size())
print(indices_test_seen.size())

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, sampler=train_sampler, 
    drop_last=True, num_workers=args.workers,worker_init_fn=worker_init_fn)

test_seen_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, sampler=test_seen_sampler, 
    drop_last=False, num_workers=args.workers,worker_init_fn=worker_init_fn)

test_unseen_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size,
    drop_last=False, shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn)

image_model = ImageModels.Resnet101()
att_model = AttModels.AttEncoder(args)

train(image_model, att_model, train_loader, test_seen_loader, test_unseen_loader, args)
    
