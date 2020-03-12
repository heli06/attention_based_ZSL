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
from models import ImageModels, AttModels, ImageModels2
from models import ModalityClassifier, ModalityTransformer, Memory
from steps import train
import torchvision.transforms as transforms 
import scipy.io as sio

# 指定使用的显卡，选利用率低的
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'

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
parser.add_argument('--batch_size', '--batch_size', default=64, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--workers',default=0,type=int,help='number of worker in the dataloader')
parser.add_argument('--lr_A', '--learning-rate-attribute', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_I', '--learning-rate-image', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_M', '--learning-rate-modal', default=0.001, type=float,
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
parser.add_argument('--out_DIM',type=int,default = 2048)

parser.add_argument('--Loss_cont',default = False)
parser.add_argument('--gamma_cont',default = 1.0)
parser.add_argument('--Loss_batch',default = True)
parser.add_argument('--gamma_batch',default = 1.0)
parser.add_argument('--Loss_dist',default = False)
parser.add_argument('--gamma_dist',default = 1.0)
parser.add_argument('--Loss_hinge',default = False)
parser.add_argument('--gamma_hinge',default = 1.0)
parser.add_argument('--Loss_modal', default=True)
parser.add_argument('--gamma_modal', type=float, default=0.1)

parser.add_argument('--smooth_gamma',type=float,default=10)

# 新增加的载入ZSL和GZSL文件的参数
parser.add_argument('--test_class_id',default = 'test_class_id.txt')
parser.add_argument('--test_class_attr',default = 'test_class_attribute_labels_continuous.txt')
parser.add_argument('--all_class_id',default = 'all_class_id.txt')
parser.add_argument('--all_class_attr',default = 'class_attribute_labels_continuous.txt')

# trainset, test_seen_set, test_set
parser.add_argument('--train_set_len', type=int, default = 7051)
parser.add_argument('--test_seen_set_len', type=int, default = 1770)
parser.add_argument('--test_set_len', type=int, default = 2967)

# 启用的模块
# memory_fusion
# modal_classifier
# unused
parser.add_argument('--modules_used',default = 'memory_fusion')

# Memory Fusion 相关参数
parser.add_argument('--num_gst', type=int, default = 128)

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


dataset_train = ZSLDataset(args.data_path, args,'train',
                        transform=image_transform)

dataset_test_seen = ZSLDataset(args.data_path, args,'test_seen',                        
                        transform=image_transform)

dataset_test = ZSLDataset(args.data_path, args,'test',                        
                        transform=image_transform)

# dataset_train.len 7051
# dataset_test_seen 1770
# dataset_test 2967
print('dataset_train.len', dataset_train.__len__())
print('dataset_test_seen', dataset_test_seen.__len__())
print('dataset_test', dataset_test.__len__())

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    drop_last=True, shuffle=True, num_workers=args.workers,worker_init_fn=worker_init_fn)

test_seen_loader = torch.utils.data.DataLoader(
    dataset_test_seen, batch_size=args.batch_size,
    drop_last=False, num_workers=args.workers,worker_init_fn=worker_init_fn)

test_unseen_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size,
    drop_last=False, shuffle=False,num_workers=args.workers, worker_init_fn=worker_init_fn)

image_model = ImageModels.Resnet101()
# image_model = ImageModels2.Resnet101()
att_model = AttModels.AttEncoder(args)
mod_model = ModalityClassifier.ModalityClassifier(args)
mod_transformer = ModalityTransformer.ModalityTransformer(args)
memory_funsion = Memory.MemoryFusion(args)

train(image_model, att_model, mod_model, mod_transformer, memory_funsion, train_loader, test_seen_loader, test_unseen_loader, args)
    
