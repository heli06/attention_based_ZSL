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
from models import ModalityClassifier, ModalityTransformer, Attention
from models import RelationNet
from steps import traintest3
import torchvision.transforms as transforms 
import scipy.io as sio

# 指定使用的显卡，选利用率低的
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#'/media/shawn/data/Data/birds'
#'/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/birds'
parser.add_argument('--data_path', type = str, default='../Bird_for_ZSL')
parser.add_argument('--class_num',type = int, default= 200)
parser.add_argument('--train_class_num',type = int, default= 150)
parser.add_argument('--exp_dir', type = str, default= 'outputs/baseline')
parser.add_argument("--resume", action="store_true", default=True,
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="adam",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--batch_size', '--batch_size', default=32, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--workers',default=0,type=int,help='number of worker in the dataloader')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
    metavar='LR', help='initial learning rate')

parser.add_argument('--lr-decay', default=60, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
    metavar='W', help='weight decay (default: 1e-4)')     #5e-7
parser.add_argument("--n_epochs", type=int, default=60,
        help="number of maximum training epochs")
parser.add_argument('--CUDA',default=True, help='whether use GPU')
parser.add_argument('--gpu_id',type = int, default= 0)
parser.add_argument('--manualSeed',type=int,default= 200, help='manual seed')
parser.add_argument('--img_size',type=int,default = 244,help='image size')

parser.add_argument('--att_DIM',type=int, default = 312)
parser.add_argument('--att_hidDIM',type=int, default = 800)
parser.add_argument('--att_outDIM',type=int, default = 1024)

parser.add_argument('--img_outDIM',type=int, default = 1024)

parser.add_argument('--rel_hidDIM',type=int, default = 72)

parser.add_argument('--attn_embedDIM',type=int, default = 128)

parser.add_argument('--smooth_gamma_r',type=float,default=40.0)
parser.add_argument('--save_file',type=str,default='result.text')
parser.add_argument('--Loss_Attn',default = True)
parser.add_argument('--gamma_attn',default = 0.1)
parser.add_argument('--Loss_BCE',default = False)
parser.add_argument('--Loss_CE',default = False)
parser.add_argument('--Loss_cont',default = False)
parser.add_argument('--gamma_cont',default = 1.0)
parser.add_argument('--Loss_batch',default = False)
parser.add_argument('--gamma_batch',default = 1.0)
parser.add_argument('--Loss_dist',default = False)
parser.add_argument('--gamma_dist',default = 1.0)
parser.add_argument('--Loss_hinge',default = False)
parser.add_argument('--gamma_hinge',default = 1.0)

parser.add_argument('--smooth_gamma',type=float,default=10.0)

# attn_gamma
parser.add_argument('--attn_gamma_1',type=float, default=5.0)
parser.add_argument('--attn_gamma_2',type=float, default=5.0)
parser.add_argument('--attn_gamma_3',type=float, default=10.0)

# 新增加的载入ZSL和GZSL文件的参数
parser.add_argument('--train_class_id',default = 'train_class_id.txt')
parser.add_argument('--train_class_attr',default = 'train_class_attribute_labels_continuous.txt')
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
parser.add_argument('--modules_used', default = 'memory_fusion')

args = parser.parse_args()

resume = args.resume

print(args)

save_path = os.path.join('outputs',args.save_file)
with open(save_path, "a") as file:
    file.write(str(args.smooth_gamma_r))

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
image_transform_test = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize)])


dataset_train = ZSLDataset(args.data_path, args,'train_neg',
                        transform=image_transform)

dataset_test_seen = ZSLDataset(args.data_path, args,'test_seen',                        
                        transform=image_transform_test)

dataset_test = ZSLDataset(args.data_path, args,'test',                        
                        transform=image_transform_test)

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
att_model = AttModels.GRU(args)

traintest3.train3(image_model, att_model, Attention, train_loader, test_seen_loader, test_unseen_loader, args)
