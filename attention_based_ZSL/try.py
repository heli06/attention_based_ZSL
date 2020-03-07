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
from steps import train
import torchvision.transforms as transforms 
import scipy.io as sio

# 指定使用的显卡，选利用率低的
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#'/media/shawn/data/Data/birds'
#'/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/birds'
parser.add_argument('--att_DIM',type=int,default = 1)
parser.add_argument('--att_hidDIM',type=int,default = 4)

args = parser.parse_args()
print(args)

att_model = AttModels.AttEncoder(args)
att_model.train()

att_input = np.ones([6,1], dtype=np.float32)
att_input = torch.from_numpy(att_input)

att_output = att_model(att_input)
print(att_output.size())