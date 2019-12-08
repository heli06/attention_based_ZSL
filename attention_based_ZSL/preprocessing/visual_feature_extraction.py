# -*- coding: utf-8 -*-
"""
Created on 23 Oct, 2019
extract visual feature and saved as .npy
these features are used for the pre-train of the speech-embedding
this extraion method follow the work of Langugage Learning using speech to image retrieval
it seems that this method is not useful for CUB dataset

"""
import os
import sys
sys.path.append('/media/shawn/data/Code/Mycode/Speech2Visual/Speech_visually_embedding_1.1d')
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import PIL.Image
import tables
import pandas as pd
from PIL import Image
# from dataloaders.dataset_SV import  Visual_data
from models import ImageModels
import pickle
import numpy as np
# this script uses a pretrained vgg16 model to extract the penultimate layer activations
# for images

def load_filenames(data_dir, split):      
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)     #filenames中保存的形式'002.Laysan_Albatross/Laysan_Albatross_0044_784',
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames


def load_bbox(data_dir):        
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

def get_imgs_tencos(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
                                    std = [0.5,0.5,0.5])
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])


    resize = transforms.Resize(355)
    tencrop = transforms.TenCrop(299)
    tens = transforms.ToTensor()

    img = resize(img)
    img = tencrop(img)
    # plt.figure()
    # for i in range(10):
    #     plt.subplot(2,5,i+1)
    #     plt.imshow(img[i])
    # plt.show()

def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
                                    std = [0.5,0.5,0.5])
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    
    tens = transforms.ToTensor()
    im = tens(img)    
    im = torch.autograd.Variable(im).cuda()
    im = im.unsqueeze(0)
    
    
    return im



############main###############
imsize = 299

extraction_way = 'whole'   # whole: without reize the image   Danny: extract 10 regions of on image

data_root = '/media/shawn/data/Data/birds'

if data_root.find('birds') != -1:   #find 函数找不到时返回-1 对于bird采用bbox
    lbbox = load_bbox(data_root)
else:
    lbbox = None

train_filenames = load_filenames(data_root, 'train')
test_filenames = load_filenames(data_root,'test')
filenames = train_filenames + test_filenames


modal = ImageModels.Inception_v3().cuda()

modal.eval()
i=0
for key in sorted(filenames):
    if lbbox is not None:
        bbox = lbbox[key]
        data_dir = '%s/CUB_200_2011' % data_root
    else:
        bbox = None
        data_dir = data_dir
    #
    img_name = '%s/images/%s.jpg' % (data_dir, key)
    
    
    
    if extraction_way =='Danny':
        img = get_imgs_tencos(img_name, imsize,
                        bbox)
        
        image_input = img.float().cuda()
        img_feature = modal(image_input).mean(0).squeeze()
    elif extraction_way == 'whole':
        img = get_imgs(img_name,bbox)
        image_input = img.float().cuda()
        img_feature = modal(image_input).mean(0).squeeze()

    save_path = '%s/images_npy/%s.npy' % (data_dir, key)
    save_file = os.path.split(save_path)[0]
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    feature = img_feature.unsqueeze(0)
    feat = feature.data.cpu().numpy()
    np.save(save_path,feat)
    i+=1
    if i%50==0:
        print('processing the %i-th images'%i)


