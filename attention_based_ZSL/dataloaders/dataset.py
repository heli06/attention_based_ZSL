from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time

from torch.utils.data.dataloader import default_collate


import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt


import os
import sys
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def get_imgs(img_path, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    return normalize(img)

# dataloader for the main programer
class ZSLDataset(data.Dataset):
    def __init__(self, data_dir, args, split='train',            
                 transform=None, target_transform=None):
        self.args = args
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:   #find 函数找不到时返回-1 对于bird采用bbox
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        if split == 'train':
            self.filenames = self.load_filenames(data_dir,'filenames_train.pickle')
            self.class_id = self.load_filenames(data_dir,'class_ids_train.pickle')
            self.attributes = self.load_filenames(data_dir,'attributes_train.pickle')
            
        elif split == 'test_seen':
            self.filenames = self.load_filenames(data_dir,'filenames_test_seen.pickle')
            self.class_id = self.load_filenames(data_dir,'class_ids_test_seen.pickle')
            self.attributes = self.load_filenames(data_dir,'attributes_test_seen.pickle')
        
        else:
            self.filenames = self.load_filenames(data_dir,'filenames_test.pickle')
            self.class_id = self.load_filenames(data_dir,'class_ids_test.pickle')
            self.attributes = self.load_filenames(data_dir,'attributes_test.pickle')

        # cacluate the sequence label for the whole dataset
        if self.split =='train':
            unique_id = np.unique(self.class_id)
            seq_labels = np.zeros(args.class_num)
            for i in range(unique_id.shape[0]):
                seq_labels[unique_id[i]-1]=i
            
            self.labels = seq_labels[np.array(self.class_id)-1]       
                    
        
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
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

    def load_filenames(self, data_dir, file_name):
        filepath = os.path.join(data_dir,file_name)
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)     #filenames中保存的形式'002.Laysan_Albatross/Laysan_Albatross_0044_784',
        return filenames


    def __getitem__(self, index):
        #
        # start = time.time()
        key = self.filenames[index]     #图像名称
        cls_id = self.class_id[index]    
        if self.split =='train':
            label = self.labels[index]
        else:
            label = cls_id  #        
        data_dir = self.data_dir
        
        if self.bbox is not None:
            bbox = self.bbox[key]
            # data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/%s' % (data_dir, key)
        imgs = get_imgs(img_name, bbox, self.transform, normalize=self.norm)
        
        attrs = self.attributes[index]
        
        return imgs, attrs, cls_id, key, label    #  处理完的图像 ，单词编号 ，单词数量（固定的cfg.TEXT.WORDS_NUM） ，类别ID，图像名称
        # else:
        #     return imgs, caps, cls_id, key
        #只需要输出imgs, caps, cls_id， key 即可

    def __len__(self):
        return len(self.filenames)





