import os
import mmap
import io
import sys
import glob
# import h5py as h5
import numpy as np
from os import path
from PIL import Image
import concurrent.futures as cf
from random import shuffle
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

import cupy as cp
def read_correct_image(path):
    offset = 0
    ct_org = None
    with Image.open(path) as img:
        ct_org = np.float32(np.array(img))
        if 270 in img.tag.keys():
            for item in img.tag[270][0].split("\n"):
                if "c0=" in item:
                    loi = item.strip()
                    offset = re.findall(r"[-+]?\d*\.\d+|\d+", loi)
                    offset = (float(offset[1]))
    ct_org = ct_org + offset
    neg_val_index = ct_org < (-1024)
    ct_org[neg_val_index] = -1024
    return ct_org
class NumpyExternalSource(object):
    

    def __init__(self, hq_dir, lq_dir, batch_size, device_id, num_gpus):
#         self.images_dir = "../../data/images/"
        self.hq_dir = hq_dir + "/"
        self.lq_dir = hq_dir + "/"
        self.batch_size = batch_size
        self.img_list_all_l = os.listdir(self.lq_dir)
        self.img_list_all_h = os.listdir(self.hq_dir)
#         self.vgg_hq_img3 = os.listdir(self.data_root_h_vgg_3)
#         self.vgg_hq_img1 = os.listdir(self.data_root_h_vgg_1)

        self.img_list_all_l.sort()
        self.img_list_all_h.sort()
        
#         with open(self.images_dir + "file_list.txt", 'r') as f:
#             self.files = [line.rstrip() for line in f if line is not '']
        # whole data set size
        self.data_set_len = len(self.img_list_all_h) 
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.img_list_h = self.img_list_all_h[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.img_list_l = self.img_list_all_l[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        
        self.n = len(self.img_list_l)
        self.rmin = 0
        self.rmax = 0

    def __iter__(self):
        self.i = 0
        shuffle(self.img_list_l)
        shuffle(self.img_list_h)
        return self

    def __next__(self):
#         rmin 
        hq_bl = []
        lq_bl = []
        max_bl = []
        min_bl = []
        vol_bl = []
#         max_i
#         min_i

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            inputs_np = None
            targets_np = None
            image_input = read_correct_image(self.lq_dir + self.img_list_l[self.i % self.n])
            image_target = read_correct_image(self.hq_dir + self.img_list_h[self.i % self.n])
            cmax1 = np.amax(image_target)
            cmin1 = np.amin(image_target)
            image_target = self.rmin + ((image_target - cmin1) / (cmax1 - cmin1) * (self.rmax - self.rmin))
            assert ((np.amin(image_target) >= 0) and (np.amax(image_target) <= 1))
            cmax2 = np.amax(image_input)
            cmin2 = np.amin(image_input)
            image_input = self.rmin + ((image_input - cmin2) / (cmax2 - cmin2) * (self.rmax - self.rmin))
            assert ((np.amin(image_input) >= 0) and (np.amax(image_input) <= 1))
            mins = ((cmin1 + cmin2) / 2)
            maxs = ((cmax1 + cmax2) / 2)
            image_target = image_target.reshape((1, 512, 512))
            image_input = image_input.reshape((1, 512, 512))
            inputs_np = image_input
            targets_np = image_target
    
            inputs = torch.from_numpy(inputs_np)
            targets = torch.from_numpy(targets_np)
    
            inputs = inputs.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)
            
#             jpeg_filename, label = self.files[self.i % self.n].split(' ')
            hq_bl.append(targets)  # we can use numpy
            lq_bl.append(inputs) # or PyTorch's native tensors
            vol_bl.append(str(self.img_list_l[self.i % self.n]))
            max_bl.append(maxs)
            min_bl.append(mins)
            
            self.i += 1
        return ( hq_bl,lq_bl,max_bl, min_bl)

    def __len__(self):
        return self.data_set_len

    next = __next__
                          
def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        hq, lq, maxs , mins = fn.external_source(source=external_data, num_outputs=4)
#         images = fn.decoders.image(jpegs, device="mixed")
#         images = fn.resize(images, resize_x=240, resize_y=240)
#         output = fn.cast(images, dtype=types.UINT8)
#         print('hq' , hq[0])
        hq = hq.gpu()
        lq = lq.gpu()
#         hq_ig = hq.to(device_id)
#         lq_ig = lq.to(device_id)
        pipe.set_outputs(hq,lq,maxs,mins)
    return pipe