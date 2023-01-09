import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from math import exp
import numpy as np

import os
from os import path
from PIL import Image

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


class CTDataset(Dataset):
    def __init__(self, root_dir_h, root_dir_l, length, transform=None):
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
        self.img_list_l = os.listdir(self.data_root_l)
        self.img_list_h = os.listdir(self.data_root_h)
        self.img_list_l.sort()
        self.img_list_h.sort()
        self.img_list_l = self.img_list_l[0:length]
        self.img_list_h = self.img_list_h[0:length]
        self.transform = transform
        self.tensor_list = []
        
        for i in range(len(self.img_list_l)):
            rmax = 0
            rmin = 1
            image_target = read_correct_image(self.data_root_h + self.img_list_h[i])
            image_input = read_correct_image(self.data_root_l + self.img_list_l[i])
            input_file = self.img_list_l[i]
            assert (image_input.shape[0] == 512 and image_input.shape[1] == 512)
            assert (image_target.shape[0] == 512 and image_target.shape[1] == 512)
            cmax1 = np.amax(image_target)
            cmin1 = np.amin(image_target)
            image_target = rmin + ((image_target - cmin1) / (cmax1 - cmin1) * (rmax - rmin))
            assert ((np.amin(image_target) >= 0) and (np.amax(image_target) <= 1))
            cmax2 = np.amax(image_input)
            cmin2 = np.amin(image_input)
            image_input = rmin + ((image_input - cmin2) / (cmax2 - cmin2) * (rmax - rmin))
            assert ((np.amin(image_input) >= 0) and (np.amax(image_input) <= 1))
            mins = ((cmin1 + cmin2) / 2)
            maxs = ((cmax1 + cmax2) / 2)
            image_target = image_target.reshape((1, 512, 512))
            image_input = image_input.reshape((1, 512, 512))
            inputs_np = image_input
            targets_np = image_target
    
            inputs = torch.from_numpy(inputs_np)
            targets = torch.from_numpy(targets_np)
            inputs = inputs.type(torch.FloatTensor).to('cuda:0')
            targets = targets.type(torch.FloatTensor).to('cuda:0')
            sample = {
                  'vol':input_file,
                  'HQ': targets,
                  'LQ': inputs,
                  'max': maxs,
                  'min': mins}
            self.tensor_list.append(sample)
            print('device from tensor_list[1]', str(self.tensor_list[0]['HQ'].get_device()))
            
    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        # print("Dataloader idx: ", idx)
#         print('len tensor)list:', len(self.tensor_list))

        return self.tensor_list[idx]
