import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from math import exp
import numpy as np

# import os
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
    def __init__(self, root_dir_h, root_dir_l, length):
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
#         self.data_root_h_vgg_3 = root_hq_vgg3 + "/"
#         self.data_root_h_vgg_1 = root_hq_vgg1 + "/"

        self.img_list_l = os.listdir(self.data_root_l)
        self.img_list_h = os.listdir(self.data_root_h)
#         self.vgg_hq_img3 = os.listdir(self.data_root_h_vgg_3)
#         self.vgg_hq_img1 = os.listdir(self.data_root_h_vgg_1)

        self.img_list_l.sort()
        self.img_list_h.sort()
#         self.vgg_hq_img3.sort()
#         self.vgg_hq_img1.sort()

        self.img_list_l = self.img_list_l[0:length]
        self.img_list_h = self.img_list_h[0:length]
#         self.vgg_hq_img_list3 = self.vgg_hq_img3[0:length]
#         self.vgg_hq_img_list1 = self.vgg_hq_img1[0:length]
        self.sample = dict()

    def __len__(self):
        return len(self.img_list_l)

    def __getitem__(self, idx):
        # print("Dataloader idx: ", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs_np = None
        targets_np = None
        rmin = 0
        rmax = 1

        # print("HQ", self.data_root_h + self.img_list_h[idx])
        # print("LQ", self.data_root_l + self.img_list_l[idx])
        # image_target = read_correct_image("/groups/synergy_lab/garvit217/enhancement_data/train/LQ//BIMCV_139_image_65.tif")
        # print("test")
        # exit()
        image_target = read_correct_image(self.data_root_h + self.img_list_h[idx])
        # print("low quality {} ".format(self.data_root_h + self.img_list_h[idx]))
        # print("high quality {}".format(self.data_root_h + self.img_list_l[idx]))
        # print("hq vgg b3 {}".format(self.data_root_h_vgg + self.vgg_hq_img_list[idx]))
        image_input = read_correct_image(self.data_root_l + self.img_list_l[idx])
        # vgg_hq_img3 = np.load(self.data_root_h_vgg_3 + self.vgg_hq_img_list3[idx]) ## shape : 1,256,56,56
        # vgg_hq_img1 = np.load(self.data_root_h_vgg_1 + self.vgg_hq_img_list1[idx]) ## shape : 1,64,244,244

        input_file = self.img_list_l[idx]  ## low quality image
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

        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)

        # vgg_hq_b3 =  torch.from_numpy(vgg_hq_img3)
        # vgg_hq_b1 =  torch.from_numpy(vgg_hq_img1)
        #
        # vgg_hq_b3 = vgg_hq_b3.type(torch.FloatTensor)
        # vgg_hq_b1 = vgg_hq_b1.type(torch.FloatTensor)

        # print("hq vgg b3 {} b1 {}".format(vgg_hq_b3.shape , vgg_hq_b1.shape))
        self.sample = {'vol': input_file,
                  'HQ': targets,
                  'LQ': inputs,
                  # 'HQ_vgg_op':vgg_hq_b3, ## 1,256,56,56
                  # 'HQ_vgg_b1': vgg_hq_b1,  ## 1,256,56,56
                  'max': maxs,
                  'min': mins}
        return self.sample