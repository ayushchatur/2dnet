#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/31/2020 1:43 PM
# @Author : Zhicheng Zhang
# @E-mail : zhicheng0623@gmail.com
# @Site :
# @File : train_main.py
# @Software: PyCharm
import torchvision
from torchvision import transforms
import collections
from collections import OrderedDict
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import path
from PIL import Image
from matplotlib import pyplot as plt
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from collections import defaultdict
from torchvision import models as torchmodels
# vizualize_folder = "./visualize"
# loss_folder = "./loss"
# reconstructed_images = "reconstructed_images"
    # os.environ['MASTER_ADDR'] = '10.21.10.4'
INPUT_CHANNEL_SIZE = 1




def init_env_variable():
    job_id=""

    try:
        job_id = os.environ['SLURM_JOBID']
    except:
        job_id=""

    print("slurm jobid: " + str(job_id))
    vizualize_folder = vizualize_folder + jobid
    loss_folder = loss_folder + jobid
    reconstructed_images = reconstructed_images + jobid

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

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


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.001:
            min_val = -0.1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (batch, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    if (torch.isnan(output)):
        print("NAN")
        print(pow1)
        print(pow2)
        print(ssims)
        print(mcs)
        exit()

    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize="simple")
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class VGGloss(torch.nn.Module):
    def __init__(self,weight=None, size_average=True):
        super(VGGloss,self).__init__()

        ## Target block 1, Target block
    def forward(self, input, target):
        assert(input.shape[0] == target.shape[0])
        loss_value = torch.mean(torch.abs(torch.sub(input,
                                                     target)))  # enhanced image : [1, 256, 56, 56] dim should be same (1,256,56,56)
        return loss_value



class denseblock(nn.Module):
    def __init__(self, nb_filter=16, filter_wh=5):
        super(denseblock, self).__init__()
        self.input = None  ######CHANGE
        self.nb_filter = nb_filter
        self.nb_filter_wh = filter_wh
        ##################CHANGE###############
        self.conv1_0 = nn.Conv2d(in_channels=nb_filter,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_0 = nn.Conv2d(in_channels=self.conv1_0.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_1 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_1.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_2 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv1_2.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_3 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels,out_channels=self.nb_filter*4,kernel_size=1)
        self.conv2_3 = nn.Conv2d(in_channels=self.conv1_3.out_channels, out_channels=self.nb_filter, kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1 = [self.conv1_0, self.conv1_1, self.conv1_2, self.conv1_3]
        self.conv2 = [self.conv2_0, self.conv2_1, self.conv2_2, self.conv2_3]

        self.batch_norm1_0 = nn.BatchNorm2d(nb_filter)
        self.batch_norm2_0 = nn.BatchNorm2d(self.conv1_0.out_channels)
        self.batch_norm1_1 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels)
        self.batch_norm2_1 = nn.BatchNorm2d(self.conv1_1.out_channels)
        self.batch_norm1_2 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels)
        self.batch_norm2_2 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.batch_norm1_3 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels)
        self.batch_norm2_3 = nn.BatchNorm2d(self.conv1_3.out_channels)

        self.batch_norm1 = [self.batch_norm1_0, self.batch_norm1_1, self.batch_norm1_2, self.batch_norm1_3]
        self.batch_norm2 = [self.batch_norm2_0, self.batch_norm2_1, self.batch_norm2_2, self.batch_norm2_3]


    #def `Forward`(self, inputs):
    def forward(self, inputs):                      ######CHANGE
        #x = self.input
        x = inputs
        # for i in range(4):
        #    #conv = nn.BatchNorm2d(x.size()[1])(x)
        #    conv = self.batch_norm1[i](x)
        #    #if(self.conv1[i].weight.grad != None ):
        #    #    print("weight_grad_" + str(i) + "_1", self.conv1[i].weight.grad.max())
        #    conv = self.conv1[i](conv)      ######CHANGE
        #    conv = F.leaky_relu(conv)

        #    #conv = nn.BatchNorm2d(conv.size()[1])(conv)
        #    conv = self.batch_norm2[i](conv)
        #    #if(self.conv2[i].weight.grad != None ):
        #    #    print("weight_grad_" + str(i) + "_2", self.conv2[i].weight.grad.max())
        #    conv = self.conv2[i](conv)      ######CHANGE
        #    conv = F.leaky_relu(conv)
        #    x = torch.cat((x, conv),dim=1)

        conv_1 = self.batch_norm1_0(x)
        conv_1 = self.conv1_0(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_0(conv_1)
        conv_2 = self.conv2_0(conv_2)
        conv_2 = F.leaky_relu(conv_2)

        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_1(x)
        conv_1 = self.conv1_1(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_1(conv_1)
        conv_2 = self.conv2_1(conv_2)
        conv_2 = F.leaky_relu(conv_2)

        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_2(x)
        conv_1 = self.conv1_2(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_2(conv_1)
        conv_2 = self.conv2_2(conv_2)
        conv_2 = F.leaky_relu(conv_2)

        x = torch.cat((x, conv_2), dim=1)
        conv_1 = self.batch_norm1_3(x)
        conv_1 = self.conv1_3(conv_1)
        conv_1 = F.leaky_relu(conv_1)
        conv_2 = self.batch_norm2_3(conv_1)
        conv_2 = self.conv2_3(conv_2)
        conv_2 = F.leaky_relu(conv_2)
        x = torch.cat((x, conv_2), dim=1)

        return x


# class vgg16(nn.Module):

class DD_net(nn.Module):
    def __init__(self):
        super(DD_net, self).__init__()
        resize = True
        self.input = None  #######CHANGE
        self.nb_filter = 16
        blocks = []
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[9:18].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1))
        self.resize = resize

        ##################CHANGE###############
        self.conv1 = nn.Conv2d(in_channels=INPUT_CHANNEL_SIZE, out_channels=self.nb_filter, kernel_size=(7, 7), padding = (3,3))
        self.dnet1 = denseblock(self.nb_filter,filter_wh=5)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels*5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet2 = denseblock(self.nb_filter,filter_wh=5)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels*5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet3 = denseblock(self.nb_filter, filter_wh=5)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels*5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet4 = denseblock(self.nb_filter, filter_wh=5)

        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels*5, out_channels=self.nb_filter, kernel_size=(1, 1))

        self.convT1 = nn.ConvTranspose2d(in_channels=self.conv4.out_channels + self.conv4.out_channels,out_channels=2*self.nb_filter,kernel_size=5, padding=(2, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=self.convT1.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT3 = nn.ConvTranspose2d(in_channels=self.convT2.out_channels + self.conv3.out_channels,out_channels=2*self.nb_filter,kernel_size=5, padding=(2, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=self.convT3.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT5 = nn.ConvTranspose2d(in_channels=self.convT4.out_channels + self.conv2.out_channels,out_channels=2*self.nb_filter,kernel_size=5, padding=(2, 2))
        self.convT6 = nn.ConvTranspose2d(in_channels=self.convT5.out_channels,out_channels=self.nb_filter,kernel_size=1)
        self.convT7 = nn.ConvTranspose2d(in_channels=self.convT6.out_channels + self.conv1.out_channels,out_channels=2*self.nb_filter,kernel_size=5, padding=(2, 2))
        self.convT8 = nn.ConvTranspose2d(in_channels=self.convT7.out_channels,out_channels=1 ,kernel_size=1)
        self.batch1 = nn.BatchNorm2d(1)
        self.max1 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.batch2 = nn.BatchNorm2d(self.nb_filter*5)
        self.max2 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.batch3 = nn.BatchNorm2d(self.nb_filter*5)
        self.max3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
        self.batch4 = nn.BatchNorm2d(self.nb_filter*5)
        self.max4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
        self.batch5 = nn.BatchNorm2d(self.nb_filter*5)

        self.batch6 = nn.BatchNorm2d(self.conv5.out_channels + self.conv4.out_channels)
        self.batch7 = nn.BatchNorm2d(self.convT1.out_channels)
        self.batch8 = nn.BatchNorm2d(self.convT2.out_channels + self.conv3.out_channels)
        self.batch9 = nn.BatchNorm2d(self.convT3.out_channels)
        self.batch10 = nn.BatchNorm2d(self.convT4.out_channels + self.conv2.out_channels)
        self.batch11 = nn.BatchNorm2d(self.convT5.out_channels)
        self.batch12 = nn.BatchNorm2d(self.convT6.out_channels + self.conv1.out_channels)
        self.batch13 = nn.BatchNorm2d(self.convT7.out_channels)

    # def Forward(self, inputs):
    def forward(self, inputs, target):

        self.input = inputs
        # print("Size of input: ", inputs.size())
        # conv = nn.BatchNorm2d(self.input)
        conv = self.batch1(self.input)  #######CHANGE
        # conv = nn.Conv2d(in_channels=conv.get_shape().as_list()[1], out_channels=self.nb_filter, kernel_size=(7, 7))(conv)
        conv = self.conv1(conv)  #####CHANGE
        c0 = F.leaky_relu(conv)

        p0 = self.max1(c0)
        D1 = self.dnet1(p0)

        #######################################################################################
        conv = self.batch2(D1)
        conv = self.conv2(conv)
        c1 = F.leaky_relu(conv)

        p1 = self.max2(c1)
        D2 = self.dnet2(p1)
        #######################################################################################

        conv = self.batch3(D2)
        conv = self.conv3(conv)
        c2 = F.leaky_relu(conv)

        p2 = self.max3(c2)
        D3 = self.dnet3(p2)
        #######################################################################################

        conv = self.batch4(D3)
        conv = self.conv4(conv)
        c3 = F.leaky_relu(conv)  ## c3.out_channel = 16

        # p3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)(c3)
        p3 = self.max4(c3)  ######CHANGE
        D4 = self.dnet4(p3)

        conv = self.batch5(D4)
        conv = self.conv5(conv)
        c4 = F.leaky_relu(conv) ## c4.out_channel= 16

        x = torch.cat((nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)(c4), c3),dim=1) # c4=16*2, c3=16, => x = 16*3 (channels)
        dc4 = F.leaky_relu(self.convT1(self.batch6(x)))         ######size() CHANGE ; d4 : in=16*2, out=16*2
        dc4_1 = F.leaky_relu(self.convT2(self.batch7(dc4))) ; #dc4_1 : in = 16*2 out = 16

        x = torch.cat((nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)(dc4_1), c2),dim=1) # dc4_1 = 16*2, c2 =16 => x = 16*3
        dc5 = F.leaky_relu(self.convT3(self.batch8(x)))
        dc5_1 = F.leaky_relu(self.convT4(self.batch9(dc5)))

        x = torch.cat((nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)(dc5_1), c1),dim=1)
        dc6 = F.leaky_relu(self.convT5(self.batch10(x)))
        dc6_1 = F.leaky_relu(self.convT6(self.batch11(dc6)))

        x = torch.cat((nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)(dc6_1), c0),dim=1)
        dc7 = F.leaky_relu(self.convT7(self.batch12(x)))
        dc7_1 = F.leaky_relu(self.convT8(self.batch13(dc7)))
        output = dc7_1

        return output , target

def gen_visualization_files(outputs, targets, inputs, file_names, val_test, maxs, mins):
    mapped_root = "./visualize/" + val_test + "/mapped/"
    diff_target_out_root = "./visualize/" + val_test + "/diff_target_out/"
    diff_target_in_root = "./visualize/" + val_test + "/diff_target_in/"
    ssim_root = "./visualize/" + val_test + "/ssim/"
    out_root = "./visualize/" + val_test + "/"
    in_img_root = "./visualize/" + val_test + "/input/"
    out_img_root = "./visualize/" + val_test + "/target/"

    # if not os.path.exists("./visualize"):
    #     os.makedirs("./visualize")
    # if not os.path.exists(out_root):
    #    os.makedirs(out_root)
    # if not os.path.exists(mapped_root):
    #    os.makedirs(mapped_root)
    # if not os.path.exists(diff_target_in_root):
    #    os.makedirs(diff_target_in_root)
    # if not os.path.exists(diff_target_out_root):
    #    os.makedirs(diff_target_out_root)
    # if not os.path.exists(in_img_root):
    #    os.makedirs(in_img_root)
    # if not os.path.exists(out_img_root):
    #    os.makedirs(out_img_root)

    MSE_loss_out_target = []
    MSE_loss_in_target = []
    MSSSIM_loss_out_target = []
    MSSSIM_loss_in_target = []
    SSIM_loss_in_target = []
    SSIM_loss_out_target = []

    outputs_size = list(outputs.size())
    # num_img = outputs_size[0]
    (num_img, channel, height, width) = outputs.size()
    for i in range(num_img):
        # output_img = outputs[i, 0, :, :].cpu().detach().numpy()
        output_img = outputs[i, 0, :, :].cpu().detach().numpy()
        target_img = targets[i, 0, :, :].cpu().numpy()
        input_img = inputs[i, 0, :, :].cpu().numpy()

        output_img_mapped = (output_img * (maxs[i].item() - mins[i].item())) + mins[i].item()
        target_img_mapped = (target_img * (maxs[i].item() - mins[i].item())) + mins[i].item()
        input_img_mapped = (input_img * (maxs[i].item() - mins[i].item())) + mins[i].item()

        # target_img = targets[i, 0, :, :].cpu().numpy()
        # input_img = inputs[i, 0, :, :].cpu().numpy()

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(target_img_mapped)
        cv2.write(out_img_root + file_name, im )

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(input_img_mapped)
        #im.save(in_img_root + file_name)
        cv2.write(in_img_root+file_name , im)
        # jy
        # im.save(folder_ori_HU+'/'+file_name)

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(output_img_mapped)
  #      im.save(mapped_root + file_name)
        cv2.save(mapped_root + file_name , im)
        # jy
        # im.save(folder_enh_HU+'/'+file_name)

        difference_target_out = (target_img - output_img)
        difference_target_out = np.absolute(difference_target_out)
        fig = plt.figure()
        plt.imshow(difference_target_out, cmap='gray')
        plt.colorbar()
        plt.clim(0, 0.2)
        plt.axis('off')
        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        fig.savefig(diff_target_out_root + file_name)
        plt.clf()
        plt.close()

        difference_target_in = (target_img - input_img)
        difference_target_in = np.absolute(difference_target_in)
        fig = plt.figure()
        plt.imshow(difference_target_in, cmap='gray')
        plt.colorbar()
        plt.clim(0, 0.2)
        plt.axis('off')
        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        fig.savefig(diff_target_in_root + file_name)
        plt.clf()
        plt.close()

        output_img = torch.reshape(outputs[i, 0, :, :], (1, 1, height, width))
        target_img = torch.reshape(targets[i, 0, :, :], (1, 1, height, width))
        input_img = torch.reshape(inputs[i, 0, :, :], (1, 1, height, width))

        MSE_loss_out_target.append(nn.MSELoss()(output_img, target_img))
        MSE_loss_in_target.append(nn.MSELoss()(input_img, target_img))
        MSSSIM_loss_out_target.append(1 - MSSSIM()(output_img, target_img))
        MSSSIM_loss_in_target.append(1 - MSSSIM()(input_img, target_img))
        SSIM_loss_out_target.append(1 - SSIM()(output_img, target_img))
        SSIM_loss_in_target.append(1 - SSIM()(input_img, target_img))

    with open(out_root + "msssim_loss_target_out", 'a') as f:
        for item in MSSSIM_loss_out_target:
            f.write("%f\n" % item)

    with open(out_root + "msssim_loss_target_in", 'a') as f:
        for item in MSSSIM_loss_in_target:
            f.write("%f\n" % item)

    with open(out_root + "ssim_loss_target_out", 'a') as f:
        for item in SSIM_loss_out_target:
            f.write("%f\n" % item)

    with open(out_root + "ssim_loss_target_in", 'a') as f:
        for item in SSIM_loss_in_target:
            f.write("%f\n" % item)

    with open(out_root + "mse_loss_target_out", 'a') as f:
        for item in MSE_loss_out_target:
            f.write("%f\n" % item)

    with open(out_root + "mse_loss_target_in", 'a') as f:
        for item in MSE_loss_in_target:
            f.write("%f\n" % item)


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
        self.sample = dict()
#         self.tensor_list = []
        
#         for i in range(len(self.img_list_l)):
            
#             print('device from tensor_list[1]', str(self.tensor_list[0]['HQ'].get_device()))
            
    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        # print("Dataloader idx: ", idx)
#         print('len tensor)list:', len(self.tensor_list))
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs_np = None
        targets_np = None
        rmin = 0
        rmax = 1
#         rmax = 0
#         rmin = 1
            
        image_target = Image.open(self.data_root_h + self.img_list_h[i])
        image_input  = Image.open(self.data_root_l + self.img_list_l[i])
        image_target = np.asarray( image_target, dtype=np.uint8 )
        image_input = np.asarray( image_input, dtype=np.uint8 )
        dims = len(image_target)

        image_target = image_target[0:dims,0:dims,0]
            
        input_file = self.img_list_l[i]
        assert (image_input.shape == image_target.shape )
#            assert ( == 512 and image_target.shape[1] == 512)
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
        image_target = image_target.reshape((1, dims, dims))
        image_input = image_input.reshape((1, dims, dims))
        inputs_np = image_input
        targets_np = image_target
    
        inputs = torch.from_numpy(inputs_np)
        targets = torch.from_numpy(targets_np)
#            inputs = inputs.type(torch.FloatTensor).to('cuda:0')
#            targets = targets.type(torch.FloatTensor).to('cuda:0')
        print('input shape ', inputs.shape)
        print('input shape ', targets.shape)
        sample = {
              'vol':input_file,
              'HQ': targets,
              'LQ': inputs,
              'max': maxs,
              'min': mins}
#         self.tensor_list.append(sample)

        return sample

# jy
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def generate_plots(epochs):
    try:
        files = ['./train_loss.png', './val_loss.png', './both_loss.png']
        for f in files:
            if os.path.isfile(f):
                os.remove(f)  # Opt.: os.system("rm "+strFile)
        rank = 0
        train_mse_list = np.load('loss/train_MSE_loss_' + str(rank) + ".npy").mean(axis=1).tolist()
        loss_b1_list = np.load('loss/train_loss_b1_' + str(rank) + ".npy").mean(axis=1).tolist()
        loss_b3_list = np.load('loss/train_loss_b3_' + str(rank) + ".npy").mean(axis=1).tolist()
        loss_total_list = np.load('loss/train_total_loss_' + str(rank) + ".npy").mean(axis=1).tolist()

        val_mse_list = np.load('loss/val_MSE_loss_' + str(rank) + ".npy").mean(axis=1).tolist()
        val_loss_b1_list = np.load('loss/val_loss_b1_' + str(rank) + ".npy").mean(axis=1).tolist()
        val_loss_b3_list = np.load('loss/val_loss_b3_' + str(rank) + ".npy").mean(axis=1).tolist()
        val_loss_total_list = np.load('loss/val_total_loss_' + str(rank) + ".npy").mean(axis=1).tolist()

        x_axis = range(epochs)
        plt.figure(num=1)
        plt.plot(x_axis, train_mse_list, label="mse loss", marker='o')
        plt.plot(x_axis, loss_b1_list, label="loss_b1", marker='o')
        plt.plot(x_axis, loss_b3_list, label="loss_b3", marker='o')
        plt.plot(x_axis, loss_total_list, label="total_loss", marker='*')
        plt.xlabel("epochs")
        plt.ylabel("values (fp)")
        plt.legend()
        plt.title('Training loss vs epoch')
        plt.savefig('train_loss.png', format='png', dpi=300)
        plt.figure(num=2)
        plt.plot(x_axis, val_mse_list, label="val loss", marker='o')
        plt.plot(x_axis, val_loss_b1_list, label="val loss_b1", marker='o')
        plt.plot(x_axis, val_loss_b3_list, label="val loss_b1", marker='o')
        plt.plot(x_axis, val_loss_total_list, label="val total_loss", marker='*')
        plt.xlabel("epochs")
        plt.ylabel("values (fp)")
        plt.legend()
        plt.title('Validation loss vs epoch')
        plt.savefig('val_loss.png', format='png', dpi=300)
        plt.figure(num=3)
        plt.plot(x_axis, loss_total_list, label="train loss", marker='o')
        plt.plot(x_axis, val_loss_total_list, label="validate loss", marker='*')
        plt.xlabel("epochs")
        plt.ylabel("values (fp)")
        plt.legend()
        plt.title('loss vs epoch')
        plt.savefig('both_loss.png', format='png', dpi=300)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('exception occurred saving graphs :', type(e), e)


def dd_train(gpu, args):
    rank = args.nr * args.gpus + gpu
    beta = 0.5
    beta = args.beta
    print('beta: ', beta)

    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    batch = args.batch
    epochs = args.epochs
    print('batch: ', batch, 'epochs: ', epochs)
    root_train_h = "/projects/synergy_lab/ayush/brain/HQ/"
    root_train_l = "/projects/synergy_lab/ayush/brain/LQ/"
    
#     root_val_h = "projects/synergy_lab/ayush/brain/val/HQ/"
#     root_val_l = "projects/synergy_lab/ayush/brain/val/LQ/"

#     root_test_h = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
#     root_test_l = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"


    #root = add
    trainset = CTDataset(root_dir_h=root_train_h, root_dir_l=root_train_l, , length=1251)
    train_data, vali_data = random_split(trainset, [851, 400])
    valida_data, test_data = random_split(vali_data, [200, 200])
    
#     testset = CTDataset(root_dir_h=root_val_h, root_dir_l=root_val_l,, length=784)
#     valset = CTDataset(root_dir_h=root_test_h, root_dir_l=root_test_l, length=784)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, num_replicas=args.world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valida_data, num_replicas=args.world_size, rank=rank)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    train_loader = DataLoader(trainset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=test_sampler)
    val_loader = DataLoader(valset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size, pin_memory=False, sampler=val_sampler)
    #train_loader = DataLoader(trainset, num_workers=world_size, pin_memory=False, batch_sampler=train_sampler)
    #test_loader = DataLoader(testset, zbatch_size=batch, drop_last=False, shuffle=False)
    #val_loader = DataLoader(valset, batch_size=batch, drop_last=False, shuffle=False)

    model = DD_net()

    # torch.cuda.set_device(rank)
    # model.cuda(rank)
    model.to(gpu)
    model = DDP(model, device_ids=[gpu])
    learn_rate = 0.0001;
    epsilon = 1e-8

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, eps=epsilon)  #######ADAM CHANGE
    # optimizer1 = torch.optim.Adam(model.dnet1.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    # optimizer2 = torch.optim.Adam(model.dnet2.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    # optimizer3 = torch.optim.Adam(model.dnet3.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    # optimizer4 = torch.optim.Adam(model.dnet4.parameters(), lr=learn_rate, eps=epsilon)     #######ADAM CHANGE
    decayRate = 0.95
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer1, gamma=decayRate)
    # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer2, gamma=decayRate)
    # scheduler3 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer3, gamma=decayRate)
    # scheduler4 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer4, gamma=decayRate)

    # outputs = ddp_model(torch.randn(20, 10).to(rank))

    # max_train_img_init = 5120;
    # max_train_img_init = 32;
    # max_val_img_init = 784;
    # max_val_img_init = 16;
    # max_test_img = 784;

    train_epoch_loss = defaultdict(list)
    train_MSE_loss = defaultdict(list)
    train_loss_b1 = defaultdict(list)
    train_loss_b3 = defaultdict(list)
    train_total_loss = defaultdict(list)

    val_MSE_loss = defaultdict(list)
    val_MSSI_loss_b1 = defaultdict(list)
    val_MSSI_loss_b3 = defaultdict(list)
    val_total_loss = defaultdict(list)

    test_MSE_loss = []
    test_loss_b1 = []
    test_loss_b3 = []
    test_total_loss = []

    # test_MSE_loss = [0]
    # test_loss_b1 =[0]
    # test_loss_b3 =[0]
    # test_total_loss =[0]

    loss_b1_list = defaultdict(list)
    loss_b3_list = defaultdict(list)
    loss_total_list = defaultdict(list)
    train_mse_list = defaultdict(list)
    val_loss_b1_list = defaultdict(list)
    val_loss_b3_list = defaultdict(list)
    val_loss_total_list = defaultdict(list)
    val_mse_list = defaultdict(list)

    start = datetime.now()

    model_file = "weights_" + str(epochs) + "_" + str(batch) + ".pt"

    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}

    print("~~~~~~~~~~~~~~~~~~~~~~~~~ training ~~~~~~~~~~~~~~~~~~~~~~")
    if (not (path.exists(model_file))):

        for k in range(epochs):
            train_sampler.set_epoch(epochs)
            for batch_index, batch_samples in enumerate(train_loader):

                file_name, HQ_img, LQ_img, maxs, mins,   = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], batch_samples['max'], batch_samples['min']
                lq_image = LQ_img.to(gpu) ## low quality image
                hq_image = HQ_img.to(gpu) ## high quality target image

                enhanced_image  = model(lq_image,hq_image)  # vgg_en_image should be 1,256,56,56

                MSE_loss = nn.MSELoss()(enhanced_image, hq_image)  # shold already nbe same dimension
                MSSSIM_loss = 1 - MSSSIM()(enhanced_image, hq_image)
                total_train_loss = MSE_loss + (beta * MSSSIM_loss)


                train_MSE_loss[k].append(MSE_loss.item())
                train_total_loss[k].append(total_train_loss.item())

                model.zero_grad()  # zero the gradients
                total_train_loss.backward()  # back propogation
                optimizer.step()  # update the parameters


            print('total training loss:', (sum(train_total_loss[k]) / len(train_total_loss[k])))
            print('training  mse:', sum(train_MSE_loss[k]) / len(train_MSE_loss[k]))
            scheduler.step()  #

            print("~~~~~~~~~~~~~Validation~~~~~~~~~~~~~~~~")
            for batch_index, batch_samples in enumerate(val_loader):
                file_name, HQ_img, LQ_img, maxs, mins   = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], batch_samples['max'], batch_samples['min']

                lq_image = LQ_img.to(gpu)
                hq_image = HQ_img.to(gpu)

                enhanced_image = model(lq_image, hq_image)

                MSE_loss = nn.MSELoss()(enhanced_image, hq_image)  # should already nbe same dimension
                MSSSIM_loss = 1 - MSSSIM()(enhanced_image, hq_image)


                val_loss = MSE_loss + (beta * MSSSIM_loss)
                val_MSE_loss[k].append(MSE_loss.item())

                val_total_loss[k].append(val_loss.item())

                if (k == epochs - 1):
                    if (rank == 0):
                        print("Training complete in: " + str(datetime.now() - start))
                    outputs_np = enhanced_image.cpu().detach().numpy()
                    (batch_size, channel, height, width) = enhanced_image.size()
                    for m in range(batch_size):
                        file_name1 = file_name[m]
                        file_name1 = file_name1.replace(".IMA", ".tif")
                        im = Image.fromarray(outputs_np[m, 0, :, :])
                        #im.save('reconstructed_images/val/' + file_name1)
                        cv2.write('reconstructed_images/val/' + file_name1, im)
                    # gen_visualization_files(outputs, targets, inputs, val_files[l_map:l_map+batch], "val")
                    gen_visualization_files(enhanced_image, hq_image, lq_image, file_name, "val", maxs, mins)
                # vali.update(1)
            print('total validation loss:', sum(val_total_loss[k]) / len(val_total_loss[k]))
            print('validation  mse:', sum(val_MSE_loss[k]) / len(val_MSE_loss[k]))

        print("train and val end.............")
        if (rank == 0):
            print("Saving model parameters")
            torch.save(model.state_dict(), model_file)
            try:
                print('serializing losses')
                np.save('loss/train_MSE_loss_' + str(rank), np.array([v for k, v in train_MSE_loss.items()]))
                np.save('loss/train_total_loss_' + str(rank), np.array([v for k, v in train_total_loss.items()]))

                np.save('loss/val_MSE_loss_' + str(rank), np.array([v for k, v in val_MSE_loss.items()]))
                np.save('loss/val_total_loss_' + str(rank), np.array([v for k, v in val_total_loss.items()]))
            except Exception as e:
                print('error serializing: ', e)
    else:
        print("Loading model parameters")
        model.load_state_dict(torch.load(model_file, map_location=map_location))

    test_MSSSIM_loss = []
    test_SSIM_loss = []
    print("~~~~~~~~~~~Testing~~~~~~~~~~~~~~~")
    for batch_index, batch_samples in enumerate(test_loader):
        file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                                                batch_samples['max'], batch_samples['min']
        lq_image = LQ_img.to(gpu)
        hq_image = HQ_img.to(gpu)

        enhanced_image, out_vgg_b3, out_vgg_b1, tar_b3, tar_b1 = model(lq_image, hq_image)

        MSE_loss = nn.MSELoss()(enhanced_image, hq_image)
        msssim_1 = 1 - MSSSIM()(enhanced_image, hq_image)
        ssim_1 = 1 - SSIM()(enhanced_image,hq_image)
        loss = MSE_loss + (beta * msssim_1)
        print("MSE_loss", MSE_loss.item())
        print("Total_loss", loss.item())
        print("MSSSIM loss", msssim_1.item())
        print("SSIM Loss", ssim_1.item())
        print("beta: ", beta)
        print("====================================")
        # test_MSE_loss
        test_MSE_loss.append(MSE_loss.item())
        test_total_loss.append(loss.item())
        test_MSSSIM_loss.append(msssim_1.item())
        test_SSIM_loss.append(ssim_1.item())
        outputs_np = enhanced_image.cpu().detach().numpy()
        (batch_size, channel, height, width) = enhanced_image.size()
        for m in range(batch_size):
            file_name1 = file_name[m]
            file_name1 = file_name1.replace(".IMA", ".tif")
            im = Image.fromarray(outputs_np[m, 0, :, :])
            cv2.imwrite('reconstructed_images/test/' + file_name1 , im)
        # outputs.cpu()
        # targets_test[l_map:l_map+batch, :, :, :].cpu()
        # inputs_test[l_map:l_map+batch, :, :, :].cpu()
        # gen_visualization_files(outputs, targets, inputs, test_files[l_map:l_map+batch], "test" )
        gen_visualization_files(enhanced_image, hq_image, lq_image, file_name, "test", maxs, mins)

    if (rank == 0):
        print("Saving model parameters")
        # torch.save(model.state_dict(), model_file)
        try:
            print('serializing test losses')
            np.save('loss/test_MSE_loss_' + str(rank), np.array(test_MSE_loss))
            np.save('loss/test_loss_b1_' + str(rank), np.array(test_loss_b1))
            np.save('loss/test_loss_b3_' + str(rank), np.array(test_loss_b3))
            np.save('loss/test_total_loss_' + str(rank), np.array(test_total_loss))
            np.save('loss/test_loss_mssim_' + str(rank), np.array(test_MSSSIM_loss))
            np.save('loss/test_loss_ssim_'+ str(rank), np.array(test_SSIM_loss))
        except Exception as e:
            print('error serializing: ', e)

    print("testing end")


    print("~~~~~~~~~~~~~~~~~~ everything completed ~~~~~~~~~~~~~~~~~~~~~~~~")

    data2 = np.loadtxt('./visualize/test/msssim_loss_target_out')
    print("size of out target: " + str(data2.shape))

    # print("size of append target: " + str(data3.shape))
    print("Final avergae MSE: ", np.average(data2), "std dev.: ", np.std(data2))
    print("Final average MSSSIM LOSS: " + str(100 - (100 * np.average(test_MSSSIM_loss))), 'std dev : ', np.std(test_MSSSIM_loss))
    print("Final average SSIM LOSS: " + str(100 - (100 * np.average(test_SSIM_loss))),'std dev : ', np.std(test_SSIM_loss))
    generate_plots(epochs)
    psnr_calc(data2)


def psnr_calc(mse_t):
    try:
        psnr = []
        for i in range(len(mse_t)):
            #     x = read_correct_image(pa +"/"+ ll[i])
            mse_sqrt = pow(mse_t[i], 0.5)
            psnr_ = 20 * np.log10(1 / mse_sqrt)
            psnr.insert(i, psnr_)
        print('psnr: ', np.mean(psnr), ' std dev', np.std(psnr))
    except Exception as e:
        print("error psrn calc" , e)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--beta', default=0.5, type=float,metavar='b',
                        help='beta value')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--epochs', default=2, type=int, metavar='e',
                        help='number of total epochs to run')
    parser.add_argument('--batch', default=2, type=int, metavar='a',
                        help='number of batch per gpu')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    # init_env_variable()
    args.nr = int(os.environ['SLURM_PROCID'])
    print("SLURM_PROCID: " + str(args.nr))
    # world_size = 4
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_ADDR'] = '10.21.10.4'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['MASTER_PORT'] = '8888'
    mp.spawn(dd_train,
             args=(args,),
             nprocs=args.gpus,
             join=True)


if __name__ == '__main__':
    # def __main__():

    ####################DATA DIRECTORY###################
    # jy
    # global root

    # if not os.path.exists("./loss"):
    #    os.makedirs("./loss")
    # if not os.path.exists("./reconstructed_images/val"):
    #    os.makedirs("./reconstructed_images/val")
    # if not os.path.exists("./reconstructed_images/test"):
    #    os.makedirs("./reconstructed_images/test")
    # if not os.path.exists("./reconstructed_images"):
    #    os.makedirs("./reconstructed_images")

    main();
    exit()

