#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/31/2020 1:43 PM
# @Author : Zhicheng Zhang
# @E-mail : zhicheng0623@gmail.com
# @Site :
# @File : train_main.py
# @Software: PyCharm
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
import numpy as np
from matplotlib import pyplot as plt

import os
from os import path
from PIL import Image

from matplotlib import pyplot as plt
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


INPUT_CHANNEL_SIZE = 1

def module_sparsity(module : nn.Module, usemasks = False):
    z  =0.0
    n  = 0
    if usemasks == True:
        for bname, bu in module.named_buffers():
            if "weight_mask" in bname:
                z += torch.sum(bu == 0).item()
                n += bu.nelement()
            if "bias_mask" in bname:
                z += torch.sum(bu == 0).item()
                n += bu.nelement()

    else:
        for name,p in module.named_parameters():
            if "weight" in name :
                z += torch.sum(p==0).item()
                n += p.nelement()
            if "bias" in name:
                z+= torch.sum(p==0).item()
                n += p.nelement()
    return  n , z

def calculate_global_sparsity(model: nn.Module):
    total_zeros = 0.0
    total_n = 0.0

    # global_sparsity = 100 * total_n / total_nonzero
    for name,m in model.named_modules():
        n , z = module_sparsity(m)
        total_zeros += z
        total_n += n


    global_sparsity = 100  * ( total_zeros  / total_n
                               )
    # global_sparsity = (
    #     100.0
    #     * float(
    #         torch.sum(model.conv1.weight == 0)
    #         + torch.sum(model.conv2.weight == 0)
    #         + torch.sum(model.fc1.weight == 0)
    #         + torch.sum(model.fc2.weight == 0)
    #     )
    #     / float(
    #         model.conv1.weight.nelement()
    #         + model.conv2.weight.nelement()
    #         + model.fc1.weight.nelement()
    #         + model.fc2.weight.nelement()
    #     )
    # )

    global_compression = 100 / (100 - global_sparsity)
    print('global sparsity', global_sparsity, 'global compression: ',global_compression)
    return global_sparsity, global_compression

def count_parameters(model):
    #print("Modules  Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
    return total_params

def init_env_variable():
    job_id = ""

    try:
        job_id = os.environ['SLURM_JOBID']
    except:
        job_id = ""

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


class denseblock(nn.Module):
    def __init__(self, nb_filter=16, filter_wh=5):
        super(denseblock, self).__init__()
        self.input = None  ######CHANGE
        self.nb_filter = nb_filter
        self.nb_filter_wh = filter_wh
        ##################CHANGE###############
        self.conv1_0 = nn.Conv2d(in_channels=nb_filter, out_channels=self.nb_filter * 4, kernel_size=1)
        self.conv2_0 = nn.Conv2d(in_channels=self.conv1_0.out_channels, out_channels=self.nb_filter,
                                 kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_1 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels, out_channels=self.nb_filter * 4,
                                 kernel_size=1)
        self.conv2_1 = nn.Conv2d(in_channels=self.conv1_1.out_channels, out_channels=self.nb_filter,
                                 kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_2 = nn.Conv2d(in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels,
                                 out_channels=self.nb_filter * 4, kernel_size=1)
        self.conv2_2 = nn.Conv2d(in_channels=self.conv1_2.out_channels, out_channels=self.nb_filter,
                                 kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1_3 = nn.Conv2d(
            in_channels=nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels,
            out_channels=self.nb_filter * 4, kernel_size=1)
        self.conv2_3 = nn.Conv2d(in_channels=self.conv1_3.out_channels, out_channels=self.nb_filter,
                                 kernel_size=self.nb_filter_wh, padding=(2, 2))
        self.conv1 = [self.conv1_0, self.conv1_1, self.conv1_2, self.conv1_3]
        self.conv2 = [self.conv2_0, self.conv2_1, self.conv2_2, self.conv2_3]

        self.batch_norm1_0 = nn.BatchNorm2d(nb_filter)
        self.batch_norm2_0 = nn.BatchNorm2d(self.conv1_0.out_channels)
        self.batch_norm1_1 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels)
        self.batch_norm2_1 = nn.BatchNorm2d(self.conv1_1.out_channels)
        self.batch_norm1_2 = nn.BatchNorm2d(nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels)
        self.batch_norm2_2 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.batch_norm1_3 = nn.BatchNorm2d(
            nb_filter + self.conv2_0.out_channels + self.conv2_1.out_channels + self.conv2_2.out_channels)
        self.batch_norm2_3 = nn.BatchNorm2d(self.conv1_3.out_channels)

        self.batch_norm1 = [self.batch_norm1_0, self.batch_norm1_1, self.batch_norm1_2, self.batch_norm1_3]
        self.batch_norm2 = [self.batch_norm2_0, self.batch_norm2_1, self.batch_norm2_2, self.batch_norm2_3]

    # def Forward(self, inputs):
    def forward(self, inputs):  ######CHANGE
        # x = self.input
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
        self.input = None  #######CHANGE
        self.nb_filter = 16

        ##################CHANGE###############
        self.conv1 = nn.Conv2d(in_channels=INPUT_CHANNEL_SIZE, out_channels=self.nb_filter, kernel_size=(7, 7),
                               padding=(3, 3))
        self.dnet1 = denseblock(self.nb_filter, filter_wh=5)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels * 5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet2 = denseblock(self.nb_filter, filter_wh=5)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels * 5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet3 = denseblock(self.nb_filter, filter_wh=5)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels * 5, out_channels=self.nb_filter, kernel_size=(1, 1))
        self.dnet4 = denseblock(self.nb_filter, filter_wh=5)

        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels * 5, out_channels=self.nb_filter, kernel_size=(1, 1))

        self.convT1 = nn.ConvTranspose2d(in_channels=self.conv4.out_channels + self.conv4.out_channels,
                                         out_channels=2 * self.nb_filter, kernel_size=5, padding=(2, 2))
        self.convT2 = nn.ConvTranspose2d(in_channels=self.convT1.out_channels, out_channels=self.nb_filter,
                                         kernel_size=1)
        self.convT3 = nn.ConvTranspose2d(in_channels=self.convT2.out_channels + self.conv3.out_channels,
                                         out_channels=2 * self.nb_filter, kernel_size=5, padding=(2, 2))
        self.convT4 = nn.ConvTranspose2d(in_channels=self.convT3.out_channels, out_channels=self.nb_filter,
                                         kernel_size=1)
        self.convT5 = nn.ConvTranspose2d(in_channels=self.convT4.out_channels + self.conv2.out_channels,
                                         out_channels=2 * self.nb_filter, kernel_size=5, padding=(2, 2))
        self.convT6 = nn.ConvTranspose2d(in_channels=self.convT5.out_channels, out_channels=self.nb_filter,
                                         kernel_size=1)
        self.convT7 = nn.ConvTranspose2d(in_channels=self.convT6.out_channels + self.conv1.out_channels,
                                         out_channels=2 * self.nb_filter, kernel_size=5, padding=(2, 2))
        self.convT8 = nn.ConvTranspose2d(in_channels=self.convT7.out_channels, out_channels=1, kernel_size=1)
        self.batch1 = nn.BatchNorm2d(1)
        self.max1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch2 = nn.BatchNorm2d(self.nb_filter * 5)
        self.max2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch3 = nn.BatchNorm2d(self.nb_filter * 5)
        self.max3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(self.nb_filter * 5)
        self.max4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch5 = nn.BatchNorm2d(self.nb_filter * 5)

        self.batch6 = nn.BatchNorm2d(self.conv5.out_channels + self.conv4.out_channels)
        self.batch7 = nn.BatchNorm2d(self.convT1.out_channels)
        self.batch8 = nn.BatchNorm2d(self.convT2.out_channels + self.conv3.out_channels)
        self.batch9 = nn.BatchNorm2d(self.convT3.out_channels)
        self.batch10 = nn.BatchNorm2d(self.convT4.out_channels + self.conv2.out_channels)
        self.batch11 = nn.BatchNorm2d(self.convT5.out_channels)
        self.batch12 = nn.BatchNorm2d(self.convT6.out_channels + self.conv1.out_channels)
        self.batch13 = nn.BatchNorm2d(self.convT7.out_channels)

    # def Forward(self, inputs):
    def forward(self, inputs):
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
        c3 = F.leaky_relu(conv)

        # p3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)(c3)
        p3 = self.max4(c3)  ######CHANGE
        D4 = self.dnet4(p3)

        conv = self.batch5(D4)
        conv = self.conv5(conv)
        c4 = F.leaky_relu(conv)

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(c4), c3), dim=1)
        dc4 = F.leaky_relu(self.convT1(self.batch6(x)))  ######size() CHANGE
        dc4_1 = F.leaky_relu(self.convT2(self.batch7(dc4)))

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc4_1), c2), dim=1)
        dc5 = F.leaky_relu(self.convT3(self.batch8(x)))
        dc5_1 = F.leaky_relu(self.convT4(self.batch9(dc5)))

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc5_1), c1), dim=1)
        dc6 = F.leaky_relu(self.convT5(self.batch10(x)))
        dc6_1 = F.leaky_relu(self.convT6(self.batch11(dc6)))

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc6_1), c0), dim=1)
        dc7 = F.leaky_relu(self.convT7(self.batch12(x)))
        dc7_1 = F.leaky_relu(self.convT8(self.batch13(dc7)))

        output = dc7_1

        return output



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
        im.save(out_img_root + file_name)

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(input_img_mapped)
        im.save(in_img_root + file_name)
        # jy
        # im.save(folder_ori_HU+'/'+file_name)

        file_name = file_names[i]
        file_name = file_name.replace(".IMA", ".tif")
        im = Image.fromarray(output_img_mapped)
        im.save(mapped_root + file_name)
        # jy
        # im.save(folder_enh_HU+'/'+file_name)

        difference_target_out = (target_img - output_img)
        difference_target_out = np.absolute(difference_target_out)
        fig = plt.figure()
        plt.imshow(difference_target_out)
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
        plt.imshow(difference_target_in)
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

    with open(out_root + "msssim_loss_target_out", 'a') as f:
        for item in MSSSIM_loss_out_target:
            f.write("%f\n" % item)

    with open(out_root + "msssim_loss_target_in", 'a') as f:
        for item in MSSSIM_loss_in_target:
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
        image_input = read_correct_image(self.data_root_l + self.img_list_l[idx])

        input_file = self.img_list_l[idx]
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

        sample = {'vol': input_file,
                  'HQ': targets,
                  'LQ': inputs,
                  'max': maxs,
                  'min': mins}
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

import torch.cuda.nvtx as nvtx
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
from apex.contrib.sparsity import ASP
def dd_train(gpu, args):
    rank = args.nr * args.gpus + gpu

    dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
    batch = args.batch
    epochs = args.epochs
    retrain = args.retrain
    root_train_h = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
    root_train_l = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"
    root_val_h = "/projects/synergy_lab/garvit217/enhancement_data/val/HQ/"
    root_val_l = "/projects/synergy_lab/garvit217/enhancement_data/val/LQ/"
    root_test_h = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
    root_test_l = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"
    root_hq_vgg3_tr = "/projects/synergy_lab/ayush/modcat1/train/vgg_output_b3/HQ/"
    root_hq_vgg3_te = "/projects/synergy_lab/ayush/modcat1/test/vgg_output_b3/HQ/"
    root_hq_vgg3_va = "/projects/synergy_lab/ayush/modcat1/val/vgg_output_b3/HQ/"

    root_hq_vgg1_tr = "/projects/synergy_lab/ayush/modcat1/train/vgg_output_b1/HQ/"
    root_hq_vgg1_te = "/projects/synergy_lab/ayush/modcat1/test/vgg_output_b1/HQ/"
    root_hq_vgg1_va = "/projects/synergy_lab/ayush/modcat1/val/vgg_output_b1/HQ/"

    # root = add
    trainset = CTDataset(root_dir_h=root_train_h, root_dir_l=root_train_l, root_hq_vgg3=root_hq_vgg3_tr,
                         root_hq_vgg1=root_hq_vgg1_tr, length=5120)
    testset = CTDataset(root_dir_h=root_val_h, root_dir_l=root_val_l, root_hq_vgg3=root_hq_vgg3_te,
                        root_hq_vgg1=root_hq_vgg1_te, length=784)
    valset = CTDataset(root_dir_h=root_test_h, root_dir_l=root_test_l, root_hq_vgg3=root_hq_vgg3_va,
                       root_hq_vgg1=root_hq_vgg1_va, length=784)
    # trainset = CTDataset(root_dir_h=root_train_h, root_dir_l=root_train_l, length=32)
    # testset = CTDataset(root_dir_h=root_val_h, root_dir_l=root_val_l, length=16)
    # valset = CTDataset(root_dir_h=root_test_h, root_dir_l=root_test_l, length=16)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=args.world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=args.world_size, rank=rank)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    train_loader = DataLoader(trainset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size,
                              pin_memory=False, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size,
                             pin_memory=False, sampler=test_sampler)
    val_loader = DataLoader(valset, batch_size=batch, drop_last=False, shuffle=False, num_workers=args.world_size,
                            pin_memory=False, sampler=val_sampler)
    # train_loader = DataLoader(trainset, num_workers=world_size, pin_memory=False, batch_sampler=train_sampler)
    # test_loader = DataLoader(testset, zbatch_size=batch, drop_last=False, shuffle=False)
    # val_loader = DataLoader(valset, batch_size=batch, drop_last=False, shuffle=False)

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

    train_MSE_loss = [0]
    train_MSSSIM_loss = [0]
    train_total_loss = [0]
    val_MSE_loss = [0]
    val_MSSSIM_loss = [0]
    val_total_loss = [0]
    test_MSE_loss = [0]
    test_MSSSIM_loss = [0]
    test_total_loss = [0]



    model_file = "weights_" + str(epochs) + "_" + str(batch) + ".pt"

    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}

    if (not (path.exists(model_file))):
        train_eval_ddnet(epochs, gpu, model, optimizer, rank, scheduler, train_MSE_loss, train_MSSSIM_loss,
                         train_loader, train_sampler, train_total_loss, val_MSE_loss, val_MSSSIM_loss, val_loader,
                         val_total_loss)
        print("train end")
        serialize_trainparams(model, model_file, rank, train_MSE_loss, train_MSSSIM_loss, train_total_loss, val_MSE_loss,
                              val_MSSSIM_loss, val_total_loss)

    else:
        print("Loading model parameters")
        model.load_state_dict(torch.load(model_file, map_location=map_location))
        print("Loading model parameters")
        print("sparifying the model....")
        calculate_global_sparsity(model)
        parm = []
        # original_model = copy.deepcopy(model)
        # # model.load_state_dict(torch.load(model_file, map_location=map_location))
        # for name, module in model.named_modules():
        #     if hasattr(module, "weight") and hasattr(module.weight, "requires_grad"):
        #         parm.append((module, "weight"))
        #         parm.append((module, "bias"))
        #
        # # layerwise_sparsity(pruned_model,0.3)
        # prune.global_unstructured(
        #     parameters=parm,
        #     pruning_method=prune.L1Unstructured,
        #     amount=0.5,
        # )
        # print('pruning masks applied successfully')
        # for name, module in model.named_modules():
        #     if hasattr(module, "weight") and hasattr(module.weight, "requires_grad"):
        #         try:
        #             prune.remove(module, "weight")
        #             prune.remove(module, "bias")
        #         except  Exception as e:
        #             print(' error pruing as ', e)

        # create new OrderedDict that does not contain `module.`
        ASP.prune_trained_model(model, optimizer)
        print('weights updated and masks removed... Model is sucessfully pruned')
        calculate_global_sparsity(model)
        if retrain > 0:
            print('fine tune retraining for ', retrain , ' epochs...')
            with torch.autograd.profiler.emit_nvtx():
                train_eval_ddnet(retrain, gpu, model, optimizer, rank, scheduler, train_MSE_loss, train_MSSSIM_loss,
                             train_loader, train_sampler, train_total_loss, val_MSE_loss, val_MSSSIM_loss, val_loader,
                             val_total_loss)
    test_ddnet(gpu, model, test_MSE_loss, test_MSSSIM_loss, test_loader, test_total_loss)

    print("testing end")
    with open('loss/test_MSE_loss_' + str(rank), 'w') as f:
        for item in test_MSE_loss:
            f.write("%f " % item)
    with open('loss/test_MSSSIM_loss_' + str(rank), 'w') as f:
        for item in test_MSSSIM_loss:
            f.write("%f " % item)
    with open('loss/test_total_loss_' + str(rank), 'w') as f:
        for item in test_total_loss:
            f.write("%f " % item)
    print("everything complete.......")

    print("Final avergae MSE: ", np.average(test_MSE_loss), "std dev.: ", np.std(test_MSE_loss))
    print("Final average MSSSIM LOSS: " + str(100 - (100 * np.average(test_MSSSIM_loss))), 'std dev : ', np.std(test_MSSSIM_loss))
    psnr_calc(test_MSE_loss)

def train_eval_ddnet(epochs, gpu, model, optimizer, rank, scheduler, train_MSE_loss, train_MSSSIM_loss,
                     train_loader, train_sampler, train_total_loss, val_MSE_loss, val_MSSSIM_loss, val_loader,
                     val_total_loss):
    start = datetime.now()
    for k in range(epochs):
        print("Training for Epocs: ", epochs)
        print('epoch: ', k, ' train loss: ', train_total_loss[k], ' mse: ', train_MSE_loss[k], ' mssi: ',
              train_MSSSIM_loss[k])
        train_sampler.set_epoch(epochs)
        for batch_index, batch_samples in enumerate(train_loader):
            file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                                                    batch_samples['max'], batch_samples['min']
            nvtx.range_push("Batch: " + str(batch_index))
            nvtx.range_push("copy to device")
            inputs = LQ_img.to(gpu)
            # inputs = LQ_img.cuda(non_blocking=True)
            targets = HQ_img.to(gpu)
            nvtx.range_pop()
            # targets = HQ_img.cuda(non_blocking=True)
            # outputs = model(inputs).to(rank)
            nvtx.range_push("Forward pass")
            outputs = model(inputs)
            MSE_loss = nn.MSELoss()(outputs, targets)
            MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
            # loss = nn.MSELoss()(outputs , targets_train) + 0.1*(1-MSSSIM()(outputs,targets_train))
            loss = MSE_loss + 0.1 * (MSSSIM_loss)
            nvtx.range_pop()
            nvtx.range_pop()
            train_MSE_loss.append(MSE_loss.item())
            train_MSSSIM_loss.append(MSSSIM_loss.item())
            train_total_loss.append(loss.item())
            # print("output shape:" + str(outputs.shape) + " target shape:" + str(targets.shape))
            model.zero_grad()
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
        # print('loss: ',loss, ' mse: ', mse
        scheduler.step()
        # print("Validation")
        for batch_index, batch_samples in enumerate(val_loader):
            file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                                                    batch_samples['max'], batch_samples['min']
            inputs = LQ_img.to(gpu)
            targets = HQ_img.to(gpu)
            outputs = model(inputs)
            # outputs = model(inputs)
            MSE_loss = nn.MSELoss()(outputs, targets)
            MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
            # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
            loss = MSE_loss + 0.1 * (MSSSIM_loss)
            # loss = MSE_loss
            # print("MSE_loss", MSE_loss.item())
            # print("MSSSIM_loss", MSSSIM_loss.item())
            # print("Total_loss", loss.item())
            # print("====================================")

            val_MSE_loss.append(MSE_loss.item())
            val_MSSSIM_loss.append(MSSSIM_loss.item())
            val_total_loss.append(loss.item())

            if (k == epochs - 1):
                if (rank == 0):
                    print("Training complete in: " + str(datetime.now() - start))
                outputs_np = outputs.cpu().detach().numpy()
                (batch_size, channel, height, width) = outputs.size()
                for m in range(batch_size):
                    file_name1 = file_name[m]
                    file_name1 = file_name1.replace(".IMA", ".tif")
                    im = Image.fromarray(outputs_np[m, 0, :, :])
                    im.save('reconstructed_images/val/' + file_name1)
                # gen_visualization_files(outputs, targets, inputs, val_files[l_map:l_map+batch], "val")
                gen_visualization_files(outputs, targets, inputs, file_name, "val", maxs, mins)


def serialize_trainparams(model, model_file, rank, train_MSE_loss, train_MSSSIM_loss, train_total_loss, val_MSE_loss,
                          val_MSSSIM_loss, val_total_loss):
    if (rank == 0):
        print("Saving model parameters")
        torch.save(model.state_dict(), model_file)
    with open('loss/train_MSE_loss_' + str(rank), 'w') as f:
        for item in train_MSE_loss:
            f.write("%f " % item)
    with open('loss/train_MSSSIM_loss_' + str(rank), 'w') as f:
        for item in train_MSSSIM_loss:
            f.write("%f " % item)
    with open('loss/train_total_loss_' + str(rank), 'w') as f:
        for item in train_total_loss:
            f.write("%f " % item)
    with open('loss/val_MSE_loss_' + str(rank), 'w') as f:
        for item in val_MSE_loss:
            f.write("%f " % item)
    with open('loss/val_MSSSIM_loss_' + str(rank), 'w') as f:
        for item in val_MSSSIM_loss:
            f.write("%f " % item)
    with open('loss/val_total_loss_' + str(rank), 'w') as f:
        for item in val_total_loss:
            f.write("%f " % item)


def test_ddnet(gpu, model, test_MSE_loss, test_MSSSIM_loss, test_loader, test_total_loss):
    for batch_index, batch_samples in enumerate(test_loader):
        file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                                                batch_samples['max'], batch_samples['min']
        inputs = LQ_img.to(gpu)
        targets = HQ_img.to(gpu)
        outputs = model(inputs)
        MSE_loss = nn.MSELoss()(outputs, targets)
        MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
        # loss = nn.MSELoss()(outputs , targets_test) + 0.1*(1-MSSSIM()(outputs,targets_test))
        loss = MSE_loss + 0.1 * (MSSSIM_loss)
        # loss = MSE_loss
        print("MSE_loss", MSE_loss.item())
        print("MSSSIM_loss", MSSSIM_loss.item())
        print("Total_loss", loss.item())
        print("====================================")
        test_MSE_loss.append(MSE_loss.item())
        test_MSSSIM_loss.append(MSSSIM_loss.item())
        test_total_loss.append(loss.item())
        outputs_np = outputs.cpu().detach().numpy()
        (batch_size, channel, height, width) = outputs.size()
        for m in range(batch_size):
            file_name1 = file_name[m]
            file_name1 = file_name1.replace(".IMA", ".tif")
            im = Image.fromarray(outputs_np[m, 0, :, :])
            im.save('reconstructed_images/test/' + file_name1)
        # outputs.cpu()
        # targets_test[l_map:l_map+batch, :, :, :].cpu()
        # inputs_test[l_map:l_map+batch, :, :, :].cpu()
        # gen_visualization_files(outputs, targets, inputs, test_files[l_map:l_map+batch], "test" )
        gen_visualization_files(outputs, targets, inputs, file_name, "test", maxs, mins)


def psnr_calc(mse_t):
    psnr = []
    for i in range(len(mse_t)):
        #     x = read_correct_image(pa +"/"+ ll[i])
        mse_sqrt = pow(mse_t[i], 0.5)
        psnr_ = 20 * np.log10(1 / mse_sqrt)
        psnr.insert(i, psnr_)
    print('psnr: ', np.mean(psnr), ' std dev', np.std(psnr))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    # parser.add_argument('-nr', '--nr', default=0, type=int,
    #                    help='ranking within the nodes')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch', default=2, type=int, metavar='N',
                        help='number of batch per gpu')
    parser.add_argument('--retrain', default=0, type=int, metavar='N',
                        help='retrain epochs')

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


