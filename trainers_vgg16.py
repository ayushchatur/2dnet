#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 01/06/2022 1:43 PM
# @Author : Ayush Chaturvedi
# @E-mail : ayushchatur@vt.edu
# @Site :
# @File : sparse_ddnet.py
# @Software: PyCharm
# from apex import amp
# import torch.cuda.nvtx as nvtx
from importlib.resources import read_text

import torch.nn.utils.prune as prune
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import parser_util as prs
import os
from os import path
from PIL import Image

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,ReduceLROnPlateau,ExponentialLR
import torchvision
from core import denseblock
from socket import gethostname
class DD_net(nn.Module):
    def __init__(self, devc):
        super(DD_net, self).__init__()
        INPUT_CHANNEL_SIZE= 1
        resize = True
        self.input = None  #######CHANGE
        self.nb_filter = 16
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device=devc).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device=devc).view(1, 3, 1, 1))
        self.resize = resize

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
        c4 = F.leaky_relu(conv)  ## c4.out_channel= 16

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(c4), c3),
                      dim=1)  # c4=16*2, c3=16, => x = 16*3 (channels)
        dc4 = F.leaky_relu(self.convT1(self.batch6(x)))  ######size() CHANGE ; d4 : in=16*2, out=16*2
        dc4_1 = F.leaky_relu(self.convT2(self.batch7(dc4)));  # dc4_1 : in = 16*2 out = 16

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc4_1), c2),
                      dim=1)  # dc4_1 = 16*2, c2 =16 => x = 16*3
        dc5 = F.leaky_relu(self.convT3(self.batch8(x)))
        dc5_1 = F.leaky_relu(self.convT4(self.batch9(dc5)))

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc5_1), c1), dim=1)
        dc6 = F.leaky_relu(self.convT5(self.batch10(x)))
        dc6_1 = F.leaky_relu(self.convT6(self.batch11(dc6)))

        x = torch.cat((nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dc6_1), c0), dim=1)
        dc7 = F.leaky_relu(self.convT7(self.batch12(x)))
        dc7_1 = F.leaky_relu(self.convT8(self.batch13(dc7)))
        output = dc7_1
        # print('shape of dc7_1', output.size()) ## 1,1,512,512
        # print("shape of vgg_inp: " + str(vgg_inp.size()))
        # print("shape of zz: " + str(self.zz.size()))
        # zz.to(gpu)
        output = output.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)

        output = (output - self.mean) / self.std
        target = (target - self.mean) / self.std

        output = self.transform(output, mode='bilinear', size=(224, 224), align_corners=False)
        target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        # y = target
        # b1,b3
        out_b1 = self.blocks[0](output)
        out_b2 = self.blocks[1](out_b1)
        out_b3 = self.blocks[2](out_b2)

        tar_b1 = self.blocks[0](target)
        tar_b2 = self.blocks[1](tar_b1)
        tar_b3 = self.blocks[2](tar_b2)

        # print('sizes out_b1: {} tar_b1{}: '.format(out_b1.shape,tar_b1.shape))
        # print('sizes out_b3: {} tar_b3{}: '.format(out_b3.shape,tar_b3.shape))

        return dc7_1, out_b3, out_b1, tar_b3, tar_b1
def dd_train(args):
    torch.manual_seed(torch.initial_seed())
    world_size =  int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node  = torch.cuda.device_count()
    if gpus_per_node >0:
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    else:
        local_rank = 0

    distback = args.distback
    dist.init_process_group(distback, rank=rank, world_size=world_size)
    print(f"Hello from local_rank: {local_rank} and global rank {dist.get_rank()} of world with size: {dist.get_world_size()} on {gethostname()} where there are {gpus_per_node} allocated GPUs per node.", flush=True)
    # torch.cuda.set_device(local_rank)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    if rank == 0: print(args)
    batch = args.batch
    # print(args)
    epochs = args.epochs
    retrain = args.retrain
    prune_t = args.prune_t
    prune_amt = float(args.prune_amt)
    # enable_gr = (args.enable_gr == "true")
    gr_mode = args.gr_mode
    mod = args.model
    gr_backend = args.gr_backend
    amp_enabled = (args.amp == "true")
    global dir_pre
    global gamma
    global beta
    dir_pre = args.out_dir
    num_w = args.num_w
    en_wan = args.wan
    inference = (args.do_infer == "true")
    enable_prof = (args.enable_profile == "true")
    new_load = (args.new_load == "true")
    gr_mode = (args.enable_gr == "true")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu", local_rank)

    torch.cuda.manual_seed(1111)
    # necessary for AMP to work

    # model.to(device)

    model_file = "/projects/synergy_lab/ayush/dense_weights_ddnet_vgg16/dense_weights/weights_dense_" + str(epochs) + ".pt"

    print("loading vgg16")
    model = DD_net(devc=device)
    gamma = 0.03
    beta = 0.05


    model.to(device)

    if gr_mode:
        model = DDP(model, device_ids=[local_rank])
        model = torch.compile(model, fullgraph=True, mode=gr_mode, backend=gr_backend)

    else:
        model = DDP(model, device_ids=[local_rank])
    global learn_rate
    learn_rate = args.lr
    epsilon = 1e-8



    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, eps=epsilon)  #######ADAM CHANGE
    decayRate = args.dr
    sched_type = args.schedtype
    if sched_type == "cos" :
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=10,eta_min=0.0005)
    elif sched_type == "platu" :
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                               threshold=0.01, threshold_mode='rel',
                                                               cooldown=5, min_lr=0.005, eps=1e-03)
    else:
        scheduler = ExponentialLR(optimizer=optimizer, gamma=decayRate)


    # model_file = "1447477/weights_dense_" + str(epochs) + "_.pt"

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # if en_wan > 0:
    #     wandb.watch(model, log_freq=100)
    dist.barrier()
    if (not (path.exists(model_file))):
        print('model file not found')
    else:
        print(f'loading model file: {model_file}')
        model.load_state_dict(torch.load(model_file, map_location=map_location))

        print("initiating training")
        if mod != "baseline":
            print('running ddnet')
            from core.sparse_ddnet_old_dl import SpraseDDnetOld
        else:
            print('running ddnet-ml-vgg')
            from core.sparse_ddnet_old_vgg import SpraseDDnetOld

    trainer = SpraseDDnetOld(epochs, retrain, batch, model, optimizer, scheduler, world_size, prune_t, prune_amt,gamma, beta,dir_pre=dir_pre,  amp=amp_enabled, sched_type=sched_type)
    trainer.train_ddnet(rank,local_rank, enable_profile=enable_prof)

    if rank == 0:
        print("saving model file")
        model_file = f'weights_{str(batch)}_{str(epochs+retrain)}.pt'
        torch.save(model.state_dict(), dir_pre + "/" + model_file)
        if not inference:
            print("not doing inference.. training only script")
    # dist.barrier()
    dist.destroy_process_group()
    return


def main():

    parser = prs.get_parser()
    args = parser.parse_args()

    if(args.wan > 0):
        import wandb
        wandb.init()
    dd_train(args)


if __name__ == '__main__':
    main()
    exit()

