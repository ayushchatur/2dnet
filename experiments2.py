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
# from PIL import Image

# from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.cuda.amp as amp
# from apex.contrib.sparsity import ASP
from socket import gethostname
# from dataload import CTDataset
from data_loader import CTDataset

#
# from ddnet_utils import mag_prune,ln_struc_spar,unstructured_sparsity
#
# from core.vgg19.ddnet_model import DD_net
# model = DD_net(devc='cpu')
#
# model = unstructured_sparsity(model,0.5)


# lr = 8e-3
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)  #######ADAM CHANGE
# decayRate = 0.95
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=10,eta_min=0.0005)
# amp_enabled = False
# lr_list = []
#
# # scaler = torch.cuda.amp.GradScaler()
# for k in range(50):
#
#     print(f'current lr: {scheduler.get_lr()} last lr: {type(scheduler.get_lr())}')
#     lr_list.append(scheduler.get_lr())
#     scheduler.step()
#
# # lr_list = [ float(x[0]) for x in lr_list]
# for item in lr_list:
#     print(item[0]
#           )
#
# from matplotlib import pyplot as plt
# # plt.plot(x_axis, loss_b1_list, label="B1", color='magenta')
# # plt.plot(x_axis,loss_b3_list,label="B3")
# plt.plot(list(range(1,51)),lr_list,label="inital lr= 8e-3", color = 'black')
# # plt.plot(x_axis,lr_e1,label="inital lr= 1e-3", color = 'red')
#
# # combined = [a + b for a, b in zip(loss_b1_list, loss_b3_list)]
# # combined = loss_b3_list
# # l = [x * beta for x in combined]
# # plt.plot(x_axis,l,label="VGG loss")
# # plt.plot(x_axis, total_avg_all,label="total_loss $(L)$", color='black')
# plt.xlabel("epochs")
# # plt.ylabel("values (fp)")
# # plt.yscale('log')
# # plt.xlim()
# plt.legend()
# plt.title('Learning rate (lr) Vs Epochs')
# plt.show()

