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

import os
from os import path


from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.cuda.amp as amp
# from apex.contrib.sparsity import ASP

from core import denseblock,DD_net

INPUT_CHANNEL_SIZE = 1


def prune_thresh(item, amount):

    w = item.weight
    b = item.bias
    w_s = w.size()
    b_s = b.size()
    b_flat = torch.flatten(b)
    w_flat = torch.flatten(w)

    w_numpy = w_flat.clone().cpu().detach().numpy()
    b_numpy = b_flat.clone().cpu().detach().numpy()

    w_threshold = np.percentile(np.abs(w_numpy), amount*100)
    b_threshold = np.percentile(np.abs(w_numpy), amount*100)

    # pp = torch.where(w_flat > threshold, w_flat, float(0))
    w_numpy_new = w_numpy[  (w_threshold <= w_numpy)  | (w_numpy <= (-1*w_threshold))]

    b_numpy_new = b_numpy[ (b_threshold <= b_numpy)  | (b_threshold <= (-1*b_numpy))]

    w_tensor = torch.from_numpy(w_numpy_new)
    b_tensor = torch.from_numpy(b_numpy_new)


    # top_k_w = torch.topk(w_flat,int(w_flat.size().numel() * amount))
    # top_k_b = torch.topk(b_flat, int(b_flat.size().numel() * amount))
    # pp = torch.zeros_like(w)
    sparse_tensor_w = torch.zeros_like(w_flat)
    sparse_tensor_b = torch.zeros_like(b_flat)
    sparse_tensor_w[:len(w_tensor)] = w_tensor
    sparse_tensor_b[:len(b_tensor)] = b_tensor
    # print(pp)
    item.weight.data = sparse_tensor_w.unflatten(dim=0,sizes=w_s)
    item.bias.data = sparse_tensor_b.unflatten(dim=0,sizes=b_s)
def prune_weNb(item, amount):

    w = item.weight
    b = item.bias
    w_s = w.size()
    b_s = b.size()
    b_flat = torch.flatten(b)
    w_flat = torch.flatten(w)
    top_k_w = torch.topk(w_flat,int(w_flat.size().numel() * amount))
    top_k_b = torch.topk(b_flat, int(b_flat.size().numel() * amount))
    # pp = torch.zeros_like(w)
    sparse_tensor_w = torch.zeros_like(w_flat)
    sparse_tensor_b = torch.zeros_like(b_flat)
    sparse_tensor_w[:int(w_flat.size().numel() * amount)] = top_k_w.values
    sparse_tensor_b[:int(b_flat.size().numel() * amount)] = top_k_b.values
    # print(pp)
    item.weight.data = sparse_tensor_w.unflatten(dim=0,sizes=w_s)
    item.bias.data = sparse_tensor_b.unflatten(dim=0,sizes=b_s)
def mag_prune(model, amt):
    enable_cudnn_tensorcore(True)
    for index, item in enumerate(model.children()):
        if(type(item) == denseblock):
            for index, items in enumerate(item.children()):
                if hasattr(items, "weight"):
                    # print('pruning :', items)
                    prune_thresh(items, amt)
                else:
                    print("not pruning in dense block: ", items)
        else:
            if hasattr(item, "weight") and hasattr(item.weight, "requires_grad"):
                # print('pruning :', item)
                prune_thresh(item, amt)
            else:
                print('not pruning: ', item)

# dir_pre="."
def ln_struc_spar(model, amt):
    enable_cudnn_tensorcore(True)
    parm = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module.weight, "requires_grad"):
                parm.append((module, "weight"))
                # parm.append((module, "bias"))
    for item in parm:
        try:
            prune.ln_structured(item[0], amount=amt, name="weight", n=1, dim=0)
        except Exception as e:
            print('Error pruning: ', item[1], "exception: ", e)
    for module, name in parm:
        try:
            prune.remove(module, "weight")
            prune.remove(module, "bias")
        except  Exception as e:
            print('error pruning weight/bias for ',name,  e)
    print('pruning operation finished')

def unstructured_sparsity(model, amt):
    enable_cudnn_tensorcore(True)
    parm = []
    for name, module in model.named_modules():
        if ("conv" in name ) or ("batch" in name):
            if hasattr(module, "weight") and hasattr(module.weight, "requires_grad"):
                parm.append((module, "weight"))
                # parm.append((module, "bias"))

    # layerwise_sparsity(model,0.3)

    for item in parm:
        try:
            # prune.random_structured(item[0], amount=0.5, name="weight", dim=0)
            prune.random_unstructured(item[0], amount=amt, name="weight")
            # prune.random_unstructured(item[0], amount=amt, name="bias")

        except Exception as e:
            print('Error pruning: ', item[1], "exception: ", e)
        try:
            # prune.random_structured(item[0], amount=0.5, name="weight", dim=0)
            # prune.random_unstructured(item[0], amount=amt, name="weight")
            prune.random_unstructured(item[0], amount=amt, name="bias")

        except Exception as e:
            print('Error pruning: ', item[1], "exception: ", e)
    for module, name in parm:
        try:
            prune.remove(module, "weight")
            prune.remove(module, "bias")
        except  Exception as e:
            print('error pruning as ', e)

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

def enable_cudnn_tensorcore(enable: bool):
    if enable and torch.backends.cudnn.allow_tf32 == False:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

