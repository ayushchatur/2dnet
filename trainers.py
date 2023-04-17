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
from core import DD_net
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,ReduceLROnPlateau,ExponentialLR

from socket import gethostname

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
    print(f"Hello from local_rank: {local_rank} and global rank {dist.get_rank()} of world with size: {dist.get_world_size()} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    # torch.cuda.set_device(local_rank)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    if rank == 0: print(args)
    batch = args.batch
    # print(args)
    epochs = args.epochs
    retrain = args.retrain
    prune_t = args.prune_t
    prune_amt = args.prune_amt
    # enable_gr = (args.enable_gr == "true")
    gr_mode = args.gr_mode
    gr_backend = args.gr_backend
    amp_enabled = (args.amp == "true")
    global dir_pre
    dir_pre = args.out_dir
    num_w = args.num_w
    en_wan = args.wan
    inference = (args.do_infer == "true")
    enable_prof = (args.enable_profile == "true")
    new_load = (args.new_load == "true")
    gr_mode = (args.enable_gr == "true")

    model = DD_net()
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu", local_rank)
    torch.cuda.manual_seed(1111)
    # necessary for AMP to work

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

    model_file = "weights_" + str(epochs) + "_" + str(batch) + ".pt"

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # if en_wan > 0:
    #     wandb.watch(model, log_freq=100)
    dist.barrier()
    if (not (path.exists(model_file))):
        print('model file not found')

    if new_load:
        print("initializing training with new loader")
        from core import Sparseddnet
        trainer = Sparseddnet(epochs,retrain, batch,model,optimizer, scheduler,world_size, prune_t, prune_amt,dir_pre, amp_enabled, sched_type=sched_type)
        print("initializing training with new loader")

        trainer.trainer_new(rank,local_rank,enable_profile=enable_prof)
    else:
        print("initiating training")
        from core import SpraseDDnetOld
        trainer = SpraseDDnetOld(epochs, retrain, batch, model, optimizer, scheduler, world_size, prune_t, prune_amt, dir_pre, amp=amp_enabled, sched_type=sched_type)
        trainer.train_ddnet(rank,local_rank, enable_profile=enable_prof)

    if rank == 0:
        print("saving model file")
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

