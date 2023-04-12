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

# from dataload import CTDataset
# from dataload_optimization import CTDataset


# vizualize_folder = "./visualize"
# loss_folder = "./loss"
# reconstructed_images = "reconstructed_images"





def gen_visualization_files(outputs, targets, inputs, file_names, val_test, maxs, mins):
    mapped_root = dir_pre + "/visualize/" + val_test + "/mapped/"
    diff_target_out_root = dir_pre +"/visualize/" + val_test + "/diff_target_out/"
    diff_target_in_root = dir_pre + "/visualize/" + val_test + "/diff_target_in/"
    # ssim_root = dir_pre + "/visualize/" + val_test + "/ssim/"
    out_root = dir_pre + "/visualize/" + val_test + "/"
    in_img_root = dir_pre +  "/visualize/" + val_test + "/input/"
    out_img_root = dir_pre + "/visualize/" + val_test + "/target/"

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




# jy
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

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
        trainer = Sparseddnet(epochs,retrain, batch,model,optimizer, scheduler,world_size, prune_t, prune_amt,dir_pre, amp_enabled)
        print("initializing training with new loader")

        trainer.trainer_new(rank,local_rank,enable_profile=enable_prof)
    else:
        print("initiating training")
        from core import SpraseDDnetOld
        trainer = SpraseDDnetOld(epochs, retrain, batch, model, optimizer, scheduler, world_size, prune_t, prune_amt, dir_pre, amp=amp_enabled)
        trainer.train_ddnet(rank,local_rank, enable_profile=enable_prof)

    if rank == 0:
        print("saving model file")
        torch.save(model.state_dict(), dir_pre + "/" + model_file)
        if not inference:
            print("not doing inference.. training only script")
    dist.barrier()
    dist.destroy_process_group()


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

