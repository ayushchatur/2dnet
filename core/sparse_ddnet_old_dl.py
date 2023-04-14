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
import parser_util as prs
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import torch.jit

from PIL import Image
import os
from os import path
import numpy as np
import re

from matplotlib import pyplot as plt
from torch import jit
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.cuda.amp as amp

from core import MSSSIM, SSIM
from ddnet_utils import mag_prune,ln_struc_spar,unstructured_sparsity
from ddnet_utils import serialize_loss_item, init_loss_params

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

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


class SpraseDDnetOld(object):
    def __init__(self, epochs, retrain, batch, model, optimizer, scheduler, world_size, prune_t, prune_amt, dir_pre=".", amp = False, sched_type='expo'):
        self.epochs = epochs
        self.retrain = retrain
        self.batch_size = batch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.scaler = grad_scaler
        self.world_size = world_size
        self.prune_type = prune_t
        self.prune_amt = prune_amt
        self.output_path = dir_pre
        self.amp_enabled = amp
        self.sched_type = sched_type

    def init_dataset_dataloader(self, global_rank: int, num_workers: int = 1):
        root_train_h = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
        root_train_l = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"
        root_val_h = "/projects/synergy_lab/garvit217/enhancement_data/val/HQ/"
        root_val_l = "/projects/synergy_lab/garvit217/enhancement_data/val/LQ/"
        root_test_h = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
        root_test_l = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"
        trainset = CTDataset(root_dir_h=root_train_h, root_dir_l=root_train_l, length=5120)
        valset = CTDataset(root_dir_h=root_val_h, root_dir_l=root_val_l, length=784)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=self.world_size, rank=global_rank)

        self.val_sampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=self.world_size, rank=global_rank)
        self.train_loader = DataLoader(trainset, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=num_workers,pin_memory=True, sampler=self.train_sampler)

        self.val_loader = DataLoader(valset, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=num_workers,pin_memory=True, sampler=self.val_sampler)

    def train_ddnet(self,global_rank, local_rank, enable_profile=False, en_wan= None):
        global dir_pre
        dir_pre = self.output_path
        self.init_dataset_dataloader(global_rank)


        sparsified = False
        densetime=0
        train_total_loss, train_MSSSIM_loss, train_MSE_loss, val_total_loss, val_MSSSIM_loss, val_MSE_loss = init_loss_params()
        start = datetime.now()
        print("beginning training epochs")
        print(f'profiling: {enable_profile}')
        for k in range(self.epochs + self.retrain):
            # train_sampler.set_epoch(epochs + retrain)
            # print("Training for Epocs: ", self.epochs + self.retrain)
            self.train_sampler.set_epoch(k)
            if enable_profile:
                import nvidia_dlprof_pytorch_nvtx
                nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
                train_total, train_mse, train_msi, val_total, val_mse, val_msi =\
                   self._epoch_profile(local_rank)
                train_total_loss[k] = train_total
                train_MSE_loss[k] = train_mse
                train_MSSSIM_loss[k] = train_msi

                val_total_loss[k] = val_total
                val_MSE_loss[k] = val_mse
                val_MSSSIM_loss[k] = val_msi
            else:
                train_total, train_mse, train_msi, val_total, val_mse, val_msi = \
                    self._epoch(local_rank)

                train_total_loss[k] = train_total
                train_MSE_loss[k] = train_mse
                train_MSSSIM_loss[k] = train_msi

                val_total_loss[k] = val_total
                val_MSE_loss[k] = val_mse
                val_MSSSIM_loss[k] = val_msi
            # optimizer.param_groups
            if sparsified == False and self.retrain > 0 and k == (self.epochs-1) :
                densetime = str(datetime.now()- start)
                print('pruning model on epoch: ', k)
                if self.prune_t == "mag":
                    print("pruning model by top k with %: ", self.prune_amt)
                    mag_prune(self.model,self.prune_amt)
                elif self.prune_t == "l1_struc":
                    print("pruning model by L1 structured with %: ", self.prune_amt)
                    ln_struc_spar(self.model, self.prune_amt)
                else:
                    print("pruning model by random unstructured with %: ", self.prune_amt)
                    unstructured_sparsity(self.model, self.prune_amt)

                sparsified = True

        # torch.cuda.current_stream().synchronize()
        print("total timw : ", str(datetime.now() - start), ' dense time: ', densetime)
        serialize_loss_item(dir_pre,"train_mse_loss",train_MSE_loss,global_rank)
        serialize_loss_item(dir_pre,"train_total_loss",train_total_loss,global_rank)
        serialize_loss_item(dir_pre,"train_mssim_loss",train_MSSSIM_loss,global_rank)
        serialize_loss_item(dir_pre,"val_mse_loss",val_MSE_loss,global_rank)
        serialize_loss_item(dir_pre,"val_total_loss",val_total_loss,global_rank)
        serialize_loss_item(dir_pre,"val_mssim_loss",val_MSSSIM_loss,global_rank)

    def _epoch(self,local_rank):
        train_MSE_loss = []
        train_MSSSIM_loss = []
        train_total_loss = []

        val_total_loss = []
        val_MSE_loss = []
        val_MSSSIM_loss = []

        for batch_index, batch_samples in enumerate(self.train_loader):
            file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                batch_samples['max'], batch_samples['min']

            self.optimizer.zero_grad(set_to_none=True)
            targets = HQ_img.to(local_rank, non_blocking=True)
            inputs = LQ_img.to(local_rank, non_blocking=True)
            with amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(inputs)
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                loss = MSE_loss + 0.1 * (MSSSIM_loss)
                # print(loss)
            #             print('calculating loss')
            train_MSE_loss.append(MSE_loss.item())
            train_MSSSIM_loss.append(MSSSIM_loss.item())
            train_total_loss.append(loss.item())
            # model.zero_grad()
            # BW pass
            if self.amp_enabled:
                # print('bw pass')
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                #                 print('loss_bacl')
                loss.backward()
                #                 print('optimia')
                self.optimizer.step()
        # print("schelud")

        self.scheduler.step()
        # print("Validation")
        for batch_index, batch_samples in enumerate(self.val_loader):
            file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                batch_samples['max'], batch_samples['min']
            inputs = LQ_img.to(local_rank)
            targets = HQ_img.to(local_rank)
            with amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(inputs)
                # outputs = model(inputs)
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                loss = MSE_loss + 0.1 * (MSSSIM_loss)

            val_MSE_loss.append(MSE_loss.item())
            val_total_loss.append(loss.item())
            val_MSSSIM_loss.append(MSSSIM_loss.item())

        return train_total_loss, train_MSE_loss, train_MSSSIM_loss, val_total_loss, val_MSE_loss, val_MSSSIM_loss

    def _epoch_profile(self, local_rank):
        train_MSE_loss = []
        train_MSSSIM_loss = []
        train_total_loss = []

        val_total_loss = []
        val_MSE_loss = []
        val_MSSSIM_loss = []
        self.optimizer.zero_grad(set_to_none=True)
        scaler = torch.cuda.amp.GradScaler()
        for batch_index, batch_samples in enumerate(self.train_loader):
            file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                batch_samples['max'], batch_samples['min']

            with amp.autocast(enabled=self.amp_enabled):
                torch.cuda.nvtx.range_push("copy to device")  # H2D
                targets = HQ_img.to(local_rank, non_blocking=True)
                inputs = LQ_img.to(local_rank, non_blocking=True)
                torch.cuda.nvtx.range_pop()  # H2D

                torch.cuda.nvtx.range_push("forward pass, step:" + str(batch_index))  # FP
                outputs = self.model(inputs)
                torch.cuda.nvtx.range_push("Loss calculation")  # Loss
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                loss = MSE_loss + 0.1 * (MSSSIM_loss)
                torch.cuda.nvtx.range_pop()  # Loss

                torch.cuda.nvtx.range_pop()  # FP
            # print(outputs.shape)

            train_MSE_loss.append(MSE_loss.item())
            train_MSSSIM_loss.append(MSSSIM_loss.item())
            train_total_loss.append(loss.item())

            torch.cuda.nvtx.range_push("backward pass")  # BP
            if self.amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            torch.cuda.nvtx.range_pop()  # BP

        self.scheduler.step()
        # nvtx.range_pop()
        # print("Validation")

        for batch_index, batch_samples in enumerate(self.val_loader):
            file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                batch_samples['max'], batch_samples['min']
            torch.cuda.nvtx.range_push("Validation step: " + str(batch_index))
            inputs = LQ_img.to(local_rank)
            targets = HQ_img.to(local_rank)

            with amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(inputs)
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                loss = MSE_loss + 0.1 * (MSSSIM_loss)

            val_MSE_loss.append(MSE_loss.item())
            val_total_loss.append(loss.item())
            val_MSSSIM_loss.append(MSSSIM_loss.item())
            torch.cuda.nvtx.range_pop()

        return train_total_loss, train_MSE_loss, train_MSSSIM_loss, val_total_loss, val_MSE_loss, val_MSSSIM_loss
