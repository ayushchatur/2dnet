#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 01/06/2022 1:43 PM
# @Author : Ayush Chaturvedi
# @E-mail : ayushchatur@vt.edu
# @Site :
# @File : sparse_ddnet.py
# @Software: PyCharm
from importlib.resources import read_text

from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math
from math import exp
import numpy as np
import os
from os import path


from socket import gethostname

import torch.distributed as dist

import torch.cuda.amp as amp
# from apex.contrib.sparsity import ASP

from ddnet_utils import mag_prune,unstructured_sparsity, ln_struc_spar
from core import MSSSIM, SSIM
from ddnet_utils import serialize_loss_item

INPUT_CHANNEL_SIZE = 1

class Sparseddnet(object):
    def __init__(self, epochs, retrain, batch,model, optimizer, scheduler, world_size, prune_t, prune_amt, dir_pre=".", amp = False):
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


    def trainer_new(self, global_rank, local_rank, output_path=".", enable_profile=False):
        global dir_pre
        dir_pre = output_path
        print(f"staging dataset on GPU {local_rank} of node {gethostname()}")
        root_train_h = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
        root_train_l = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"
        root_val_h = "/projects/synergy_lab/garvit217/enhancement_data/val/HQ/"
        root_val_l = "/projects/synergy_lab/garvit217/enhancement_data/val/LQ/"

        from data_loader.custom_load import CTDataset
        self.train_loader = CTDataset(root_train_h, root_train_l, 5120, local_rank, self.batch_size)
        self.val_loader = CTDataset(root_val_h, root_val_l, 784, local_rank, self.batch_size)

        self.scaler = torch.cuda.amp.GradScaler()
        sparsified = False
        densetime = 0
        # list of random indexes
        g = torch.Generator()
        g.manual_seed(0)

        if global_rank == 0:
            train_index_list = torch.randperm(len(self.train_loader), generator=g).tolist()
            val_index_list = torch.randperm(len(self.val_loader), generator=g).tolist()
        else:
            train_index_list = [0 for i in range(len(self.train_loader))]
            val_index_list = [0 for i in range(len(self.val_loader))]

        # share permuted list of index with all ranks
        dist.broadcast_object_list(train_index_list, src=0)
        dist.broadcast_object_list(val_index_list, src=0)

        train_items_per_rank = math.ceil((len(self.train_loader) - self.world_size) / self.world_size)
        val_items_per_rank = math.ceil((len(self.train_loader) - self.world_size) / self.world_size)

        q_fact_train = len(self.train_loader) // self.world_size
        q_fact_val = len(self.val_loader) // self.world_size
        from ddnet_utils import init_loss_params
        train_total_loss, train_MSSSIM_loss, train_MSE_loss, val_total_loss, val_MSSSIM_loss, val_MSE_loss = init_loss_params()

        #         train_sampler.set_epoch(epochs + prune_ep)


        for k in range(self.epochs + self.retrain):
            if global_rank == 0: print(f"q_factor train {q_fact_train} , qfactor va : {q_fact_val} ")

            train_index_list = train_index_list[global_rank * q_fact_train: (global_rank * q_fact_train + q_fact_train)]
            val_index_list = val_index_list[global_rank * q_fact_val: (global_rank * q_fact_val + q_fact_val)]
            print(f"rank {global_rank} index list: {train_index_list}")

            train_index_list = [int(x) for x in train_index_list]
            train_index_list = [list(train_index_list[i:i + self.batch_size]) for i in
                                range(0, len(train_index_list), self.batch_size)]
            val_index_list = [int(x) for x in val_index_list]
            val_index_list = [list(val_index_list[i:i + self.batch_size]) for i in range(0, len(val_index_list), self.batch_size)]

            if enable_profile:
                import nvidia_dlprof_pytorch_nvtx
                nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
                train_total, train_mse, train_msi, val_total, val_mse, val_msi = \
                    self._epoch_profile(train_index_list,val_index_list)
                train_total_loss[k] = train_total
                train_MSE_loss[k] = train_mse
                train_msi[k] = train_msi

                val_total_loss[k] = val_total
                val_MSE_loss[k] = val_mse
                val_MSSSIM_loss[k] = val_msi
            else:
                train_total, train_mse, train_msi, val_total, val_mse, val_msi = \
                    self._epoch(train_index_list,val_index_list)
                train_total_loss[k] = train_total
                train_MSE_loss[k] = train_mse
                train_msi[k] = train_msi

                val_total_loss[k] = val_total
                val_MSE_loss[k] = val_mse
                val_MSSSIM_loss[k] = val_msi


            # dist.barrier()
            start = datetime.now()

            if sparsified == False and self.retrain > 0 and k == (self.epochs - 1):
                densetime = str(datetime.now() - start)
                print('pruning model on epoch: ', k)
                if self.prune_t == "mag":
                    print("pruning model by top k with %: ", self.prune_amt)
                    mag_prune(self.model, self.prune_amt)
                elif self.prune_t == "l1_struc":
                    print("pruning model by L1 structured with %: ", self.prune_amt)
                    ln_struc_spar(self.model, self.prune_amt)
                else:
                    print("pruning model by random unstructured with %: ", self.prune_amt)
                    unstructured_sparsity(self.model, self.prune_amt)
                sparsified = True
        print("total timw : ", str(datetime.now() - start), ' dense time: ', densetime)
        serialize_loss_item(dir_pre, "train_mse_loss", train_MSE_loss, global_rank)
        serialize_loss_item(dir_pre, "train_total_loss", train_total_loss, global_rank)
        serialize_loss_item(dir_pre, "train_mssim_loss", train_MSSSIM_loss, global_rank)
        serialize_loss_item(dir_pre, "val_mse_loss", val_MSE_loss, global_rank)
        serialize_loss_item(dir_pre, "val_total_loss", val_total_loss, global_rank)
        serialize_loss_item(dir_pre, "val_mssim_loss", val_MSSSIM_loss, global_rank)

    def _epoch(self,train_index_list, val_index_list):
        train_MSE_loss = []
        train_MSSSIM_loss = []
        train_total_loss = []

        val_total_loss = []
        val_MSE_loss = []
        val_MSSSIM_loss = []
        for idx in train_index_list:
            sample_batched = self.train_loader.get_item(idx)
            HQ_img, LQ_img, maxs, mins, file_name = sample_batched['HQ'], sample_batched['LQ'], \
                sample_batched['max'], sample_batched['min'], sample_batched['vol']
            # print('indexes: ', idx)
            # print('shape: ', HQ_img.shape)
            # print('device: ', HQ_img.get_device())

            targets = HQ_img
            inputs = LQ_img
            with amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(inputs)
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                loss = MSE_loss + 0.1 * (MSSSIM_loss)
                # print(loss)
            # print('calculating backpass')
            train_MSE_loss.append(MSE_loss.item())
            train_MSSSIM_loss.append(MSSSIM_loss.item())
            train_total_loss.append(loss.item())
            # model.zero_grad()
            self.optimizer.zero_grad(set_to_none=True)
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
        print("schelud")
        self.scheduler.step()
        print("Validation")
        for idx in list(val_index_list):
            sample_batched = self.val_loader.get_item(idx)
            HQ_img, LQ_img, maxs, mins, fname = sample_batched['HQ'], sample_batched['LQ'], \
                sample_batched['max'], sample_batched['min'], sample_batched['vol']
            inputs = LQ_img
            targets = HQ_img
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


    def _epoch_profile(self,train_index_list, val_index_list):
        train_MSE_loss = [0]
        train_MSSSIM_loss = [0]
        train_total_loss = [0]
        val_total_loss = [0]
        val_MSE_loss = [0]
        val_MSSSIM_loss = [0]
        for idx in train_index_list:
            sample_batched = self.train_loader.get_item(idx)
            HQ_img, LQ_img, maxs, mins, file_name = sample_batched['HQ'], sample_batched['LQ'], \
                sample_batched['max'], sample_batched['min'], sample_batched['vol']
            self.optimizer.zero_grad(set_to_none=True)
            targets = HQ_img
            inputs = LQ_img
            torch.cuda.nvtx.range_push("Training loop:  " + str(idx))
            torch.cuda.nvtx.range_push("Forward pass")
            with amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(inputs)
                torch.cuda.nvtx.range_push("Loss calculation")
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                loss = MSE_loss + 0.1 * (MSSSIM_loss)
                torch.cuda.nvtx.range_pop()
                print(loss)
            torch.cuda.nvtx.range_pop()

            train_MSE_loss.append(MSE_loss.item())
            train_MSSSIM_loss.append(MSSSIM_loss.item())
            train_total_loss.append(loss.item())
            torch.cuda.nvtx.range_push("backward pass")
            # BW pass
            if self.amp_enabled:
                # print('bw pass')
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        print("schelud")
        self.scheduler.step()
        print("Validation")
        torch.cuda.nvtx.range_push("Validation " + str(idx))
        for idx in val_index_list:
            sample_batched = self.val_loader.get_item(idx)
            HQ_img, LQ_img, maxs, mins, fname = sample_batched['HQ'], sample_batched['LQ'], \
                sample_batched['max'], sample_batched['min'], sample_batched['vol']
            inputs = LQ_img
            targets = HQ_img
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
        torch.cuda.nvtx.range_pop()
        return train_total_loss, train_MSE_loss, train_MSSSIM_loss, val_total_loss, val_MSE_loss, val_MSSSIM_loss
