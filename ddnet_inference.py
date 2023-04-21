import torch.nn.utils.prune as prune
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

from PIL import Image
import os
from os import path
import numpy as np
import re

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.cuda.amp as amp
INPUT_CHANNEL_SIZE = 1

from core import DD_net, denseblock

from core import MSSSIM, SSIM

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

def main(args):
    if args.filepath is None:
        print("filepath not given")
        return

    global dir_pre
    dir_pre = args.out_dir
    print("dir prefix: " + dir_pre)
    batch= args.batch
    epochs = args.epochs
    rank = 0
    world_size = 1
    model_path = args.filepath
    model_file = "weights_" + str(epochs) + "_" + str(batch) + ".pt"
    file_m = model_path + "/" + model_file
    root_test_h = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ/"
    root_test_l = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ/"
    testset = CTDataset(root_dir_h=root_test_h, root_dir_l=root_test_l, length=784)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(testset, batch_size=batch, drop_last=False, shuffle=False, num_workers=1,pin_memory=False, sampler=test_sampler)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    list_of_imp_images = ['image_96.tif', 'sub-S04243_ses-E08488_acq-1_run-8_bp-chest_ct.nii0111.tif', 'sub-S04216_ses-E08457_run-3_bp-chest_ct.nii0284.tif', 'image_85.tif', 'image_192.tif']
    model = DD_net()
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    test_MSE_loss =[0]
    test_MSSSIM_loss = [0]
    test_total_loss = [0]
    model.load_state_dict(torch.load(file_m, map_location=map_location))
    with torch.no_grad():
        for batch_index, batch_samples in enumerate(test_loader):
            file_name, HQ_img, LQ_img, maxs, mins = batch_samples['vol'], batch_samples['HQ'], batch_samples['LQ'], \
                                                    batch_samples['max'], batch_samples['min']
            inputs = LQ_img.to(rank)
            targets = HQ_img.to(rank)
            outputs = model(inputs)
            MSE_loss = nn.MSELoss()(outputs, targets)
            MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
            loss = MSE_loss + 0.1 * (MSSSIM_loss)
            test_MSE_loss.append(MSE_loss.item())
            test_MSSSIM_loss.append(MSSSIM_loss.item())
            test_total_loss.append(loss.item())
            if file_name in list_of_imp_images:
                gen_visualization_files(outputs, targets, inputs, file_name, "test", maxs, mins)
        # torch.save(model.state_dict(), model_file)
        try:
            print('serializing test losses')
            np.save(dir_pre + '/loss/test_MSE_loss_' + str(rank), np.array(test_MSE_loss))
            np.save(dir_pre + '/loss/test_total_loss_' + str(rank), np.array(test_total_loss))
            np.save(dir_pre + '/loss/test_loss_mssim_' + str(rank), np.array(test_MSSSIM_loss))
        #            np.save('loss/test_loss_ssim_'+ str(rank), np.array(test_SSIM_loss))
        except Exception as e:
            print('error serializing: ', e)

        print("testing end")
        print("~~~~~~~~~~~~~~~~~~ everything completed ~~~~~~~~~~~~~~~~~~~~~~~~")

        #     data2 = np.loadtxt('./visualize/test/msssim_loss_target_out')
        #     print("size of out target: " + str(data2.shape))

        # print("size of append target: " + str(data3.shape))
        with open(dir_pre + "/myfile.txt", "w") as file1:
            s1 = "Final avergae MSE: " + str(np.average(test_MSE_loss)) + "std dev.: " + str(np.std(test_MSE_loss))
            file1.write(s1)
            s2 = "Final average MSSSIM LOSS: " + str(100 - (100 * np.average(test_MSSSIM_loss))) + 'std dev : ' + str(
                np.std(test_MSSSIM_loss))
            file1.write(s2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, metavar='e',
                        help='model file path')
    parser.add_argument('--out_dir', default=".", type=str, metavar='e',
                        help='model file path')
    parser.add_argument('--batch', default=1, type=int, metavar='b',
                        help='model file path')
    parser.add_argument('--epochs', default=50, type=int, metavar='b',
                        help='model file path')
    args = parser.parse_args()
    main(args)
    exit()