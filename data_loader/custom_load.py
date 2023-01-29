from utils.image_read import read_correct_image
import os
import torch
from random import shuffle
import numpy as np
from operator import itemgetter


class CTDataset(object):
    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def create_batch(self, tensor_list: list, batch_size: int, is_tensor: bool, device="cpu"):
        batch_list = []
        if is_tensor:
            for x in self.batch(tensor_list, batch_size):
                batch_list.append(torch.stack(x).to(device))
        else:
            for x in self.batch(tensor_list, batch_size):
                batch_list.append(x)
        return batch_list
    def __len__(self):
        return len(self.ba_tensor_list_fname)

    def __init__(self, root_dir_h, root_dir_l, length, device="cpu", batch_size=1, seed=333):
        self.batch_size = batch_size
        self.device = device
        data_root_l = root_dir_l + "/"
        data_root_h = root_dir_h + "/"
        img_list_l = os.listdir(data_root_l)
        img_list_h = os.listdir(data_root_h)
        img_list_l.sort()
        img_list_h.sort()
        img_list_l = img_list_l[0:length]
        img_list_h = img_list_h[0:length]

        self.tensor_list_hq = []
        self.tensor_list_lq = []
        self.tensor_list_maxs = []
        self.tensor_list_mins = []
        self.tensor_list_fname = []
        for i in range(len(img_list_l)):
            rmax = 0
            rmin = 1
            image_target = read_correct_image(data_root_h + img_list_h[i])
            image_input = read_correct_image(data_root_l + img_list_l[i])
            input_file = img_list_l[i]
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
            self.tensor_list_fname.append(input_file)
            self.tensor_list_hq.append(targets)
            self.tensor_list_lq.append(inputs)
            self.tensor_list_maxs.append(maxs)
            self.tensor_list_mins.append(mins)
        self.ba_tensor_list_hq = self.create_batch(self.tensor_list_hq,self.batch_size,True,self.device)
        self.ba_tensor_list_lq = self.create_batch(self.tensor_list_lq,self.batch_size,True,self.device)
        self.ba_tensor_list_maxs = self.create_batch(self.tensor_list_maxs,self.batch_size,False)
        self.ba_tensor_list_mins = self.create_batch(self.tensor_list_mins,self.batch_size,False)
        self.ba_tensor_list_fname = self.create_batch(self.tensor_list_fname,self.batch_size,False)

        print("done staging data to GPU")

    def get_item(self, idx):
        hq = self.ba_tensor_list_hq[idx]
        lq = self.ba_tensor_list_lq[idx]
        mins = self.ba_tensor_list_mins[idx]
        maxs = self.ba_tensor_list_maxs[idx]
        vol = self.ba_tensor_list_fname[idx]
        sample = {
            'vol': vol,
            'HQ': hq,
            'LQ': lq,
            'min': mins,
            'maxs': maxs
        }
        return sample

def main():
    print('hell')
    dir_hq = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ"
    dir_lq = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ"
    hq, lq, mins, maxs, fnames = get_all_lists(dir_hq, dir_lq, 100, batch_size=4)
    b_hq = create_batch(hq, 4, True, "cpu")
    b_lq = create_batch(lq, 4, True, "cpu")
    b_min = create_batch(mins, 4, False)
    b_maxs = create_batch(maxs, 4, False)
    b_fname = create_batch(fnames, 4, False)
    index_list = np.random.default_rng(seed=22).permutation(range(len(b_hq)))


#         print(f)

if __name__ == '__main__':
    main()
