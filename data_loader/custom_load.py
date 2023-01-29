from utils.image_read import read_correct_image
import os
import torch
from random import shuffle
import numpy as np
from operator import itemgetter
class CTDataset(object):

    def __init__(self, root_dir_h, root_dir_l, length, device,batch_size = 1, seed = 333):
        self.seed = seed
        self.pos = 0
        self.batch_size = batch_size
        self.device = device
        self.data_root_l = root_dir_l + "/"
        self.data_root_h = root_dir_h + "/"
        self.img_list_l = os.listdir(self.data_root_l)
        self.img_list_h = os.listdir(self.data_root_h)
        self.img_list_l.sort()
        self.img_list_h.sort()
        self.img_list_l = self.img_list_l[0:length]
        self.img_list_h = self.img_list_h[0:length]
        # self.transform = transform

        self.index_list = [x for x in range(len(length))]
        shuffle(self.index_list)

        self.tensor_list_hq = []
        self.tensor_list_lq = []
        self.tensor_list_maxs = []
        self.tensor_list_mins = []
        self.tensor_list_fname = []

        for i in range(len(self.img_list_l)):
            rmax = 0
            rmin = 1
            image_target = read_correct_image(self.data_root_h + self.img_list_h[i])
            image_input = read_correct_image(self.data_root_l + self.img_list_l[i])
            input_file = self.img_list_l[i]
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
            inputs = inputs.type(torch.FloatTensor).to(self.device)
            targets = targets.type(torch.FloatTensor).to(self.device)
            self.tensor_list_fname.append(input_file)
            self.tensor_list_hq.append(targets)
            self.tensor_list_lq.append(inputs)
            self.tensor_list_maxs.append(maxs)
            self.tensor_list_mins.append(mins)


        print("done staging data to GPU")
    def __iter__(self):
        # shuffle(self.index_list)
        return self

    def return_stack(self, list_inp, index_list) -> torch.Tensor:
        hq_list = list(itemgetter(*index_list)(list_inp))
        hq_batch = torch.stack(hq_list)
        return hq_batch


    def create_batch(self):
        batch_list = self.index_list[self.pos:self.pos+self.batch_size]
        hq_batch = self.return_stack(self.tensor_list_hq,batch_list)
        lq_batch = self.return_stack(self.tensor_list_hq,batch_list)
        mins_batch = self.return_stack(self.tensor_list_hq,batch_list)
        maxs_batch = self.return_stack(self.tensor_list_hq,batch_list)
        fname_batch = self.return_stack(self.tensor_list_hq,batch_list)


        self.pos += self.batch_size
        return hq_batch,lq_batch,mins_batch,maxs_batch,fname_batch


    def __next__(self):
        hq_batch, lq_batch, mins_batch, maxs_batch, fname_batch =
        self.create_batch()


        sample = {
            'vol': fname_batch,
            'HQ': hq_batch,
            'LQ': lq_batch,
            'max': maxs_batch,
            'min': mins_batch
        }
        return sample

    def __len__(self):
        return len(self.tensor_list_mins)


def main():
    print('hell')
    dir_hq = "/projects/synergy_lab/garvit217/enhancement_data/test/HQ"
    dir_lq = "/projects/synergy_lab/garvit217/enhancement_data/test/LQ"
    loader = CTDataset(dir_hq, dir_lq, 10, device=torch.device("cuda:0"))
    print('hello')
    for hq, lq, ma, mi in loader:
        print(hq.shape)
        print(ma)
        print(mi)


#         print(f)

if __name__ == '__main__':
    main()