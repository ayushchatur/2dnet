
import torch.nn.utils.prune as prune
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import ImageGrid
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

from socket import gethostname

# model_file = ""

def plot_layer(name, activation):
    print("Processing {} layer...".format(name))
    how_many_features_map = activation.shape[3]

    figure_size = how_many_features_map * 2
    fig = plt.figure(figsize=(figure_size, figure_size),)

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(how_many_features_map // 16, 16),
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    images = [activation[0, :, :, i] for i in range(how_many_features_map)]

    for ax, img in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.matshow(img)
    plt.savefig("plot.png")
from core import DD_net
model = DD_net()


image_file = "/Users/ayushchaturvedi/Documents/image_0.tif"


from ddnet_utils.image_read import read_correct_image

np_array = read_correct_image(image_file)

rmax = 0
rmin = 1

cmax1 = np.amax(np_array)
cmin1 = np.amin(np_array)
image_input = rmin + ((np_array - cmin1) / (cmax1 - cmin1) * (rmax - rmin))


image_input = image_input.reshape((1,1, 512, 512))
image_input = torch.from_numpy(image_input)
inputs = image_input.type(torch.FloatTensor)


features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook




model.max3.register_forward_hook(get_features(('feats')))

en_image = model(inputs)

    # add feats and preds to lists
# PREDS.append(preds.detach().cpu().numpy())
x = features['feats'].cpu().numpy()

plot_layer("max pool", x)
print(x.shape)