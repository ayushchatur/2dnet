import os
import collections

import torch
import torch.nn as nn
import torchsummary
from torch.utils.data import Dataset
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch import Tensor
from collections import OrderedDict
from torchvision.models.vgg import *
from torchvision import datasets, transforms
from torchvision import models as torchmodels
from PIL import Image
import re
from torchvision.models.vgg import vgg16, vgg16_bn, model_urls
import numpy as np
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

from pathlib import Path
#
# Path.open()
# img_path = 'C:/Users/ayush/Downloads/images.jpg'
# img_path2 = 'C:/Users/ayush/Downloads/BIMCV_139_image_0_h.tif'

# dir = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
read_path = '/projects/synergy_lab/garvit217/enhancement_data/test/HQ/'
write_path = '/projects/synergy_lab/ayush/modcat1/test/vgg_output_b1/HQ'
write_path3 = '/projects/synergy_lab/ayush/modcat1/test/vgg_output_b3/HQ'

r = Path(read_path)
w = Path(write_path)
w3 = Path(write_path3)

if not w.exists():
    Path.mkdir(w,parents=True)

if not w3.exists():
    Path.mkdir(w3,parents=True)

model_ft = torchmodels.vgg16(pretrained=True)
class vggextract_b3(nn.Module):
    def __init__(self, submodule):
        super(vggextract_b3, self).__init__()
        self.submodule = submodule
        for param in self.submodule.features.parameters():  # disable grad for trained layers
            param.requires_grad = False

        # first_conv_layer = {str(0) :nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)}
        # first_conv_layer.extend(list(model.features))
        modules = list(list(self.submodule.children())[0])[:16]
        # first_conv_layer.fromkeys(modules: modules)
        # module_dict = { **first_conv_layer,**{str(i+1): modules[i] for i in range(len(modules))} }
        module_dict = { **{str(i+1): modules[i] for i in range(len(modules))} }

        module_ordict = collections.OrderedDict(module_dict)
        # first_conv_layer.extend(modules)
        self.submodule = nn.Sequential(module_ordict)

    def forward(self, x):
        x = self.submodule(x)
        return x


class vggextract_b1(nn.Module):
    def __init__(self, submodule):
        super(vggextract_b1, self).__init__()
        self.submodule = submodule
        for param in self.submodule.features.parameters():  # disable grad for trained layers
            param.requires_grad = False

        # first_conv_layer = {str(0) :nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)}
        # first_conv_layer.extend(list(model.features))
        modules = list(list(self.submodule.children())[0])[:4]
        # first_conv_layer.fromkeys(modules: modules)
        module_dict = { **{str(i+1): modules[i] for i in range(len(modules))} }
        module_ordict = collections.OrderedDict(module_dict)
        # first_conv_layer.extend(modules)
        self.submodule = nn.Sequential(module_ordict)

    def forward(self, x):
        x = self.submodule(x)
        return x

extractor_b1 = vggextract_b1(model_ft)
extractor_b3 = vggextract_b3(model_ft)

new_transform =   transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        # transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        # transforms.CenterCrop((224, 224))
        transforms.Resize((224,224))
        # transforms.ToPILImage(),
        # transforms.ToTensor()
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
#
# img = read_correct_image(img_path2)  # 512x512 ndarray
# tra = new_transform(img) # 1,224,224 tensor
# xx = torch.zeros((3,224,224))
#
# xx[0,:,:] = tra
# xx[1,:,:] = tra
# xx[2,:,:] = tra
# nl = tra.numpy()
# zz = xx[None, :]
# features_b1 = extractor_b1(zz)
# features_b3 = extractor_b3(zz)
# print()


for entry in r.iterdir():
    img = read_correct_image(entry) # 512x512
    tra = new_transform(img)  # 1,224,224 tensor
    xx = torch.zeros((3, 224, 224))

    xx[0, :, :] = tra
    xx[1, :, :] = tra
    xx[2, :, :] = tra
    zz = xx[None, :]
    features_b1 = extractor_b1(zz)
    features_b3 = extractor_b3(zz)
    p = Path.joinpath(w, entry.name)
    p3 = Path.joinpath(w3, entry.name)
    torch.save(p, features_b1)
    torch.save(p3, features_b3)
    print("done processing " + str(entry.name))
    # torch.save(features_b3,'tt.pt')

# print()
# p = Path.joinpath(w,entry.name)
# np.save(p, features_b1)
# p3 = Path.joinpath(w3, entry.name)
# np.save(p3, features_b3)
# print("done processing " + str(entry.name))
