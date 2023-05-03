
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from core import DD_net,denseblock

INPUT_CHANNEL_SIZE = 1




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
import numpy as np
def prune_thresh(item, amount):

    w = item.weight
    b = item.bias
    w_s = w.size()
    b_s = b.size()
    b_flat = torch.flatten(b)
    w_flat = torch.flatten(w)

    w_numpy = w_flat.clone().detach().numpy()
    b_numpy = b_flat.clone().detach().numpy()

    w_threshold = np.percentile(np.abs(w_numpy), amount)
    b_threshold = np.percentile(np.abs(w_numpy), amount)

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
def mag_prune(model, amt):
    print(f" type of mode: type(model)")
    for index, item in enumerate(model.children()):
        if(type(item) == denseblock):
            for index, items in enumerate(item.children()):
                if hasattr(items, "weight"):
                    # print('pruning :', items)
                    prune_thresh(items,amt)
                else:
                    print("not pruning in dense block: ", items)
        else:
            if hasattr(item, "weight") and hasattr(item.weight, "requires_grad"):
                # print('pruning :', item)
                prune_thresh(item, amt)
            else:
                print('not pruning: ', item)
import os
def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    model_file = 'C:/Users/ayush/Downloads/weights_50_32.pt'
    dist.init_process_group("gloo", rank=0, world_size=1)
    map_location = torch.device('cpu')
    model = DD_net()
    model = DDP(model)
    calculate_global_sparsity(model)
    model.load_state_dict(torch.load(model_file, map_location=map_location))
    model_size = {}
    model_length = {}
    compress_rate = {}
    mat = {}
    # model = model
    mask_index = []
    # for i,n in model.named_parameters():
    #     print("asdf")
    mag_prune(list(model.children())[0], 0.9 )
    calculate_global_sparsity(model)

    params = list(list(model.children())[0].named_parameters())
    # k = list(keys)
    items = []
    for tei in params:
        if 'weight' in tei[0] or 'bias' in tei[0]:
            items.append(tei[1].data)
    wight_list = []

    for x in items:
        wight_list.append(torch.flatten(x))
    pp = torch.cat(wight_list, dim=0)
    bins = np.arange(-0.001, 0.001, 0.0001)
    import matplotlib.pyplot as plt
    plt.hist(pp.numpy(), bins=bins)
    plt.show()



        # print(x.count_nonzero())


    print("sadf")
if __name__ == '__main__':
    main()
