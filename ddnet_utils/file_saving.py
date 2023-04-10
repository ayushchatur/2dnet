from collections import defaultdict
import numpy as np


def init_loss_params():
    train_total_loss = defaultdict(list)
    train_MSSSIM_loss = defaultdict(list)
    train_MSE_loss = defaultdict(list)

    val_MSE_loss = defaultdict(list)
    val_total_loss = defaultdict(list)
    val_MSSSIM_loss = defaultdict(list)
    return train_total_loss,train_MSSSIM_loss,train_MSE_loss,val_total_loss, val_MSSSIM_loss,val_MSE_loss,
def serialize_loss_item(path: str, name: str, item: defaultdict, global_rank: int):
    np.save(f'{path}/loss/{name}_{str(global_rank)}', np.array([v for k, v in item.items()]))
