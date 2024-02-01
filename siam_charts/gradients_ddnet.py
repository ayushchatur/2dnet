import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

SMALL_SIZE = 8
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
# plt.rc("xtick.top", True)
# plt.rc("xtick.labeltop", True)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


train_loss = np.load('/Users/ayushchaturvedi/Library/CloudStorage/OneDrive-Personal/Documents/thesis_stuff/Loss Curves/train_total_loss_0.npy').mean(axis=1).tolist()
val_loss = np.load('/Users/ayushchaturvedi/Library/CloudStorage/OneDrive-Personal/Documents/thesis_stuff/Loss Curves/val_total_loss_0.npy').mean(axis=1).tolist()


# print(train_loss.shape)


x_axis = list(range(1,51))
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
rects1 = ax.plot(x_axis,train_loss, label='L1', color='red')
rects1 = ax.plot(x_axis,lr_e1, label='L2', color='black')


ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Values', fontsize=16)
ax.set_xlim([0,50])
ax.legend()
fig.tight_layout()
plt.show()
# plt.savefig('ddnet.png',format='png',dpi = 250