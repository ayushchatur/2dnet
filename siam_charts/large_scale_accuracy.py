import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 20
# plt.rc("xtick.top", True)
# plt.rc("xtick.labeltop", True)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

x = [1,2,4,8,16]
ddnet_dense = [
97.22,
93.02,
89,
86,
80]

ddnet_dense_op = [
97.22,
97.11,
97.02,
96.99,
96.66]

ddnet_sparse = [
97.22,
92.02,
86,
80,
72]

ddnet_sparse_op = [
97.22,
97.02,
96.99,
96.92,
96.77]


fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.set_ylim([70,100])
ax.set_xlim([0,17])
y = np.arange(70,101,3)
ax.set_yticks(y)
# ax.set_xticks([0,2,4,6,8,10,12,14,16,18])
rects1 = ax.plot(x,ddnet_dense, label='D1', color='blue', marker='*', linewidth=2)
rects2 = ax.plot(x,ddnet_dense_op, label='D2', color='pink', marker='*', linewidth=2)

rects3 = ax.plot(x,ddnet_sparse, label='S1', color='green', marker='*', linewidth=2)
rects4 = ax.plot(x,ddnet_sparse_op, label='S2', color='red', marker='*', linewidth=2)

ax.axhline(y = 97.22, color = 'black', linestyle = 'dashed', linewidth=1, label='baseline')

ax.legend(loc='lower left', fontsize=8)
# fig()
# Create the bars for the two categories at each x position
ax.set_xlabel('# of Ranks (GPUs)',fontsize=16)
ax.set_ylabel('MS-SSIM',fontsize=16)
fig.tight_layout()
plt.show()