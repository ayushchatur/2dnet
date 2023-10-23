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
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

x = [1,2,4,8,16]
ddnet_dense = [1,
2.3,
4.3,
8.7,
15.8]

ddnet_all = [1,
1.923076923,
3.846153846,
7.142857143,
14.70588235]
ddnet_sparse_all = [1,
2.653846154,
4.3125,
9.857142857,
17.25]

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.set_ylim([0,18])
ax.set_xlim([0,18])
ax.set_yticks([0,2,4,6,8,10,12,14,16,18])
ax.set_xticks([0,2,4,6,8,10,12,14,16,18])
rects1 = ax.plot(x,ddnet_all, label='DDNet', color='blue', marker='*')
rects1 = ax.plot(x,ddnet_sparse_all, label='Sparse DDNet', color='green', marker='*')
rects2 = ax.plot(x,x, label='Linear Scaling', color='red', linestyle='dashed')
ax.legend(loc='upper left', fontsize=10)
# fig()
# Create the bars for the two categories at each x position
ax.set_xlabel('# of Ranks (GPUs)', fontsize=16)
ax.set_ylabel('Speedup', fontsize=16)
fig.tight_layout()
plt.show()