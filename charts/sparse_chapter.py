
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# plt.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 32
# plt.rc("xtick.top", True)
# plt.rc("xtick.labeltop", True)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=32)     # fontsize of the axes title
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=26)    # fontsize of the tick labels
plt.rc('ytick', labelsize=26)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


x_ep = list(range(5,55,5))
figure(figsize=(12, 12), dpi=300, layout='tight')
# ax1.tight_layout()
ax1 = plt.subplot(1,1,1)
ax1.set_ylabel("MS-SSIM", labelpad=10, fontweight='bold')
ax1.set_xlim([5,50])
ax1.set_xlabel("Dense Epochs", labelpad=10, fontweight='bold')
ax1.tick_params(axis='x', which='major', pad=20, length=15, width=3)
ax1.tick_params(axis='y', which='major', pad=20, length=15, width=3)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(3)
# ax1.set_yscale('log')
ax1.set_ylim([88, 100])
# ax1.tick_params(axis='x',top=True, labeltop=True, bottom=False, labelbottom=False)
# ax1.set_xlim([0.001,0.0])
y = [97.20,97.76,96.44,96.89,97.30,95.15,96.07,95.77,94.34,97.22] # ddnet
# y = [97.93,98.14,97.92,97.64,97.76,97.39,97.89,97.34,95.79,98.88] # ml-vgg-16
# y = [97.67,97.55,97.70,97.40,97.00,97.53,96.48,97.46,95.68,95.48,98.03] # ml-vgg19
# y = [97.93]

ax1.plot(x_ep,y,label="MS-SSIM", color = 'blue', marker='*', markersize=20, linewidth=3)
ax1.set_xticks(x_ep)

# ax1.plot(x_ep,val_total_grad,label="Gradient Val. Total loss ", color = 'green')
ax1.axhline(y = 97.22, color = 'black', linestyle = 'dashed', linewidth=4, label='baseline')
ax1.axvline(x = 10, color = 'r', linestyle = 'dashed', linewidth=4, label='gradient staturation')
ax1.legend(loc='lower right')

ax2 = ax1.twiny()
x2 = list(range(0,46,5))
ax2.set_xticks(x2)
ax2.tick_params(axis='x', which='major', pad=10, length=15, width=3, )
ax2.tick_params(axis='y', which='major', pad=10, length=15, width=3)
# ax2.set_xlabel('Sparse epcohs')
ax2.set_xlim([45,0])
# ax2.invert_xaxis()


# ax2 = ax1.twiny()
# # ax2.set_xticks(x_axis)
# # ax2.set_xtickslabels(x_axis)
# ax2.xaxis.set_ticks_position('bottom')
# ax2.xaxis.set_label_position('bottom')
# ax2.spines['bottom'].set_position(('outward', 45))
ax2.set_xlabel('Sparse Epochs', labelpad=10, fontweight='bold')
# ax2.set_xlim([0,50])
# plt.legend()
# plt.title('Hybrid Schedule for DDNet-ML-VGG19', y=1.1)
# plt.show()
plt.savefig('ddnet_hybrid_grad_new.png',format='png',dpi = 300, transparent=True)