import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
SMALL_SIZE = 8
MEDIUM_SIZE = 26
BIGGER_SIZE = 28
# plt.rc("xtick.top", True)
# plt.rc("xtick.labeltop", True)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Data
categories = ['Dense', 'Structured \nSparse' ,'Random\nSparse', 'Top-K\nSparse']
sub_categories = ['MP', 'UD', 'GC']
values = np.array([
    [1.49, 1.21, 1.16],
    [1.38, 1.21, 1.16],
    [1.35, 1.21, 1.16],
    [1.34, 1.21, 1.16]
])

x = np.arange(len(categories))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Add bars for each sub-category
for i in range(len(sub_categories)):
    bars = ax.bar(x - width/2 + i*width, values[:, i], width, label=sub_categories[i])
    for bar in bars:
        yval = bar.get_height()
        # plt.text(bar.get_x() + bar.get_width()/8.0, yval, yval, va='bottom',fontsize=14) # va: vertical alignment

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Categories')
ax.set_ylabel('Speedup', fontsize=26)
ax.set_ylim([0.0,2.0])
ax.set_xticks(x + width)
ax.set_xticklabels(categories, fontsize=24)
formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x))
plt.gca().yaxis.set_major_formatter(formatter)

# Move legend to top right corner
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.215), ncol=4)

# Add grid lines and put them behind the bars
ax.grid(True, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()

plt.show()
