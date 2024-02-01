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
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Data
categories = [ 'DDNet', ]
sub_categories = ['Single Precision', 'Mixed Precision']
# values = np.array([
#     [31.48, 16.66],
#     [31.53,17.44],
#     [32.07,17.71]
# ])

values = np.array([
    [31.48, 16.66],

])

x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 6))

# Add bars for each sub-category
for i in range(len(sub_categories)):
    bars = ax.bar(x - width/16 + i*width, values[:, i], width, label=sub_categories[i])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 8.0, yval, yval, va='bottom', fontsize=14)  # va: vertical alignment
# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Categories')
ax.set_ylabel('Memory Usage (GB)', fontsize=20)
ax.set_ylim([0.0,40.0])
ax.set_xticks(x + 0.1)
ax.set_xticklabels(categories, fontsize=20)
formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x))
plt.gca().yaxis.set_major_formatter(formatter)

# Move legend to top right corner
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.215), ncol=4)

# Add grid lines and put them behind the bars
ax.grid(True, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()

plt.show()