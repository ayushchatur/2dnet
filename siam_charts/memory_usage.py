import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
SMALL_SIZE = 8
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
# plt.rc("xtick.top", True)
# plt.rc("xtick.labeltop", True)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Data
categories = [ 'DDL' ,'DoLL']
sub_categories = ['Host to Device', 'Device to Device', 'Device to Host', 'Memset']
values = np.array([
    [7.20, 1.15, 0.22, 2.12],
    [0.22, 3.12, 0.11, 5.65]
])

x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Add bars for each sub-category
for i in range(len(sub_categories)):
    ax.bar(x - width/2 + i*width, values[:, i], width, label=sub_categories[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Categories')
ax.set_ylabel('GigaBytes', fontsize=30)
ax.set_ylim([0.0,10])
ax.set_xticks(x + 0.1)
ax.set_xticklabels(categories, fontsize=30)
formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x))
plt.gca().yaxis.set_major_formatter(formatter)

# Move legend to top right corner
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=20)

# Add grid lines and put them behind the bars
ax.grid(True, zorder=0)
# ax.set_axisbelow(True)

fig.tight_layout()

plt.show()
