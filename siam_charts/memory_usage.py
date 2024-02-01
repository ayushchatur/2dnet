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
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
# Data
categories = [ 'DDL' ,'DoLL']
sub_categories = ['Host to Device', 'Device to Device', 'Device to Host', 'Memset']
values = np.array([
    [7.20, 2.9, 0.75, 1.12],
    [.79, 7.6, .88, 3.6]
])

x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars
colors=['#30578b','#d0a13d','#5d9548',"#bc2d2f"]
fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

# Add bars for each sub-category
for i in range(len(sub_categories)):
    ax.bar(x - width/2 + i*width, values[:, i], width, label=sub_categories[i], color=colors[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Categories')
ax.set_ylabel('GigaBytes', fontsize=30)
ax.set_ylim([0.0,8])
ax.set_xticks(x + 0.1)
ax.set_xticklabels(categories, fontsize=30)
formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x))
plt.gca().yaxis.set_major_formatter(formatter)

# Move legend to top right corner
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=20)

# Add grid lines and put them behind the bars
ax.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.set_title("Data Movement (in Bytes)")
fig.tight_layout()

plt.show()