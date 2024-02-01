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
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Data
categories = [ 'Dense', 'Structured \nSparse' ,'Random\nSparse', 'Magnitude\nSparse']
sub_categories = ['MP+UD', 'UD+GC', 'GC+MP', 'MP+UD+GC']
values = np.array([
    [2.7, 1.99, 1.55, 2.66],
    [2.57, 1.97, 1.67,2.49],
    [2.57, 1.94, 1.75,2.45],
    [2.57, 1.91, 1.69,2.47]
])

x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['sienna', 'olive','firebrick', 'gold']
# Add bars for each sub-category
for i in range(len(sub_categories)):
    bars = ax.bar(x - width/2 + i*width, values[:, i], width, label=sub_categories[i], color=colors[i])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 8.0, yval, yval, va='bottom', fontsize=14)  # va: vertical alignment
# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Categories')
ax.set_ylabel('Speedup', fontsize=26)
ax.set_ylim([0.0,3.0])
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
