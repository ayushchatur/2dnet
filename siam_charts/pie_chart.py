import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# Data for the pie charts
labels_pie1 = ['Host to Device', 'Device to Device', 'Device to Host', 'Memset', 'Other']
sizes_pie1 = [77, 12, 7,3,1]
labels_pie2 = ['CUDA Stream Synchronize', 'CUDA Kernel Launch', 'CUDA Memcopy Async', 'Other']
sizes_pie2 = [48, 25, 16, 11]

# Data for the bar charts
labels_bar = ['Host to Device', 'Device to Device', 'Device to Host', 'Memset']
values = [7.2, 1, 0.5, 2]

colors_pie1 = ['teal', 'orange', 'green', 'red']
colors_pie2 = ['teal', 'orange', 'green', 'red']
colors_bar = ['gray', 'gray', 'gray', 'gray']

# Create a figure and a 2x3 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(22, 20))

# Top row
axs[0, 0].pie(sizes_pie1, autopct='%1.1f%%', colors=colors_pie1)  # Pie chart 1
axs[0,0].set_title('Application Profile')
axs[0, 1].pie(sizes_pie2,  autopct='%1.1f%%', colors=colors_pie2)  # Pie chart 2
axs[0,1].set_title('Data Movement Profile')


axs[0, 2].bar(labels_bar, values, color=colors_bar)  # Bar chart
axs[0,2].set_title('Memory Ops (in GB)')
# Bottom row
axs[1, 0].pie(sizes_pie1,  autopct='%1.1f%%', colors=colors_pie1)  # Pie chart 3
axs[1, 1].pie(sizes_pie2, autopct='%1.1f%%', colors=colors_pie2)  # Pie chart 4
axs[1, 2].bar(labels_bar, values, color=colors_bar)  # Bar chart

# Create legend patches
patches_pie1 = [mpatches.Patch(color=color, label=label) for color, label in zip(colors_pie1, labels_pie1)]
patches_pie2 = [mpatches.Patch(color=color, label=label) for color, label in zip(colors_pie2, labels_pie2)]
patches_bar = [mpatches.Patch(color=color, label=label) for color, label in zip(colors_bar, labels_bar)]

# Add a legend for each column at the bottom
fig.legend(handles=patches_pie1, loc='lower left', bbox_to_anchor=(0.1, 0.05), title="Pie Charts 1 Legend")
fig.legend(handles=patches_pie2, loc='lower center', bbox_to_anchor=(0.5, 0.05), title="Pie Charts 2 Legend")
# fig.legend(handles=patches_bar, loc='lower right', bbox_to_anchor=(0.9, 0.05), title="Bar Charts Legend")

# Adjust subplots to provide space for the legend
plt.subplots_adjust(bottom=0.2)

plt.tight_layout()
plt.show()
