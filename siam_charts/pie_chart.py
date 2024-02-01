import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
SMALL_SIZE = 8
MEDIUM_SIZE = 36
BIGGER_SIZE = 40
# plt.rc("xtick.top", True)
# plt.rc("xtick.labeltop", True)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
# Data for the pie charts
labels_pie1 = ['Host to Device', 'Device to Device', 'Device to Host', 'Memset']
sizes_pie1 = [60, 24,6 ,9]

# labels_pie1 = ['Host to Device', 'Device to Device', 'Device to Host', 'Memset']
sizes_pie2 = [7, 64,7 ,22]
# Create a figure and a 2x3 grid of subplots
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(22, 20))
categories = labels_pie1
values1 = sizes_pie1
values2 =sizes_pie2

# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12),dpi=300)
# fig, ax = plt.subplots(figsize=(9, 8), )
# Pie chart 1
ax1.pie(values1,  autopct=lambda p: '{:.0f}%'.format(p) if p > 0 else '', startangle=140)
ax1.set_title('Using DDL', fontweight='bold', fontsize=50)
# ax1.set_xticklabels(fontsize=24)x
# Pie chart 2
ax2.pie(values2, autopct=lambda p: '{:.0f}%'.format(p) if p > 0 else '',  startangle=140)
ax2.set_title('Using DoLL',fontweight='bold', fontsize=50 )
plt.subplots_adjust(wspace=0.1)  # Adjust the width space
# ax2.set_xticklabels(fontsize=24)
# Add a common legend
# fig.legend(categories, loc='lower center', bbox_to_anchor=(0.5, -0.009), ncol=len(categories), )
plt.subplots_adjust(wspace=-0.2)  # Adjust the width space
# Adjust layout to prevent overlap
plt.tight_layout()

plt.show()