import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

SMALL_SIZE = 8
MEDIUM_SIZE = 22
BIGGER_SIZE = 38
# plt.rc("xtick.top", True)
# plt.rc("xtick.labeltop", True)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# ax1 = plt.subplot(1,1,1)


species = ()
penguin_means = {
    'Dense (baseline)': (79),
    'Structured': (70),
    'Random': (72),
    'Magnitude': (71)
}
# import matplotlib.pyplot as plt
# import numpy as np

# Some example data
# models = ["DDNet", "DDNet-ML-VGG16", "DDNet-ML-VGG19"]
models = ["DDNet"]

category1_values = [79 ]
category2_values = [70 ]
category3_values = [72 ]
category4_values = [71]

x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
# fig()
# Create the bars for the two categories at each x position
rects1 = ax.bar(x - 1.5*width, category1_values, width, label='Dense (baseline)', color='blue')
rects2 = ax.bar(x - width/2, category2_values, width, label='Structured', color='orange')
rects3 = ax.bar(x + ( (width/2)), category3_values, width, label='Random', color='green')
rects4 = ax.bar(x + (3*(width/2)), category4_values, width, label='Magnitude', color='maroon')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('')
ax.set_ylabel('Time (min.)')

ax.set_title('Total training times for DDNet')
ax.set_xticks(x)
ax.set_ylim([0,100])
# ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()

plt.show()

# plt.show()