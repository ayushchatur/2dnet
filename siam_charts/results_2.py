import matplotlib.pyplot as plt

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

# Values for your 7 data points
values = [1, 1.7, 1.6 , 1.2, 1.4]

# Corresponding labels for the data points
labels = ['Dense', 'MP+UD', 'UD+GCO', 'GCO+MP', 'MP+GCO+UD']

# Create a figure and a set of subplots
colors = ['gray', 'gray', 'gray', 'gray', 'gray']

# Create a figure and a set of subplots
plt.subplots()

# Create a bar chart with different colors
bars = plt.bar(labels, values, color=colors, width=0.7, zorder=2)

plt.grid(True, zorder=1)
# Add the value on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/8.0, yval, yval, va='bottom',fontsize=14) # va: vertical alignment


# Add gridlines
# plt.grid(True)

# Add a legend
# plt.legend()
plt.xticks(rotation=90)
# Provide titles for the x and y axes
# plt.xlabel('Labels')
plt.ylabel('Speedup')
plt.ylim([0,2.0])
# Provide a title for the bar chart
# plt.title()
plt.tight_layout()
plt.show()
# Show the bar chart
# plt.savefig('ddnet_results_combine.png',format='png',dpi = 300, transparent=True)