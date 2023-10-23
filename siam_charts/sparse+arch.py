import matplotlib.pyplot as plt
import numpy as np

# Categories and sub-categories
categories = ['ddnet', 'ddnet-ml-vgg16', 'ddnet-ml-vgg19']
colors = ['red', 'blue', 'green']

sub_categories = ['structured', 'unstructured', 'random']
markers = ['D', 's', 'o']  # D for diamond, s for square, o for circle


values = np.array([
    [1.7, 1.6, 1.2, 1.4],
    [1.6, 1.54, 1.15,1.52],
    [1.6, 1.54, 1.16,1.43],
    [1.6, 1.55, 1.13,1.54]
])

# Generate random data
num_points = 1
# data = {{'ddnet',}}
data = {}
for cat, color in zip(categories, colors):
    for sub_cat, marker in zip(sub_categories, markers):
        x = np.random.rand(num_points) + categories.index(cat)  # Shift x-values to separate categories
        y = np.random.rand(num_points)
        data[(cat, sub_cat)] = (x, y)

#(model,sparse_type): (x,y) or (accuracy drop, sppedup)
data = {
    ('ddnet', 'structured'): ([],[])
}

# Create the scatter plot
fig, ax = plt.subplots(figsize=(10, 6))

for (cat, sub_cat), (x, y) in data.items():
    ax.scatter(x, y, color=colors[categories.index(cat)], marker=markers[sub_categories.index(sub_cat)])

# Set x-ticks and labels
ax.set_xticks([0.5, 1.5, 2.5])
ax.set_xticklabels(categories)

# Dummy scatter plots for legend
for cat, color in zip(categories, colors):
    ax.scatter([], [], color=color, label=cat)
for sub_cat, marker in zip(sub_categories, markers):
    ax.scatter([], [], color='gray', marker=marker, label=sub_cat)

ax.set_title('Scatter Chart with Categories and Sub-Categories')
ax.legend()

plt.show()
