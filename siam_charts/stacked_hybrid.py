import matplotlib.pyplot as plt
import numpy as np

# Sample data
items = ['Item 1', 'Item 2', 'Item 3']
dense_epochs = [10, 15, 20]
sparse_epochs = [5, 10, 15]

# Performance scores for each item (for the secondary X-axis)
performance_scores = [0.85, 0.9, 0.88]

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Stack the bars vertically
ax.bar(items, dense_epochs, label='Dense Epochs', color='blue')
ax.bar(items, sparse_epochs, bottom=dense_epochs, label='Sparse Epochs', color='red')

# Create a secondary X-axis
ax2 = ax.twiny()
ax2.plot(items, performance_scores, color='green', marker='o', linestyle='-', label='Performance Score')
ax2.set_xlabel('Performance Score')

# Set the limits and labels for the secondary X-axis
ax2.set_xlim(0, 1)  # Assuming performance score is between 0 and 1
ax2.set_xticks(np.arange(len(items)))
ax2.set_xticklabels(items)

# Add labels, title, and legend
ax.set_ylabel('Epoch Values')
ax.set_title('Vertical Stacked Bar Chart with Secondary X-axis')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
