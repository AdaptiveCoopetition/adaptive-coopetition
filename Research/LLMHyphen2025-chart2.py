
import matplotlib.pyplot as plt
import numpy as np

# Data sizes
data_sizes = [200, 600, 1000, 2000, 4000]

# Model performances and standard deviations
ucb_means = [54.25, 55.92, 55.70, 55.20, 57.00]
ucb_stds = [2.5, 0.1, 0.3, 0.4, 0.4]

gemma_means = [44.00, 47.92, 47.30, 45.63, 45.91]
gemma_stds = [0.7, 0.4, 0.8, 0.5, 0.1]

gpt_means = [36.75, 39.92, 39.2, 37.00, 39.60]
gpt_stds = [2.5, 1.5, 2.1, 0.3, 0.3]

deepseek_means = [40.75, 46.25, 48.95, 47.48, 47.30]
deepseek_stds = [0.4, 0.1, 0.4, 0.3, 0.2]

# Plotting the graph
plt.figure(figsize=(4, 3))

plt.errorbar(data_sizes, ucb_means, yerr=ucb_stds, fmt='-o', label='UCB', color='lightsteelblue')
plt.errorbar(data_sizes, gemma_means, yerr=gemma_stds, fmt='-o', label='gemma-3-27b-it', color='limegreen')
plt.errorbar(data_sizes, gpt_means, yerr=gpt_stds, fmt='-o', label='gpt-4o', color='goldenrod')
plt.errorbar(data_sizes, deepseek_means, yerr=deepseek_stds, fmt='-o', label='deepseek-v3-0324', color='mediumpurple')

plt.xlabel('Data Size', fontsize=6)
plt.ylabel('Performance (%)', fontsize=6)
plt.title('Model Performance Across Different Data Sizes', fontsize=6)
plt.legend(fontsize=6)
plt.grid(True)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

# Save the plot as a PDF file
plt.tight_layout()
plt.savefig("data_curves.pdf")
