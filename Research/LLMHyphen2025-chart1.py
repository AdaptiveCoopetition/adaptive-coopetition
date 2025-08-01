
import matplotlib.pyplot as plt

# Data
agents = ["GPT-4o", "Deepseek v3", "Gemma", "MA", "MA PRM Flip", "AC"]
accuracy = [36.7, 40.7, 44.0, 43.3, 52.0, 54.0]

# Create the plot
plt.figure(figsize=(4, 3))
bars = plt.bar(agents, accuracy, color='lightsteelblue')

# Add title and labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Rotate x-axis labels for better readability
plt.xticks(fontsize=6, wrap=True)
#plt.yticks(fontsize=6)

# Hide y-axis labels
plt.gca().yaxis.set_ticklabels([])

# Hide tick marks
plt.tick_params(axis='both', which='both', length=0)

# Add data labels inside the top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval - 2, round(yval, 2), ha='center', va='top', fontsize=6, color='white')

# Turn off the border around the entire chart
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

# Save the plot as a PDF file
plt.tight_layout()
plt.savefig("model_accuracy.pdf")
