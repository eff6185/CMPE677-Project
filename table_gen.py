import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Data-traffic-distribution-results.csv")

# Rename columns for clarity
data.columns = ['Class_Info', 'TP', 'FP', 'FN', 'TN', 'Accuracy', 'Unused']

# Identify the rows that start new class tables
class_starts = data[data['Class_Info'].str.contains("Target Class", na=False)].index

# Prepare a dictionary to store accuracy values for each class
class_accuracies = {}

# Loop through each class
for i in range(len(class_starts)):
    start_idx = class_starts[i]
    end_idx = class_starts[i + 1] if i + 1 < len(class_starts) else len(data)

    # Get the class name
    class_name = data.loc[start_idx, 'Class_Info'].replace("Target Class: ", "").strip()

    # Extract the accuracy values for the current class
    class_data = data.loc[start_idx + 1:end_idx]
    class_data['Accuracy'] = pd.to_numeric(class_data['Accuracy'], errors='coerce')  # Convert to numeric

    # Store the accuracy values for min, max, and average calculations
    valid_accuracies = class_data['Accuracy'].dropna()
    if not valid_accuracies.empty:
        class_accuracies[class_name] = {
            'Min': valid_accuracies.min(),
            'Max': valid_accuracies.max(),
            'Avg': valid_accuracies.mean()
        }

# Convert the results into a DataFrame for visualization
summary_df = pd.DataFrame(class_accuracies).T
summary_df.reset_index(inplace=True)
summary_df.columns = ['Class', 'Min_Accuracy', 'Max_Accuracy', 'Avg_Accuracy']

# Create a bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(summary_df))  # Class indices for x-axis
width = 0.25

bars_min = ax.bar(x - width, summary_df['Min_Accuracy'], width, label='Min Accuracy', color='skyblue')
bars_max = ax.bar(x, summary_df['Max_Accuracy'], width, label='Max Accuracy', color='limegreen')
bars_avg = ax.bar(x + width, summary_df['Avg_Accuracy'], width, label='Avg Accuracy', color='orange')

ax.set_xlabel('Class')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Metrics by Class')
ax.set_xticks(x)
ax.set_xticklabels(summary_df['Class'], rotation=45, ha='right')
ax.legend()

# Display the chart
plt.tight_layout()
plt.show()

# Print the LaTeX table
latex_table = summary_df.to_latex(index=False, float_format="%.2f", caption="Accuracy Metrics by Class",
                                  label="tab:accuracy_metrics")
print(latex_table)