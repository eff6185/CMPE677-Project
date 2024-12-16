import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

# Load the CSV file
file_path = 'Data-traffic-distribution-giga_nox264_results.csv'
data = pd.read_csv(file_path, header=None)

# Identify the start indices of each table
target_class_indices = data[data.iloc[:, 0].str.contains("Target Class:", na=False)].index

# Iterate over each table
for i in range(len(target_class_indices)):
    start_idx = target_class_indices[i]
    end_idx = target_class_indices[i + 1] if i + 1 < len(target_class_indices) else len(data)

    # Extract the table
    table = data.iloc[start_idx:end_idx].reset_index(drop=True)

    # Extract the target class name
    target_class = re.search(r"Target Class: (.+)", table.iloc[0, 0]).group(1)

    # Set the first row as the header
    table.columns = table.values[0]
    table = table[2:]  # Skip the first two rows (target class and headers)

    # Extract 'Nu' and 'Gamma' from the first column
    table[['Nu', 'Gamma']] = table.iloc[:, 0].str.extract(r'Nu: ([\d.e-]+), Gamma: ([\d.e-]+)').astype(float)

    # Convert 'ACC (%)' to float
    table['ACC (%)'] = table['ACC (%)'].astype(float)

    # Extract noise and upsampling
    noise_percentage = table.values[0][6]
    upsampling_value = table.values[0][7]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(table['Nu'], table['Gamma'], table['ACC (%)'], c='b', marker='o')

    ax.set_xlabel('Nu')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Accuracy (%)')
    ax.set_title(f"3D Plot for {target_class}\nNoise: {noise_percentage}%, Upsampling: {upsampling_value}")

    plt.savefig(f"Plots/plot_{target_class}_noise_{noise_percentage}_upsampling_{upsampling_value}.png")