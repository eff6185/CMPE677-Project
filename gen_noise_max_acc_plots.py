import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # File path (update as needed)
    file_path = 'Data-traffic-distribution-giga_nox264_results.csv'

    # Ensure the output directory exists
    output_dir = 'Plots'
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    data = pd.read_csv(file_path)

    # Find the start of each table
    table_starts = data[data.iloc[:, 0].str.contains('Target Class:', na=False)].index

    # Noise levels to plot
    noise_levels = [0, 10, 20, 30, 40, 50]

    # Dictionary to store max accuracies for each class
    class_data = {}

    for i, start_idx in enumerate(table_starts):
        end_idx = table_starts[i + 1] if i + 1 < len(table_starts) else len(data)
        table = data.iloc[start_idx:end_idx]

        # Extract class name from the first row
        class_name = table.iloc[0, 0].split(':')[-1].strip()

        # Extract relevant data (skip the first row with class info)
        table_data = table.iloc[1:].reset_index(drop=True)
        table_data.columns = table.values[0]  # Set the column headers
        table_data = table_data[1:].reset_index(drop=True)  # Drop the header row

        # Ensure necessary columns are present
        required_columns = ['Noise Percantages', 'ACC (%)']
        table_data.columns = table_data.columns.str.strip()  # Strip whitespace from column headers
        for col in required_columns:
            if col not in table_data.columns:
                raise ValueError(f"Missing required column: {col} in table for class {class_name}")

        # Convert columns to numeric
        table_data['Noise Percantages'] = pd.to_numeric(table_data['Noise Percantages'], errors='coerce')
        table_data['ACC (%)'] = pd.to_numeric(table_data['ACC (%)'], errors='coerce')

        # Extract data for the six specific noise percentages
        filtered_data = table_data[table_data['Noise Percantages'].isin(noise_levels)]

        # Group by Noise Percentage and find the max accuracy
        grouped = filtered_data.groupby('Noise Percantages', as_index=False)['ACC (%)'].max()

        # Ensure all noise levels are present, filling missing ones with NaN
        grouped = grouped.set_index('Noise Percantages').reindex(noise_levels).reset_index()

        # Store max accuracies for the class
        if class_name not in class_data:
            class_data[class_name] = grouped
        else:
            # Update with new max values
            class_data[class_name] = pd.concat([class_data[class_name], grouped]).groupby('Noise Percantages', as_index=False)['ACC (%)'].max()

    # Generate plots after gathering all data for each class
    for class_name, grouped in class_data.items():
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['Noise Percantages'], grouped['ACC (%)'], marker='o', label=f'Class: {class_name}')

        # Add plot details
        plt.title(f'Maximum Accuracy vs Noise Percentage for Class: {class_name}')
        plt.xlabel('Noise Percentage')
        plt.ylabel('Maximum Accuracy (%)')
        plt.xticks(noise_levels)
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(f'2DPlots/{class_name}_max_accuracy_plot.png')
        plt.close()

    print("Plots have been generated and saved.")

if __name__ == "__main__":
    main()