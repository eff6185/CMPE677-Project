import os
import pandas as pd

# Get the current directory
current_directory = os.getcwd()

# List all CSV files in the current directory
csv_files = [f for f in os.listdir(current_directory) if f.endswith('.csv')]

# Initialize an empty list to collect the dataframes
dataframes = []

# Flag to ensure we only write the header once
header_written = False

# Iterate over each CSV file
for file in csv_files:
    # Get the full file path
    file_path = os.path.join(current_directory, file)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # If the header has not been written, set header=True; otherwise, header=False
    if not header_written:
        dataframes.append(df)  # Add the dataframe including headers
        header_written = True   # Mark that the header has been written
    else:
        # Append without the header row
        dataframes.append(df.iloc[1:])  # Skip the first row (the header)

# Combine all DataFrames into one large DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame into a new CSV file
combined_df.to_csv('Data-traffic-distribution-giga.csv', index=False)

print("CSV files have been successfully combined into 'Data-traffic-distribution-giga.csv'.")