import pandas as pd

# Load the CSV file
file_path = 'country_level_data_0.csv'
df = pd.read_csv(file_path, delimiter=';')

# Calculate the percentage of missing values for each column
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Define a threshold (10% in this case)
threshold = 50

# Identify columns with missing values exceeding the threshold
columns_to_drop = missing_percentage[missing_percentage > threshold].index

# Drop the identified columns
df = df.drop(columns=columns_to_drop)

# Save the modified DataFrame back to a CSV file
df.to_csv('country_level_data_processed.csv', index=False)

print(f"Columns with more than {threshold}% missing values have been removed.")
