import pandas as pd

# Load the first CSV file
file1_path = 'city_level_data_processed.csv'
df1 = pd.read_csv(file1_path, delimiter=';')

# Lowercase column names
df1.columns = df1.columns.str.lower()

# Define the columns to be added from the second file
columns_to_add = ['city_name', 'measurement', 'units', 'year', 'source']

# Read only the necessary columns from the second CSV file
file2_path = 'city_level_codebook_no_duplicates.csv'
df2_part = pd.read_csv(file2_path, usecols=columns_to_add)

# Lowercase column names
df2_part.columns = df2_part.columns.str.lower()

# Merge dataframes on the 'iso3c' column using inner join
merged_df = pd.merge(df1, df2_part, on='city_name', how='inner')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_data.csv', index=False)