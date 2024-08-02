# import pandas as pd

# # Step 1: Read CSV files
# city_level_codebook = pd.read_csv('city_level_codebook.csv')
# city_level_data = pd.read_csv('city_level_data.csv')

# # Step 2: Convert column names to lowercase
# city_level_codebook.columns = city_level_codebook.columns.str.lower()
# city_level_data.columns = city_level_data.columns.str.lower()

# # Step 3: Merge DataFrames based on 'city_name'
# merged_data = pd.merge(city_level_codebook, city_level_data, on='city_name')

# # Step 4: Drop duplicate rows based on 'city_name'
# merged_data = merged_data.drop_duplicates(subset='city_name')

# # Step 5: Drop duplicate columns
# merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

# # Step 6: Save the merged DataFrame to a new CSV file
# merged_data.to_csv('merged_city_data.csv', index=False)
import pandas as pd

# Replace 'your_file.csv' with the actual file path
file_path = 'merged_city_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Drop the "comment" column
df = df.drop(columns=['country_name.1','income_id.1','region_id'])

# Save the updated DataFrame back to a CSV file
df.to_csv('merged_city_data.csv', index=False)
