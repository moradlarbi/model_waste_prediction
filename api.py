import os
import joblib
import pymysql
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Database connection setup using JAWSDB_URL
JAWSDB_URL = os.getenv("JAWSDB_URL")
parsed_url = urlparse(JAWSDB_URL)

# Extracting the connection details from the URL
db_host = parsed_url.hostname
db_user = parsed_url.username
db_password = parsed_url.password
db_name = parsed_url.path[1:]  # remove leading '/'

# Load the trained model
model = joblib.load('model.pkl')

# Load CSV data to compute mean values for composition fields (excluding population)
csv_file_path = 'merged_data_city.csv'
df_csv = pd.read_csv(csv_file_path)

# Define composition features that will be imputed from the CSV
composition_features = [
    'composition_food_organic_waste_percent',
    'composition_glass_percent',
    'composition_metal_percent',
    'composition_other_percent',
    'composition_paper_cardboard_percent',
    'composition_plastic_percent'
]

# Compute mean values from CSV for composition features
mean_values = df_csv[composition_features].mean().to_dict()

# Flask app
app = Flask(__name__)

def get_db_connection():
    return pymysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)

@app.route('/predict_all', methods=['GET'])
def predict_all_waste():
    # Fetch all regions from the database
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM Region")
    regions = cursor.fetchall()
    conn.close()

    if not regions:
        return jsonify({"error": "No regions found"}), 404

    # Define default values for categorical fields (as used during model training)
    categorical_defaults = {
        "iso3c": "DZA",
        "region_id": "AFR",
        "income_id": "LMC",
        "institutional_framework_department_dedicated_to_solid_waste_management_na": "Yes",
        "legal_framework_long_term_integrated_solid_waste_master_plan_na": "Yes",
        "legal_framework_solid_waste_management_rules_and_regulations_na": "Yes",
        "primary_collection_mode_form_of_primary_collection_na": "Yes",
        "separation_existence_of_source_separation_na": "Yes"
    }

    # Convert DB data to DataFrame
    df_regions = pd.DataFrame(regions)

    # Add missing composition columns if they don't exist in the DB table.
    for col in composition_features:
        if col not in df_regions.columns:
            df_regions[col] = pd.NA

    # Convert composition features to numeric and fill missing values with CSV mean values
    for col in composition_features:
        df_regions[col] = pd.to_numeric(df_regions[col], errors='coerce')
        df_regions[col].fillna(mean_values[col], inplace=True)

    # For population, use the value from the DB and rename it to match the model input.
    df_regions['population_population_number_of_people'] = pd.to_numeric(
        df_regions['population'], errors='coerce'
    )
    # In case of missing population, you might set a default (here, 0)
    df_regions['population_population_number_of_people'].fillna(0, inplace=True)

    # Set the categorical defaults (these columns do not exist in the DB, so we create them)
    for col, default in categorical_defaults.items():
        df_regions[col] = default

    # Define the expected order of columns as used during model training
    expected_cols = [
        'iso3c',
        'region_id',
        'income_id',
        'composition_food_organic_waste_percent',
        'composition_glass_percent',
        'composition_metal_percent',
        'composition_other_percent',
        'composition_paper_cardboard_percent',
        'composition_plastic_percent',
        'institutional_framework_department_dedicated_to_solid_waste_management_na',
        'legal_framework_long_term_integrated_solid_waste_master_plan_na',
        'legal_framework_solid_waste_management_rules_and_regulations_na',
        'population_population_number_of_people',
        'primary_collection_mode_form_of_primary_collection_na',
        'separation_existence_of_source_separation_na'
    ]

    input_df = df_regions[expected_cols]

    # Make predictions with the model
    predictions = model.predict(input_df)

    # Prepare the response (using the region id from the DB)
    result = [
        {"region_id": region["id"], "predicted_waste_tons_per_year": float(pred),"name":region["nom"]}
        for region, pred in zip(regions, predictions)
    ]

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
