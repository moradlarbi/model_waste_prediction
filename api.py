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
db_name = parsed_url.path[1:]  # Remove the leading '/' from the database name

# Load the trained model
model = joblib.load('model.pkl')

# Flask app
app = Flask(__name__)

# Database connection function
def get_db_connection():
    return pymysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)

@app.route('/predict_all', methods=['GET'])
def predict_all_waste():
    # Fetch all regions from database
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM region")
    regions = cursor.fetchall()
    conn.close()
    if not regions:
        return jsonify({"error": "No regions found"}), 404

    # Prepare input data for all regions
    input_data = []
    for region in regions:
        input_data.append({
            "iso3c": "DZA",
            "region_id": "AFR",
            "income_id": "LMC",
            "institutional_framework_department_dedicated_to_solid_waste_management_na": "Yes",
            "legal_framework_long_term_integrated_solid_waste_master_plan_na": "Yes",
            "legal_framework_solid_waste_management_rules_and_regulations_na": "Yes",
            "population_population_number_of_people": region["population"],
            "primary_collection_mode_form_of_primary_collection_na": "Yes",
            "separation_existence_of_source_separation_na": "Yes"
        })

    input_df = pd.DataFrame(input_data)

    # Predict for all regions
    predictions = model.predict(input_df)
    print(predictions)
    # Prepare response with predictions
    result = []
    for i, region in enumerate(regions):
        result.append({
            "region_id": region["id"],
            "predicted_waste_tons_per_year": float(predictions[i])
        })

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
