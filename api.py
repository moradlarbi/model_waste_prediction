import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import logging
from flask import Flask, request, jsonify

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data from a CSV file
file_path = 'merged_data_city.csv'
df = pd.read_csv(file_path)
df.dropna(subset=['total_msw_total_msw_generated_tons_year'], inplace=True)

# Feature selection and target variable
X = df[['iso3c', 'region_id', 'income_id',
        'institutional_framework_department_dedicated_to_solid_waste_management_na',
        'legal_framework_long_term_integrated_solid_waste_master_plan_na',
        'legal_framework_solid_waste_management_rules_and_regulations_na',
        'population_population_number_of_people',
        'primary_collection_mode_form_of_primary_collection_na',
        'separation_existence_of_source_separation_na']]

# Target variables
target_variables = ['total_msw_total_msw_generated_tons_year']

# List of numeric and categorical features
numeric_features = ['population_population_number_of_people']
categorical_features = ['iso3c', 'region_id', 'income_id',
                         'institutional_framework_department_dedicated_to_solid_waste_management_na',
                         'legal_framework_long_term_integrated_solid_waste_master_plan_na',
                         'legal_framework_solid_waste_management_rules_and_regulations_na',
                         'primary_collection_mode_form_of_primary_collection_na',
                         'separation_existence_of_source_separation_na']

# Preprocessing pipeline for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Changed to median
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining numeric and categorical preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that includes preprocessing and model training
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))  # Using RandomForestRegressor
])

# Train the model
logging.info('Training model...')
X_train, X_test, y_train, y_test = train_test_split(X, df[target_variables], test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train[target_variables[0]])

# Flask app setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON request data
    data = request.json
    
    # Convert data into a DataFrame
    input_data = pd.DataFrame([data])
    
    # Predict using the trained model pipeline
    prediction = model_pipeline.predict(input_data)
    
    # Return the prediction as a JSON response
    return jsonify({'predicted_waste': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
