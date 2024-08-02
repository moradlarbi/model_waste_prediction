import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Charger les données depuis un fichier CSV
file_path = 'merged_data_city.csv'
df = pd.read_csv(file_path)
df.dropna(subset=['total_msw_total_msw_generated_tons_year'], inplace=True)

# Sélection des caractéristiques et des variables cibles
X = df[['iso3c', 'region_id', 'income_id',
        'institutional_framework_department_dedicated_to_solid_waste_management_na',
        'legal_framework_long_term_integrated_solid_waste_master_plan_na',
        'legal_framework_solid_waste_management_rules_and_regulations_na',
        'population_population_number_of_people',
        'primary_collection_mode_form_of_primary_collection_na',
        'separation_existence_of_source_separation_na']]

# Liste des caractéristiques numériques
numeric_features = ['population_population_number_of_people']

# Liste des caractéristiques catégoriques
categorical_features = ['iso3c', 'region_id', 'income_id',
                         'institutional_framework_department_dedicated_to_solid_waste_management_na',
                         'legal_framework_long_term_integrated_solid_waste_master_plan_na',
                         'legal_framework_solid_waste_management_rules_and_regulations_na',
                         'primary_collection_mode_form_of_primary_collection_na',
                         'separation_existence_of_source_separation_na']

# Variables cibles
target_variables = ['total_msw_total_msw_generated_tons_year']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, df[target_variables], test_size=0.2, random_state=42)

# Prétraitement des données
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Appliquer le prétraitement aux données
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Entraîner un modèle pour chaque variable cible
models = {}
for target_variable in target_variables:
    model = LinearRegression()
    model.fit(X_train_processed, y_train[target_variable])
    models[target_variable] = model

# Prédire les valeurs pour l'ensemble de test
predictions = pd.DataFrame()
for target_variable, model in models.items():
    predictions[target_variable] = model.predict(X_test_processed)

# Fonction pour calculer l'Adjusted R-squared
def adjusted_r2_score(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Évaluation du modèle
for target_variable in target_variables:
    mse = mean_squared_error(y_test[target_variable], predictions[target_variable])
    mae = mean_absolute_error(y_test[target_variable], predictions[target_variable])
    mape = (abs((y_test[target_variable] - predictions[target_variable]) / y_test[target_variable])).mean() * 100
    r2 = r2_score(y_test[target_variable], predictions[target_variable])
    adj_r2 = adjusted_r2_score(r2, X_test_processed.shape[0], X_test_processed.shape[1])
    
    print(f'\nMetrics for {target_variable}:')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}')
    print(f'R-squared (R2): {r2}')
    print(f'Adjusted R-squared (Adj R2): {adj_r2}')
