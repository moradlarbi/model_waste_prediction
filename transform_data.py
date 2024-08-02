
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Charger les données depuis un fichier CSV
file_path = 'merged_data_city.csv'  # Remplacez 'votre_fichier.csv' par le chemin réel de votre fichier
df = pd.read_csv(file_path)
df.dropna(subset=['total_msw_total_msw_generated_tons_year'], inplace=True)

# Sélection des caractéristiques et de la variable cible
X = df[['iso3c', 'region_id', 'income_id', 'composition_food_organic_waste_percent',
        'composition_glass_percent', 'composition_metal_percent', 'composition_other_percent',
        'composition_paper_cardboard_percent', 'composition_plastic_percent',
        'institutional_framework_department_dedicated_to_solid_waste_management_na',
        'legal_framework_long_term_integrated_solid_waste_master_plan_na',
        'legal_framework_solid_waste_management_rules_and_regulations_na',
        'population_population_number_of_people',
        'primary_collection_mode_form_of_primary_collection_na',
        'separation_existence_of_source_separation_na']]

y = df['total_msw_total_msw_generated_tons_year']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraitement des données
# Traitement des données manquantes
numeric_features = ['composition_food_organic_waste_percent',
        'composition_glass_percent', 'composition_metal_percent', 'composition_other_percent',
        'composition_paper_cardboard_percent', 'composition_plastic_percent', 'population_population_number_of_people']  # Liste des caractéristiques numériques
categorical_features = ['iso3c', 'region_id', 'income_id', 
        'institutional_framework_department_dedicated_to_solid_waste_management_na',
        'legal_framework_long_term_integrated_solid_waste_master_plan_na',
        'legal_framework_solid_waste_management_rules_and_regulations_na',
        'primary_collection_mode_form_of_primary_collection_na',
        'separation_existence_of_source_separation_na']  # Liste des caractéristiques catégoriques

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

# Instanciation et entraînement du modèle
model = LinearRegression()
model.fit(X_train_processed, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test_processed)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
