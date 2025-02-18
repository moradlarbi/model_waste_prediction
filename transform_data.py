import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données depuis un fichier CSV
file_path = 'merged_data_city.csv'
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
numeric_features = ['composition_food_organic_waste_percent', 'composition_glass_percent', 'composition_metal_percent',
                    'composition_other_percent', 'composition_paper_cardboard_percent', 'composition_plastic_percent',
                    'population_population_number_of_people']
categorical_features = ['iso3c', 'region_id', 'income_id',
                        'institutional_framework_department_dedicated_to_solid_waste_management_na',
                        'legal_framework_long_term_integrated_solid_waste_master_plan_na',
                        'legal_framework_solid_waste_management_rules_and_regulations_na',
                        'primary_collection_mode_form_of_primary_collection_na',
                        'separation_existence_of_source_separation_na']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputer les valeurs manquantes
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputer les valeurs manquantes pour les variables catégorielles
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Instanciation du modèle avec RandomForestRegressor
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Entraînement du modèle
model_pipeline.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model_pipeline.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Save the improved model
joblib.dump(best_model, 'model.pkl')
print("Improved model saved as model.pkl")

# Residual plot for evaluation
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
