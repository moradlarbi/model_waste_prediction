import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Instanciation du modèle avec Ridge Regression et les hyperparamètres optimaux
# alpha=10.0 et solver='lsqr' ont été identifiés comme les meilleurs paramètres
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=10.0, solver='lsqr', random_state=42))
])

# Entraînement du modèle
model_pipeline.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model_pipeline.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Hyperparameter tuning with GridSearchCV
# Nous utilisons un ensemble réduit de solveurs compatibles avec les données éparses
param_grid = {
    'regressor__alpha': [0.1, 1.0, 10.0, 100.0],
    'regressor__solver': ['auto', 'lsqr', 'sparse_cg', 'sag'] 
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Évaluation du modèle optimisé
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print(f'Best Model - Mean Squared Error: {mse_best}')
print(f'Best Model - Root Mean Squared Error: {rmse_best}')
print(f'Best Model - R-squared: {r2_best}')

# Save the improved model
joblib.dump(best_model, 'model.pkl')

# Residual plot for evaluation
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.savefig('residuals_distribution.png')
plt.close()

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig('actual_vs_predicted.png')
plt.close()

# Feature importance for Ridge Regression
# For Ridge, we use the coefficients as a measure of feature importance
feature_names = []
for name, transformer, features in preprocessor.transformers_:
    if name == 'cat':
        # Get the feature names from the OneHotEncoder
        feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(features))
    else:
        feature_names.extend(features)

# Get the coefficients from the Ridge model
coefficients = best_model.named_steps['regressor'].coef_

# Create a DataFrame with feature names and their coefficients
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot the top 20 most important features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance (Ridge Regression)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Evaluation plots saved as residuals_distribution.png, actual_vs_predicted.png, and feature_importance.png")
