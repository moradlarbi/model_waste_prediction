import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as XGBRegressor
import time
import warnings
warnings.filterwarnings('ignore')

# Fonction pour charger et préparer les données
def load_and_prepare_data(file_path):
    print("Chargement et préparation des données...")
    # Charger les données
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
    
    # Identifier les types de variables
    numeric_features = ['composition_food_organic_waste_percent', 'composition_glass_percent', 
                        'composition_metal_percent', 'composition_other_percent', 
                        'composition_paper_cardboard_percent', 'composition_plastic_percent',
                        'population_population_number_of_people']
    
    categorical_features = ['iso3c', 'region_id', 'income_id',
                           'institutional_framework_department_dedicated_to_solid_waste_management_na',
                           'legal_framework_long_term_integrated_solid_waste_master_plan_na',
                           'legal_framework_solid_waste_management_rules_and_regulations_na',
                           'primary_collection_mode_form_of_primary_collection_na',
                           'separation_existence_of_source_separation_na']
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

# Fonction pour créer le préprocesseur
def create_preprocessor(numeric_features, categorical_features):
    print("Création du préprocesseur...")
    # Transformateur pour les variables numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Transformateur pour les variables catégorielles
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinaison des transformateurs
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Fonction pour définir les modèles à comparer
def define_models():
    print("Définition des modèles à comparer...")
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'SVR': SVR(kernel='rbf'),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'XGBoost': XGBRegressor.XGBRegressor(n_estimators=100, random_state=42)
    }
    return models

# Fonction pour évaluer les modèles avec validation croisée
def evaluate_models_cv(models, preprocessor, X_train, y_train, cv=5):
    print("Évaluation des modèles avec validation croisée...")
    results = {'Model': [], 'R2 Score': [], 'MSE': [], 'MAE': [], 'Explained Variance': [], 'Training Time': []}
    
    for name, model in models.items():
        print(f"Évaluation de {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Mesurer le temps d'entraînement
        start_time = time.time()
        
        # Validation croisée pour R²
        r2_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
        
        # Validation croisée pour MSE
        mse_scores = -cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        
        # Validation croisée pour MAE
        mae_scores = -cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
        
        # Validation croisée pour Explained Variance
        ev_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='explained_variance')
        
        training_time = time.time() - start_time
        
        # Enregistrer les résultats
        results['Model'].append(name)
        results['R2 Score'].append(r2_scores.mean())
        results['MSE'].append(mse_scores.mean())
        results['MAE'].append(mae_scores.mean())
        results['Explained Variance'].append(ev_scores.mean())
        results['Training Time'].append(training_time)
    
    # Convertir en DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R2 Score', ascending=False)
    
    return results_df

# Fonction pour évaluer les modèles sur l'ensemble de test
def evaluate_models_test(models, preprocessor, X_train, X_test, y_train, y_test):
    print("Évaluation des modèles sur l'ensemble de test...")
    results = {'Model': [], 'R2 Score': [], 'MSE': [], 'RMSE': [], 'MAE': [], 'Explained Variance': [], 'Training Time': []}
    
    for name, model in models.items():
        print(f"Évaluation de {name} sur l'ensemble de test...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Mesurer le temps d'entraînement
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prédictions sur l'ensemble de test
        y_pred = pipeline.predict(X_test)
        
        # Calcul des métriques
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        ev = explained_variance_score(y_test, y_pred)
        
        # Enregistrer les résultats
        results['Model'].append(name)
        results['R2 Score'].append(r2)
        results['MSE'].append(mse)
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['Explained Variance'].append(ev)
        results['Training Time'].append(training_time)
    
    # Convertir en DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R2 Score', ascending=False)
    
    return results_df

# Fonction pour visualiser les résultats
def visualize_results(cv_results, test_results):
    print("Visualisation des résultats...")
    
    # Créer un dossier pour les visualisations
    import os
    if not os.path.exists('visualisations'):
        os.makedirs('visualisations')
    
    # 1. Comparaison des scores R² (validation croisée)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='R2 Score', y='Model', data=cv_results)
    plt.title('Comparaison des scores R² (validation croisée)')
    plt.tight_layout()
    plt.savefig('visualisations/r2_scores_cv.png')
    
    # 2. Comparaison des scores R² (ensemble de test)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='R2 Score', y='Model', data=test_results)
    plt.title('Comparaison des scores R² (ensemble de test)')
    plt.tight_layout()
    plt.savefig('visualisations/r2_scores_test.png')
    
    # 3. Comparaison des RMSE (ensemble de test)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='RMSE', y='Model', data=test_results)
    plt.title('Comparaison des RMSE (ensemble de test)')
    plt.tight_layout()
    plt.savefig('visualisations/rmse_test.png')
    
    # 4. Comparaison des temps d'entraînement
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Training Time', y='Model', data=test_results)
    plt.title('Comparaison des temps d\'entraînement')
    plt.tight_layout()
    plt.savefig('visualisations/training_time.png')
    
    # 5. Graphique radar pour les 5 meilleurs modèles
    top_models = test_results.head(5)
    
    # Normaliser les métriques pour le graphique radar
    metrics = ['R2 Score', 'Explained Variance', 'Training Time']
    normalized_metrics = top_models[metrics].copy()
    
    # Pour R2 et Explained Variance, plus c'est élevé, mieux c'est
    for metric in ['R2 Score', 'Explained Variance']:
        min_val = normalized_metrics[metric].min()
        max_val = normalized_metrics[metric].max()
        normalized_metrics[metric] = (normalized_metrics[metric] - min_val) / (max_val - min_val)
    
    # Pour Training Time, plus c'est bas, mieux c'est
    min_time = normalized_metrics['Training Time'].min()
    max_time = normalized_metrics['Training Time'].max()
    normalized_metrics['Training Time'] = 1 - ((normalized_metrics['Training Time'] - min_time) / (max_time - min_time))
    
    # Créer le graphique radar
    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D
    
    def radar_chart(df, metrics, models):
        # Nombre de variables
        N = len(metrics)
        
        # Angle pour chaque variable
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Fermer le graphique
        
        # Initialiser la figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Ajouter les axes
        plt.xticks(angles[:-1], metrics, size=12)
        
        # Dessiner les polygones pour chaque modèle
        for i, model in enumerate(models):
            values = df.loc[df['Model'] == model, metrics].values.flatten().tolist()
            values += values[:1]  # Fermer le polygone
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Ajouter la légende
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Comparaison des performances des 5 meilleurs modèles', size=15)
        
        return fig, ax
    
    fig, ax = radar_chart(pd.concat([normalized_metrics, top_models[['Model']]], axis=1), metrics, top_models['Model'].tolist())
    plt.tight_layout()
    plt.savefig('visualisations/radar_chart.png')
    
    print("Visualisations sauvegardées dans le dossier 'visualisations'")

# Fonction principale
def main():
    # Chemin vers le fichier de données
    file_path = 'merged_data_city.csv'
    
    try:
        # Charger et préparer les données
        X_train, X_test, y_train, y_test, numeric_features, categorical_features = load_and_prepare_data(file_path)
        
        # Créer le préprocesseur
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        
        # Définir les modèles
        models = define_models()
        
        # Évaluer les modèles avec validation croisée
        cv_results = evaluate_models_cv(models, preprocessor, X_train, y_train)
        print("\nRésultats de la validation croisée:")
        print(cv_results)
        
        # Évaluer les modèles sur l'ensemble de test
        test_results = evaluate_models_test(models, preprocessor, X_train, X_test, y_train, y_test)
        print("\nRésultats sur l'ensemble de test:")
        print(test_results)
        
        # Sauvegarder les résultats
        cv_results.to_csv('resultats_validation_croisee.csv', index=False)
        test_results.to_csv('resultats_test.csv', index=False)
        
        # Visualiser les résultats
        visualize_results(cv_results, test_results)
        
        # Identifier le meilleur modèle
        best_model = test_results.iloc[0]['Model']
        best_r2 = test_results.iloc[0]['R2 Score']
        best_rmse = test_results.iloc[0]['RMSE']
        
        print(f"\nMeilleur modèle: {best_model}")
        print(f"R² Score: {best_r2:.4f}")
        print(f"RMSE: {best_rmse:.4f}")
        
        # Recommandation finale
        print("\nRecommandation:")
        print(f"Basé sur les performances, le modèle {best_model} est recommandé pour la prédiction de génération de déchets.")
        print("Cependant, considérez également les aspects suivants pour votre choix final:")
        print("1. Interprétabilité: Les modèles linéaires sont plus faciles à interpréter")
        print("2. Temps d'entraînement: Important pour les mises à jour fréquentes du modèle")
        print("3. Complexité de déploiement: Certains modèles sont plus faciles à déployer que d'autres")
        print("4. Besoins spécifiques: Certains modèles peuvent être plus adaptés à des caractéristiques particulières de vos données")
        
    except FileNotFoundError:
        print(f"Erreur: Le fichier {file_path} n'a pas été trouvé.")
        print("Veuillez placer le fichier de données dans le même répertoire que ce script ou modifier le chemin d'accès.")
    except Exception as e:
        print(f"Une erreur s'est produite: {str(e)}")

if __name__ == "__main__":
    main()
