import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib

# Charger les ensembles d'entraînement et de validation
X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv').values.ravel()
X_val = pd.read_csv('./data/X_val.csv')
y_val = pd.read_csv('./data/y_val.csv').values.ravel()

# Choix du modèle - Random Forest Regressor
model = RandomForestRegressor(random_state=42)

# Étape 1 : Validation croisée sur l'ensemble d'entraînement
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = (-cv_scores.mean()) ** 0.5

print(f"Validation Croisée RMSE sur l'ensemble d'entraînement: {cv_rmse}")

# Étape 2 : Recherche d'hyperparamètres avec GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Meilleurs hyperparamètres
best_params = grid_search.best_params_
best_score = (-grid_search.best_score_) ** 0.5

print(f"Meilleurs paramètres: {best_params}")
print(f"Meilleur RMSE après optimisation: {best_score}")

# Étape 3 : Réentraîner le modèle avec les meilleurs hyperparamètres
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Évaluation finale sur l'ensemble de validation
y_val_pred = best_model.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5

print(f"RMSE sur l'ensemble de validation: {val_rmse}")

# Sauvegarder le modèle optimisé
joblib.dump(best_model, './models/optimized_salary_prediction_model.pkl')
print("Modèle optimisé sauvegardé.")