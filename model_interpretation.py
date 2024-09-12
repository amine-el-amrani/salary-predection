import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

# Charger le modèle optimisé
model_filename = './models/optimized_salary_prediction_model.pkl'
best_model = joblib.load(model_filename)

# Charger les données nettoyées
file_path = './data/cleaned_salary_data.csv'
df = pd.read_csv(file_path)

# Séparation des caractéristiques (features) et de la cible (target)
X = df.drop(columns=['salary'])
y = df['salary']

# Étape 1 : Importance des caractéristiques
feature_importances = best_model.feature_importances_
features = X.columns

# Créer un DataFrame pour organiser les importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Afficher les importances des caractéristiques
print("Importance des caractéristiques :")
print(importance_df)

# Visualiser les importances des caractéristiques
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importance de la caractéristique")
plt.ylabel("Caractéristique")
plt.title("Importance des caractéristiques dans le modèle RandomForest")
plt.gca().invert_yaxis()
plt.show()

# Étape 2 : Utilisation des valeurs SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)

# Résumé des valeurs SHAP (global)
shap.summary_plot(shap_values, X, plot_type="bar")

# SHAP pour une prédiction spécifique (exemple pour la première ligne)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])
