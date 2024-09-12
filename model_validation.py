import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

# Charger l'ensemble de test
X_test = pd.read_csv('./data/X_test.csv')
y_test = pd.read_csv('./data/y_test.csv').values.ravel()

# Charger le modèle optimisé
model = joblib.load('./models/optimized_salary_prediction_model.pkl')

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul du RMSE sur l'ensemble de test
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"RMSE sur l'ensemble de test: {rmse}")