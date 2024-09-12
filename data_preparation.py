import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger le fichier CSV
file_path = './data/Dataset_salary_2024.csv'
df = pd.read_csv(file_path)

# Suppression des colonnes spécifiées
df = df.drop(columns=['work_year', 'salary_in_usd', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])

# Filtrer les lignes où 'salary_currency' est égal à 'USD'
df = df[df['salary_currency'] == 'USD']

# Supprimer la colonne 'salary_currency' car elle est désormais inutile
df = df.drop(columns=['salary_currency'])

# Modifier les intitulés de poste
df['job_title'] = df['job_title'].replace({
    'AI Developer': 'AI Engineer',
    'AI Engineer': 'AI Engineer',
    'AI Programmer': 'AI Engineer',
    'AI Research Engineer': 'AI Engineer',
    'AI Research Scientist': 'AI Engineer',
    'AI Scientist': 'AI Engineer',
    'AI Software Engineer': 'AI Engineer',
    'AI Architect': 'AI Manager',
    'AI Product Manager': 'AI Manager',
    'Data Analytics Consultant': 'Data Analytics Manager',
    'Data Analytics Engineer': 'Data Analytics Manager',
    'Data Analytics Specialist': 'Data Analytics Manager',
    'Business Intelligence': 'Machine Learning Engineer',
    'Business Intelligence Analyst': 'Machine Learning Engineer',
    'Business Intelligence Data Analyst': 'Machine Learning Engineer',
    'Business Intelligence Developer': 'Machine Learning Engineer',
    'Business Intelligence Engineer': 'Machine Learning Engineer',
    'Business Intelligence Lead': 'Machine Learning Engineer',
    'Business Intelligence Manager': 'Machine Learning Engineer',
    'Business Intelligence Specialist': 'Machine Learning Engineer',
    'ML Engineer': 'ML Engineer',
    'ML Ops Engineer': 'ML Engineer',
    'MLOps Engineer': 'ML Engineer',
    'Data Scientist': 'Data Scientist',
    'Lead Data Scientist': 'Data Scientist',
    'Principal Data Scientist': 'Data Scientist',
    'Staff Data Scientist': 'Data Scientist',
    'Head of Data Science': 'Head of Data Science',
    'Director of Data Science': 'Head of Data Science',
    'Managing Director Data Science': 'Head of Data Science',
    'ETL Developer': 'ETL Specialist',
    'ETL Engineer': 'ETL Specialist',
    'Robotics Engineer': 'Robotics Engineer',
    'Robotics Software Engineer': 'Robotics Engineer'
})

# Encodage des colonnes catégorielles
label_encoders = {}
categorical_columns = ['experience_level', 'employment_type', 'job_title']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Sauvegarder les encoders pour les utiliser lors de la prédiction
import joblib
joblib.dump(label_encoders, './models/label_encoders.pkl')

# Séparer les caractéristiques et la cible
X = df.drop(columns=['salary'])
y = df['salary']

# Vérifier les types de données avant la division
print(X.dtypes)

# Division des données en ensembles d'entraînement, de validation et de test
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Sauvegarder les ensembles
X_train.to_csv('./data/X_train.csv', index=False)
y_train.to_csv('./data/y_train.csv', index=False)
X_val.to_csv('./data/X_val.csv', index=False)
y_val.to_csv('./data/y_val.csv', index=False)
X_test.to_csv('./data/X_test.csv', index=False)
y_test.to_csv('./data/y_test.csv', index=False)

print("Les ensembles d'entraînement, de validation et de test ont été créés et sauvegardés avec succès.")
