from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Charger le modèle et les encoders
model = joblib.load('./models/optimized_salary_prediction_model.pkl')
label_encoders = joblib.load('./models/label_encoders.pkl')

experience_level_mapping = {
    'EN': 'Entry level',
    'MI': 'Mid level',
    'SE': 'Senior',
    'EX': 'Executive'
}

employment_type_mapping = {
    'FT': 'Full time',
    'PT': 'Part time',
    'FL': 'Freelance'
}

@app.route('/', methods=['GET', 'POST'])
def predict_salary():
    experience_levels = label_encoders['experience_level'].classes_
    employment_types = label_encoders['employment_type'].classes_
    job_titles = label_encoders['job_title'].classes_

    if request.method == 'POST':
        experience_level = label_encoders['experience_level'].transform([request.form['experience_level']])[0]
        employment_type = label_encoders['employment_type'].transform([request.form['employment_type']])[0]
        job_title = label_encoders['job_title'].transform([request.form['job_title']])[0]

        # Créer le tableau de données d'entrée
        data = {
            'experience_level': experience_level,
            'employment_type': employment_type,
            'job_title': job_title
        }

        # Prédire le salaire
        prediction = model.predict([list(data.values())])
        return render_template('result.html', prediction=prediction[0])

    return render_template(
        'index.html', 
        experience_levels=experience_levels, 
        employment_types=employment_types, 
        job_titles=job_titles, 
        experience_level_mapping=experience_level_mapping,
        employment_type_mapping=employment_type_mapping,
    )

if __name__ == '__main__':
    app.run(debug=True)