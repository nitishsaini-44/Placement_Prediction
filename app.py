from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)

# Load models
RF_model = pickle.load(open('models/rf.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            IQ = float(request.form['IQ'])
            Prev_Sem_Result = float(request.form['Prev_Sem_Result'])
            CGPA = float(request.form['CGPA'])
            Academic_Performance = float(request.form['Academic_Performance'])
            Internship_Experience = float(request.form['Internship_Experience'])
            Extra_Curricular_Score = float(request.form['Extra_Curricular_Score'])
            Communication_Skills = float(request.form['Communication_Skills'])
            Projects_Completed = float(request.form['Projects_Completed'])

            input_data = np.array([[IQ, Prev_Sem_Result, CGPA, Academic_Performance,
                                    Internship_Experience, Extra_Curricular_Score,
                                    Communication_Skills, Projects_Completed]])
            scaled_data = scaler_model.transform(input_data)
            prediction = RF_model.predict(scaled_data)

            if prediction[0] == 1:
                message = "🔥 Congratulations! You are highly eligible to be placed."
            else:
                message = "🔥 Improve on your skills; the current ones are not enough."
        except Exception as e:
            message = f"Error: {e}"
        return render_template('index.html', prediction=message)
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        IQ = float(data['iq'])
        Prev_Sem_Result = float(data['prevSemResult'])
        CGPA = float(data['cgpa'])
        Academic_Performance = float(data['academicPerformance'])
        Internship_Experience = float(data['internshipExperience'])
        Extra_Curricular_Score = float(data['extracurricular'])
        Communication_Skills = float(data['communicationSkills'])
        Projects_Completed = float(data['projectsCompleted'])

        input_data = np.array([[IQ, Prev_Sem_Result, CGPA, Academic_Performance,
                                Internship_Experience, Extra_Curricular_Score,
                                Communication_Skills, Projects_Completed]])
        scaled_data = scaler_model.transform(input_data)
        prediction = RF_model.predict(scaled_data)

        if prediction[0] == 1:
            message = "🔥 Congratulations! You are highly eligible to be placed."
        else:
            message = "🔥 Improve on your skills; the current ones are not enough."
        return {'message': message}
    except Exception as e:
        return {'message': f"Error: {e}"}, 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
