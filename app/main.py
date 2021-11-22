from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = pickle.load(open('app/model.pkl', 'rb'))
modelSal = pickle.load(open('app/modelsalary.pkl', 'rb'))


@app.route('/')
def index():
    return 'Welcome to career predictor'


@app.route('/api/v1/ic/current-job', methods=['POST'])
def initial_career():
    output = {'initial_career': '','initial_salray_range':''
              }
    int_features = [x for x in request.form.values()]
    print(int_features)
    final = np.array(int_features)
    data_unseen = pd.DataFrame(
        [final], columns=['Gender', 'Age', 'Marital Status', 'Ethnicity', 'Home Town', 'A/L Stream', 'Subject_1', 'Result_1', 'Subject_2', 'Result_2', 'Subject_3', 'Result_3', 'Univesity/Degree Awarding Institute', 'Degree', 'Duration Of Degree', 'Graduation Year', 'GPA', 'Knowledge on programming concepts', 'Knowledge on programming languages', 'Knowledge on software engineering concepts', 'Knowledge on UI/UX engineering concepts', 'Knowledge on UI/UX interface', 'Fluent communication English', 'Problem Solving', 'Creativity', 'Self-learning', 'Management', 'Team Playing', 'Decision Making'])

    prediction = model.predict(data_unseen)
    predictionSal = modelSal.predict(data_unseen)
    output = {'initial_career': str(prediction[0]),'initial_salray_range':str(predictionSal[0]),
              }
    return output





if __name__ == "__main__":
    app.run(debug=True)
