from flask import Flask,render_template,request

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictionPipeline

app = Flask(__name__)

#Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictData' , methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch_type=request.form.get('lunch_type'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictionPipeline()
        try:
            results = predict_pipeline.predict(features=pred_df)
        except Exception as e:
            return f"Prediction error: {str(e)}"

        return render_template('home.html' , results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")