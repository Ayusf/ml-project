from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Creating an instance of CustomData
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race/ethnicity'),
            parental_level_of_education=request.form.get('parental level of education'),  # Fixed field name
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test preparation course'),  # Fixed field name
            reading_score=float(request.form.get('reading score')),
            writing_score=float(request.form.get('writing score'))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()

        # Creating an instance of PredictPipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Return the result to home.html
        return render_template('home.html', results=round(results[0], 2))  # Rounded for cleaner output

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
