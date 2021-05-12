"""
comorbidity-ml: a webapp to the users the chance of morbidity
"""
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import pygal


# Loads the machine learning model
df = pd.read_csv("data/comorbid.csv")
lr_cv_model = joblib.load("data/comorbid-trained-model.pkl")

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/survey")
def survey():
    return render_template("survey.html")


@app.route('/predict', methods=['POST'])
def result():
    """
    Function: result
    Input: none
    Returns: The result page of the user's prediction
    """
    letters = ['age', 'sex', 'smoking', 'healthcare_worker', 'hypertension',
               'diabetes', 'dementia', 'cancer', 'copd', 'asthma',
               'chd', 'ccd', 'cnd', 'cld', 'ckd']
    pred_date = [int(request.form[letter]) for letter in letters]
    arr = np.array([pred_date])
    pred = lr_cv_model.predict(arr)
    return render_template('results.html', data=pred)


@app.route('/charts')
def charts() -> render_template:
    """
    Function: charts
    Input: none
    Returns: Embedding a pygal chart in the webapp
    """
    male_deaths = len(df[(df.death == 1) & (df.sex == 1)])
    female_deaths = len(df[(df.death == 1) & (df.sex == 0)])
    # Covid cases by mortality
    bar_chart = pygal.Bar()
    bar_chart.title = "Deaths by Gender"
    bar_chart.add('male', male_deaths)
    bar_chart.add('female', female_deaths)
    bar_data = bar_chart.render_data_uri()
    return render_template('charts.html', bar_data=bar_data)


if __name__ == "__main__":
    app.run(debug=True)
