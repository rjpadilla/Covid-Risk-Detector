"""
comorbidity-ml: a webapp to the users the chance of morbidity
"""
from flask import Flask, render_template, request, Response
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
import joblib
import numpy as np
import pandas as pd
import pygal
import io
import random


# Loads the machine learning model
df = pd.read_csv("data/comorbid.csv")
lr_cv_model = joblib.load("data/comorbid-trained-model.pkl")

app = Flask(__name__)


@app.route("/")
def home():
    """
    Function: home
    Input: none
    Returns: The main page of the webapp
    """
    return render_template("home.html")


@app.route("/survey")
def survey():
    """
    Function: entry
    Input: none
    Returns: The survey form to predict morbidity
    """
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

@app.route("/svg")
def svg():
    """
    Returns svg matplotlib
    """
    num_x_points = int(request.args.get("num_x_points", 50))
    return f"""
    # in a real app you probably want to use a flask template.
    <h1>Flask and matplotlib</h1>

    <h2>Random data with num_x_points={num_x_points}</h2>

    <form method=get action="/">
      <input name="num_x_points" type=number value="{num_x_points}" />
      <input type=submit value="update graph">
    </form>

    <h3>Plot as a SVG</h3>
    <img src="/matplot-as-image-{num_x_points}.svg"
         alt="random points as svg"
         height="200"
    >

    """
    
@app.route("/matplot-as-image-<int:num_x_points>.svg")
def plot_svg(num_x_points=50):
    """ renders the plot on the fly.
    """
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x_points = range(num_x_points)
    axis.plot(x_points, [random.randint(1, 30) for x in x_points])

    output = io.BytesIO()
    FigureCanvasSVG(fig).print_svg(output)
    return Response(output.getvalue(), mimetype="image/svg+xml")


if __name__ == "__main__":
    app.run(debug=True)
