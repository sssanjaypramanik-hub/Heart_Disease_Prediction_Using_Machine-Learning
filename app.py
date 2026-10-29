from flask import Flask, request, render_template
import pickle
import numpy as np 

app=Flask(__name__)

with open("svu.pkl", 'rb') as model_file:
	ml_model=pickle.load(model_file)

@app.route("/")
def index():
	return render_template('index.html')


@app.route("/process", methods=["POST"])
def process():

    features = [
        float(request.form["Age"]),
        float(request.form["Sex"]),
        float(request.form["Chest pain type"]),
        float(request.form["BP"]),
        float(request.form["Cholesterol"]),
        float(request.form["FBS over 120"]),
        float(request.form["EKG results"]),
        float(request.form["Max HR"]),
        float(request.form["Exercise angina"]),
        float(request.form["ST depression"]),
        float(request.form["Slope of ST"]),
        float(request.form["Number of vessels fluro"]),
        float(request.form["Thallium"])
    ]

    input_data = np.array(features).reshape(1, -1)

    result = ml_model.predict(input_data)[0]

    diagnosis = "Heart Disease" if result == 1 else " No Heart Disease"

    return render_template("myresult.html", prediction=diagnosis)











if __name__=='__main__':
	app.run(debug=True)