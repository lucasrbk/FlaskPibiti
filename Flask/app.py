from flask import Flask, render_template, url_for
from flask import *

import sklearn
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

UPLOAD_FOLDER = 'Flask/upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/success', methods = ['GET','POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename) 
    
    

    data = pd.read_csv("train.csv", sep=",")

    mass = data["mean_atomic_mass"]

    predict = "critical_temp"

    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])



    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    score = linear.score(x_test, y_test)

    print('Score', score)
    print('Coeficiente: \n', linear.coef_)

    p = "critical_temp"

    style.use("ggplot")
    pyplot.scatter(data[p], data["mean_atomic_mass"])
    pyplot.xlabel("Critical temperature")
    pyplot.ylabel("Atomic mass")
    pyplot.savefig("static/img/linearimg.png")

    return render_template("success.html", name = f.filename)  

if __name__ =='__main__':
    app.run(debug=True)