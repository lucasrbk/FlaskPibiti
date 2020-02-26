from flask import Flask, render_template, url_for
from flask import *

import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import joblib
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
def index():
    return render_template('base.html')

@app.route('/linear', methods = ['GET','POST'])  
def linear():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename) 
    
    
    
    data = pd.read_csv("train.csv", sep=",")

    predict = "critical_temp"

    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])



    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    score = linear.score(x_test, y_test)
    tabela = data['number_of_elements']
    count = tabela.count()


    #print('Score', score)
    #print('Coeficiente: \n', linear.coef_)

    temp = data['critical_temp']
    atomicMass = data['mean_atomic_mass']
    condutividade = data['mean_ThermalConductivity']
    eletric = data['mean_ElectronAffinity']
    valence = data['mean_Valence']
    raio = data['mean_atomic_radius']

    pyplot.plot(temp.sort_values(), atomicMass.sort_values(), color='green', label='Massa atômica')
    pyplot.plot(temp.sort_values(), condutividade.sort_values(), color='orange', label= 'condutividade')
    pyplot.plot(temp.sort_values(), eletric.sort_values(), color='blue', label= 'afinidade eletrônica')
    pyplot.plot(temp.sort_values(), raio.sort_values(), color='black', label= 'Raio atômico')
    pyplot.plot(temp.sort_values(), temp.sort_values(), color='red', label= 'Temperatura')

    pyplot.xlabel('Temperatura')
    #plt.ylabel('Massa atômica')
    pyplot.title('Relação temperatura critica')
    pyplot.legend(loc="upper left")
    pyplot.savefig("static/img/linearimg.png")

    return render_template("linear.html", name = f.filename, score= score, count=count)  

@app.route('/knn', methods = ['GET','POST'])  
def knn():  
        if request.method == 'POST':  
            f = request.files['file']  
            f.save(f.filename) 

        data = pd.read_csv("train.csv", sep=",")

        le = preprocessing.LabelEncoder()
        cls = le.fit_transform(list(data["critical_temp"]))

        predict = "critical_temp"

        x = np.array(data.drop([predict], 1))
        y = np.array(data[predict])

        y = list(cls)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5)

        model = KNeighborsClassifier(n_neighbors=1)

        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print(score)
        print('Coeficiente: \n', model.n_neighbors)

        tabela = data['number_of_elements']
        count = tabela.count()
        temp = data['critical_temp']
        atomicMass = data['mean_atomic_mass']
        condutividade = data['mean_ThermalConductivity']
        eletric = data['mean_ElectronAffinity']
        raio = data['mean_atomic_radius']

        pyplot.plot(temp.sort_values(), atomicMass.sort_values(), color='green', label='Massa atômica')
        pyplot.plot(temp.sort_values(), condutividade.sort_values(), color='orange', label= 'condutividade')
        pyplot.plot(temp.sort_values(), eletric.sort_values(), color='blue', label= 'afinidade eletrônica')
        pyplot.plot(temp.sort_values(), raio.sort_values(), color='black', label= 'Raio atômico')
        pyplot.plot(temp.sort_values(), temp.sort_values(), color='red', label= 'Temperatura')

        pyplot.xlabel('Temperatura')
        #plt.ylabel('Massa atômica')
        pyplot.title('Relação temperatura critica')
        pyplot.legend(loc="upper left")
        pyplot.savefig("static/img/knnimg.png")
        
        return render_template("knn.html", name = f.filename, score= score, count= count)  

if __name__ =='__main__':
    app.run(debug=True)