from flask import Flask,request, url_for, redirect, render_template
from tensorflow.keras.models import load_model


import numpy as np

app = Flask(__name__)

model=load_model('../saved_models/saved_model.keras')


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][0], 1)
                              
                              # Make a prediction


    if output == 1.0:
        return render_template('Im sorry you have diabetes'.format(output),bhai="Your Forest is Safe for now")
    else:
        return render_template('YIPEE! no diabetes'.format(output),bhai="Your Forest is Safe for now")



if __name__ == '__main__':
    app.run(debug=True)