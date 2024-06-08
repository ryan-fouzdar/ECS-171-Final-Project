from flask import Flask, request, url_for, redirect, render_template
from tensorflow.keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__, template_folder='./templates')
print("\n\n\nfinding model: \n\n\n")
print("running gpt.keras")
model = load_model('./saved_models/better_model.keras')


@app.route('/')
def hello_world():
    # Ensure the template name is just the filename if it's directly inside the 'templates' folder
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = []
    for x in request.form.values():
        try:
            int_features.append(float(x))
        except ValueError:
            return render_template('index.html',pred='Make sure to input numeric values for every input!')
    
    #through the values in int_features into the same scalar used from the notebook
    from joblib import load

    # Load the scaler
    scaler = load('./saved_scalars/scaler.joblib')
    dtc = joblib.load('./saved_models/dtc.joblib')

    # Use the loaded scaler to transform the test set or new data
    int_features = scaler.transform([int_features])
    print('transformed values: ', int_features)




    # Reshape the array to include a batch dimension
    features_array = np.array([int_features])
    

  

    features_array_dtc = np.array(int_features[0]).reshape(1,-1)
    print(features_array_dtc)
    prediction_dtc = dtc.predict(features_array_dtc)
    print("\n\nDTC prediction: ", prediction_dtc)
    
    # Model prediction
    prediction = model.predict(features_array[0])
    print("prediction: ",prediction)
    output = '{0:.{1}f}'.format(prediction[0][0], 1)

    print("\n\n\n\nPrediction tables: ",prediction, "\n\n\nPrediction is: ",output,"\n\n",print(type(output)))
    output = float(output)

    if output > 0.5:  # Make sure the comparison is done with a string if output is formatted as string
        
        print("val < 0.5 = DIABETES")
        return render_template('index.html',pred='You have diabetes.')
    else:
        print("NO NOTHING!")
        return render_template('index.html',pred="YIPEE! no diabetes")

if __name__ == '__main__':
    app.run(debug=True)
