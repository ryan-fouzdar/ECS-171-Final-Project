from flask import Flask, request, url_for, redirect, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__, template_folder='./templates')
print("\n\n\nfinding model: \n\n\n")
print("running gpt.keras")
model = load_model('./saved_models/gpt.keras')


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
            return render_template('index.html',pred='Make sure to input float values for every input!')
        


    
    # Reshape the array to include a batch dimension
    features_array = np.array([int_features])
    print("\ntype of features_array: ", type(features_array))
    
    # Model prediction
    prediction = model.predict(features_array)
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
