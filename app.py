from flask import Flask, request, url_for, redirect, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__, template_folder='./templates')
print("\n\n\nfinding model: \n\n\n")
model = load_model('./saved_models/saved_model.keras')


@app.route('/')
def hello_world():
    # Ensure the template name is just the filename if it's directly inside the 'templates' folder
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    print("\n\n\nRegularList:\n\n", int_features, "\n\n")
    
    # Reshape the array to include a batch dimension
    features_array = np.array([int_features])

    print("\ntype of features_array: ", type(features_array))
    
    # Model prediction
    prediction = model.predict(features_array)
    output = '{0:.{1}f}'.format(prediction[0][0], 1)

    if output == '1.0':  # Make sure the comparison is done with a string if output is formatted as string
        print("DIABETES")
        return render_template('index.html',pred='You have diabetes.')
    else:
        print("NO NOTHING!")
        return render_template('index.html', result="YIPEE! no diabetes")

if __name__ == '__main__':
    app.run(debug=True)
