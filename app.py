import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Load the Random Forest CLassifier model
filename = 'diabetes_prediction_model.pkl'
model = pickle.load(open(filename, 'rb'))


app = Flask(__name__)
dataset = pd.read_csv('diabetes.csv')
X=dataset.drop('Outcome',axis=1)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X)
Standardized_data=scaler.transform(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( scaler.transform(final_features) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
