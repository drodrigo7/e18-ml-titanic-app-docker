# app.py
# ==================================================
# --------------------------------------------------
# standard
import pickle, os
# requirements
from flask import Flask, render_template, request
import pandas as pd
# --------------------------------------------------

app = Flask(__name__)

with open('./model/XGBC_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def home():
    try:
        new_data = dict(
            Pclass   = int(request.form['Pclass']),
            Sex      = str(request.form['Sex']).lower(),
            Age      = int(request.form['Age']),
            SibSp    = int(request.form['SibSp']),
            Parch    = int(request.form['Parch']),
            Fare     = float(request.form['Fare']),
            Embarked = str(request.form['Embarked']).upper()
        )
        row = pd.DataFrame(new_data, index=[0])
        pred = int(model.predict(row)[0])
    except:
        pred = 9
    return render_template('results.html', data=pred)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=os.environ.get('PORT', 5000))
