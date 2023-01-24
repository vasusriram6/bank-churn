import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, render_template

app = Flask(__name__,template_folder='templates')


@app.route("/")
def home_page():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():


    CreditScore = float(request.form['CreditScore'])
    Geography = (request.form['Geography'])
    Gender = request.form['Gender']
    Age = float(request.form['Age'])
    Tenure = float(request.form['Tenure'])
    Balance = float(request.form['Balance'])
    NumOfProducts = float(request.form['NumOfProducts'])
    HasCrCard = float(request.form['HasCrCard'])
    IsActiveMember = float(request.form['IsActiveMember'])
    EstimatedSalary = float(request.form['EstimatedSalary'])

    model = pickle.load(open('churn_model.pkl', 'rb'))
    data = [[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]
    new_df = pd.DataFrame(data, columns=['CreditScore', 'Geography', 'Gender',
        'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])


    geomap={'France':0, 'Germany':1, 'Spain':2}
    new_df.Geography=new_df.Geography.map(geomap)

    gendermap={'Female':0, 'Male':1}
    new_df.Gender=new_df.Gender.map(gendermap)
    #df_encoded=new_df.replace({'Geography':{'France':0, 'Germany':1, 'Spain':2}}, {'Gender':{'Female':0, 'Male':1}}, inplace=True)


    single = model.predict(new_df.tail(1))
    probability = model.predict_proba(new_df.tail(1))[:, 1]

    print(single)
    

    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = "Confidence level is {}".format(probability*100)
    else:
        op1 = "This Customer is likely to Continue!"
        op2 = "Confidence level is {}".format((1-probability)*100)

    return render_template("index.html", op1=op1, op2=op2,
                           CreditScore=request.form['CreditScore'],
                           Geography=request.form['Geography'],
                           Gender=request.form['Gender'],
                           Age=request.form['Age'],
                           Tenure=request.form['Tenure'],
                           Balance=request.form['Balance'],
                           NumOfProducts=request.form['NumOfProducts'],
                           HasCrCard=request.form['HasCrCard'],
                           IsActiveMember=request.form['IsActiveMember'],
                           EstimatedSalary=request.form['EstimatedSalary'])


if __name__ == '__main__':
    app.run(debug=True)