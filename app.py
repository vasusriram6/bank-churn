import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, render_template

app = Flask(__name__,template_folder='templates')

df_1=pd.read_csv("Churn_Modelling.csv")
df_dummy=pd.read_csv("bank_churn_dummy.csv")

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
    NumOfProducts = request.form['NumOfProducts']
    HasCrCard = request.form['HasCrCard']
    IsActiveMember = request.form['IsActiveMember']
    EstimatedSalary = float(request.form['EstimatedSalary'])

    model = pickle.load(open('churn_model.pkl', 'rb'))
    data = [[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]
    new_df = pd.DataFrame(data, columns=['CreditScore', 'Geography', 'Gender',
        'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

    df_2=df_1.drop(columns= ['RowNumber','CustomerId','Surname','Exited'], axis=1, inplace=True)

    
    
    #df_dummy.columns
    #dfd=df_dummy.drop(columns=['Unnamed: 0','Exited'], axis=1, inplace=True)
    
    
    df_3 = pd.concat([df_2, new_df], ignore_index = True)
    #df_3 = df_2.drop(columns= ['RowNumber','CustomerId','Surname','Exited'], axis=1, inplace=True)
    

    #df_2.drop(columns= ['Exited'], axis=1, inplace=True)

    #categorical_feature = {feature for feature in new_df.columns if new_df[feature].dtypes == 'O'}

    #encoder = LabelEncoder()
    #for feature in categorical_feature:
    #    new_df[feature] = encoder.fit_transform(new_df[feature])

    #bins = [18,24,30,36,42,48,54,60,66,72,78,84,90,96]

    #labels = ['18-23','24-29','30-35','36-41','42-47','48-53','54-59','60-65','66-71','72-77','78-83','84-89','90-95']

    #df_1['Age_group'] = pd.cut(df_1['Age'], bins=bins, labels=labels)
    
    #df_1.drop(columns= ['RowNumber','CustomerId','Surname','Age','Exited'], axis=1, inplace=True)   
     

    new_df__dummies = pd.get_dummies(df_3)#[['CreditScore', 'Geography',
       #'Gender', 'Age','Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       #'IsActiveMember', 'EstimatedSalary']])

    
    print(new_df__dummies.shape)

    single = model.predict(new_df__dummies.tail(1))
    probability = model.predict_proba(new_df__dummies.tail(1))[:, 1]

    print(single)
    

    if single == 1:
        op1 = "This Customer is likely to be Churned!"
        op2 = "Confidence level is {}".format(probability*100)
    else:
        op1 = "This Customer is likely to Continue!"
        op2 = "Confidence level is {}".format(probability*100)

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