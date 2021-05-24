import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
data = pd.read_csv("dataset.csv")
print('Shape of dataset', data.shape)
print('No of columns', data.columns)
obj_cols = [i for i in data.columns if data[i].dtype == np.object0]
data = data.drop('Loan_ID', axis = 1)
obj_cols.remove('Loan_ID')
for i in obj_cols:
    data[i] = data[i].fillna(value = data[i].mode()[0])
data.LoanAmount = data.LoanAmount.fillna(value=data.LoanAmount.mean())
data['Credit_History'] = data['Credit_History'].fillna(value = data['Credit_History'].mode()[0])
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(value = data['Loan_Amount_Term'].mean())
data.Gender = data.Gender.map({'Male':1, 'Female':0})
data.Married = data.Married.map({'Yes':1, 'No':0})
data.Education = data.Education.map({'Graduate':1, 'Not Graduate':0})
data.Self_Employed = data.Self_Employed.map({'Yes':1, 'No':0})
data.Dependents = data.Dependents.map({'0':0, '1':1, '2':2, '3':3, '3+':5})
data.Loan_Status = data.Loan_Status.map({'Y':1, 'N':0})
data.Property_Area = data.Property_Area.map({'Urban':3, 'Semiurban':2,'Rural':1 })

X = data.drop('Loan_Status', axis = 1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.1 )

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(confusion_matrix(y_test, model.predict(X_test)), accuracy_score(y_test, model.predict(X_test)))
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)