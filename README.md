# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Importing Libraries
2.Reading Data from CS
3.Inspecting the Data
4.Label Encoding the 'salary' Column
5.Selecting Features (X)
6.Selecting Target Variable (Y)
7.Splitting the Data into Training and Testing Sets
8.Training the Decision Tree Model
9.Making Predictions
10.Evaluating Model Accuracy
11.Making a Single Prediction



## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MAGESH BOOPATHI.M
RegisterNumber: 24900855 
```
```python

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred= dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]]) 
```

## Output:
![image](https://github.com/user-attachments/assets/3b6d8cd6-02db-4f9b-b1fb-402909b8794c)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
