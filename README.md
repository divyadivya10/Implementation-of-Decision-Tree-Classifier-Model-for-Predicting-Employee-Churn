# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Divya R
RegisterNumber:  212222040040
*/
```

```
import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
y.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
## data.head()
![image](https://github.com/user-attachments/assets/16018307-bf86-425d-a459-9f9279180beb)
## value_count
![image](https://github.com/user-attachments/assets/9c13d31e-0c03-498a-acac-5bad34ea8c56)
## data salary
![image](https://github.com/user-attachments/assets/0a7ad893-8b60-4668-8941-64dbfe677ee8)
## x.head()
![image](https://github.com/user-attachments/assets/40f4595c-c197-4035-ab70-3485507668c5)
## y.head()
![image](https://github.com/user-attachments/assets/6f84f6eb-f771-44b8-801c-002a5ed3d169)
## Accuracy
![image](https://github.com/user-attachments/assets/3a813356-b8ce-465e-9b8d-2d1f168163ba)
## Prediction
![image](https://github.com/user-attachments/assets/a0ac5ca9-9577-4fa8-ad31-16073fb82277)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
