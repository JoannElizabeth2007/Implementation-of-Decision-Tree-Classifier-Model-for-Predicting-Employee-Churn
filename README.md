# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.import pandas module and import the required data seT.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.
```
## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:   JOANN ELIZABETH SAMUEL
RegisterNumber:  212224040139

```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:

![image](https://github.com/user-attachments/assets/6f42bdc9-cbcd-4960-8072-98fff0531740)

![image](https://github.com/user-attachments/assets/7bd8192f-818f-42b1-bdfd-f34dfaff0e7d)

![image](https://github.com/user-attachments/assets/cbc67134-8ae7-412e-88b1-c36bc46c40a0)

![image](https://github.com/user-attachments/assets/b8d613c2-c58c-44ce-b5fb-c3ff11c8fa66)

![image](https://github.com/user-attachments/assets/f1f7c447-a17b-4561-b64b-9a1beed88d53)

![image](https://github.com/user-attachments/assets/d543183e-decf-44eb-a43c-dbb9ddb4c8a6)

![image](https://github.com/user-attachments/assets/fc0454bf-a9f0-4a9b-90bc-79ef2cd4e856)

![image](https://github.com/user-attachments/assets/e7539f8f-8f0c-41e3-b78c-13f30570a405)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
