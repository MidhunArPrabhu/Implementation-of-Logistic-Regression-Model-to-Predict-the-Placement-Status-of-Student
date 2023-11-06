# EXPERIMENT-04

# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated()function respectively.
3. LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.
## Program:
```py
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MIDHUN AZHAHU RAJA P
RegisterNumber:  212222240066
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#remove the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:

### 1.Placement Data;

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/5a443319-3279-448f-a8ef-4cd14d35e40c)

### 2.Salary Data:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/a397d08f-a0f3-457a-a3a5-6c06d0ac34d5)

### 3.Checking the null() function:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/b657cd95-3b9f-4ec7-bf39-9ecc814e52f0)

### 4.Data Duplicate:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/5dc77f17-28c1-4b8f-a9de-ab5b3287d45b)

### 5.Print Data:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/dbc4fbdc-fad9-427a-9513-19886e96e3b3)

### 6.Data-status:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/d393a2c4-8377-4e86-8c5f-57ee48bc88cb)

### 7.y_prediction array:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/bf17cf4a-4164-4d58-97d5-c6d5fa1f9b0e)

### 8.Accuracy Value:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/8fa23dce-8f39-4901-9b7a-563c7ba8c28b)

### 9.Confusion Array:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/6964ed6d-2ca3-44cf-8fd4-43461cbbe1c8)

### 10.Classification Report:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/407856dc-63ce-4eab-8e65-c1d6d561b129)

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/ed21bf3f-06b3-4f79-8894-868147ead0fe)

### 11.Prediction of LR:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393818/5fd200a3-748b-41ac-bf32-6d45d56330e8)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

