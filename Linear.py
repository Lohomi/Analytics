"""
dataset : https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P14-Part2-Regression.zip
"""
```python
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:06:46 2020

@author: Parth
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Salary_Data.csv")
df.head()
# Check for missing Data
df.isnull().sum()
x = df["YearsExperience"]
y = df["Salary"]
plt.scatter(x,y)
plt.xlim(0,)
plt.ylim(0,)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Train Test Split
from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1].values
Y = df.iloc[:,1].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 0)

#Fitting Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

#Predicting for X_test
Y_pred = lr.predict(X_test)
#Visualize
plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train,lr.predict(X_train),color = 'blue') ##Regression Line
plt.title("Salary v/s Experience (Training Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

#Test Data Visualization
plt.scatter(X_test,Y_test, color = 'red')
plt.plot(X_test,Y_pred,color = 'blue') ##Regression Line
plt.title("Salary v/s Experience (Test Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()
 # Accuracy
rss=((Y_test-Y_pred)**2).sum()
mse=np.mean((Y_test-Y_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((Y_test-Y_pred)**2)))

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test,Y_pred)
```
RMSE = 3580.979237321343
