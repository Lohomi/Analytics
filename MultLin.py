#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset

df = pd.read_csv('50_startups.csv')
x = df.iloc[: , :-1].values
y = df.iloc[: , 4].values

#Convert categorical data by Label Encode/One-Hot Encoding

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:,3] = labelEncoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding Dummy Variable Trap (See Readme for info on Dummy Variable Trap)

x = x[:,1:]

#Test Train Split to be made

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Fitting the Regression

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
y_pred = lm.predict(x_test)

#Calculate the accuracy(r2 score and MSE)

from sklearn.metrics import r2_score,mean_squared_error
r2_accuracy = r2_score(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)

#r2_score = 0.9347068473282965 MSE = 83502864.0325083
