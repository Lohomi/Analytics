#Polyniomial Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("Position_Salaries.csv")

x = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

#Not Needed as data set is very small
#depending on your version of Spyder/Jupyter it can be sklearn.preprocessing
'''from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
'''
#Linear Regression Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x,y)
y_pred_lm = lm.predict(x)

#Polynomial Linear Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly=poly_reg.fit_transform(x)
lm_2 = LinearRegression()
lm_2.fit(x_poly,y)
y_pred_poly = lm_2.predict(x_poly)
 #Visualizing Linear REgression Results
 plt.scatter(x,y,color='red')
 plt.plot(x,lm.predict(x),color='blue')
 plt.title("Truth or Bluff(Linear Regression)")
 plt.xlabel("Position Level")
 plt.ylabel("Salary")
 plt.show()
 #Visulaize Poly Reg
 plt.scatter(x,y,color='red')
 plt.plot(x,lm_2.predict(x_poly),color='blue')
 plt.title("Truth or Bluff(Polynomial Regression)")
 plt.xlabel("Position Level")
 plt.ylabel("Salary")
 plt.show()
 #ACCURACY
from sklearn.metrics import r2_score, mean_squared_error
LinearRegR2 = r2_score(y_pred_lm,y)
PolyRegR2 = r2_score(y_pred_poly,y)
MSELinear = mean_squared_error(y_pred_lm,y)
MSEPoly = mean_squared_error(y_pred_poly,y)

#R2 Score = 0.9808499387901439 MSE = 1515662004.662002 (Polynomial Regression, degree=3)
#R2 Score = 0.5053238120653423 MSE = 26695878787.878784 (Linear Regression)
