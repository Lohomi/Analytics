#Decision Tree Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
#importing the dataset
df = pd.read_csv("Position_Salaries.csv")
x = df.iloc[:,1:2].values
y = df.iloc[:,2].values.astype("float")
#Fitting the decision tree model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
#Predictions
y_pred = regressor.predict(x)
y_expected = regressor.predict(np.array([[6.5]]))
#Accuracy Check
from sklearn.metrics import r2_score
R2 = r2_score(y,y_pred)
#Visulization
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title("Truth or Bluff(Decision Tree)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
