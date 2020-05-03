# Explanations and Insights to various Machine Learning Models
Hereby lies various insights as well as Visualizations of the Models

## Linear Model Visualization
### Training Data 
![Regression Line for Training Data](https://github.com/Lohomi/Analytics/blob/master/Training%20Data.png)

### Test Data
![TestData](https://github.com/Lohomi/Analytics/blob/master/Test%20Set.png)

We can see that the dataset we had in the above example had only a single independent variable, which is rarely the case when you do any   hands on project/work, let us have a look at a Linear Model that incorporates these multiple independent variables (check: MultLin.py)
  - **What is Dummy Variable Trap?**
  > The [Dummy Variable](https://www.algosome.com/articles/dummy-variable-trap-regression.html) trap is a scenario in which the independent variables are multicollinear - a scenario in which two or more        variables are highly correlated; in simple terms one variable can be predicted from the others. 
  ## What is Backward Elimination?
  When we’re building a machine learning model, it is very important that we select only those features or predictors which are necessary. Suppose we have 100 features or predictors in our dataset. That doesn’t necessarily mean that we need to have all 100 features in our model. This is because not all 100 features will have significant influence on the model. But then again, this doesn’t mean it will be true for all cases. It depends entirely on the data we have in hand. Here is more info about why we need feature selection.
There are various ways in which you can find out which features have very little impact on the model and which ones you can remove from your dataset, we’ll look at Backward Elimination and how we can do this, step by step. But before we start talking about backward elimination, make sure you make yourself familiar with P-value.

![Procedure Flow](https://miro.medium.com/max/1400/1*Jub_nEYtN0htxFpTRzRtBQ.png)

### Step 1
The first step in backward elimination is pretty simple, you just select a significance level, or select the P-value. Usually, in most cases, a 5% significance level is selected. This means the P-value will be 0.05. You can change this value depending on the project.
### Step 2
The second step is also very simple. You simply fit your machine learning model with all the features selected. So if there are 100 features, you include all of them in your model and fit the model on your test dataset. No changes here.
### Step 3
In step 3, identify the feature or predictor which has the highest P-value. Pretty simple again, right?
### Step 4
This is a significant step. Here, we take decisions. In the previous step, we identified the feature which has the highest P-value. If the P-value of this feature is greater than the significance level we selected in the first step, we remove this feature from our dataset. If the P-value of this feature, which is the highest in the set, is less than the significance level, we’ll just jump to Step 6, which means that we’re done. Remember, highest P-value greater than significance level, remove that feature.
### Step 5
Once we find out the feature which has to be removed from the dataset, we’ll do that in this step. So we remove the feature from the dataset, and we’ll fit the model again with the new dataset. After fitting the model for the new dataset, we’ll jump back to step 3.
This process continues until we reach a point in step 4 where the highest P-value from all the remaining features in the dataset is less than the significance selected in step 1. In our example, this means we iterate from step 3 to step 5 and back till the highest P-value in the dataset is less than 0.05. This could take a while. Out of the 100 assumed features, we might filter out a good 10 features this way (which is just a random number I selected). Refer the flowchart at the top of this post to get a better idea of these steps.
### Step 6
Once we reach step 6, we’re done with the feature selection process. We have successfully used backward elimination to filter out features which were not significant enough for our model.

- Let's have a look at the `code`
```python
import statsmodels.formula.api as sm
X = np.append(arr = X, values = np.ones(rows,columns),axis =1) #for a linear model we need to add a column which corresponds to the constant term b_0 in **y = b_0+b_1*x_1+ ... **
X_opt = X[rows,columns]
reg = sm.OLS(endog = y, exog = X_opt).fit()
reg.summary #Look for P-value of the independent variables
```

## Polynomial Linear Regression
First let us see how our simple linear model fits the data :

![](https://github.com/Lohomi/Analytics/blob/master/LinearReg(PolyModel).png)

We can see that the linear model does not fit the data at all, we get a R2 score of **0.50**, which is not acceptable as we want to make robust predictions, after that we fit the polynomial regression of degree 3 and hence see the visualization.

![](https://github.com/Lohomi/Analytics/blob/master/PolyRegVisualization(degree3).png)

So here we see a well fitting curve, now robust predictions can be made from this model, we can make the fitting even more accurate by increasing the Degree of the polynomia, but you will have to make sure that you are avoiding **Overfitting** of data, Overfitting leads to **high accuracy** in the **Training set** but very **poor predictions** in any new data, its often reffered to as High Variance.

| Model | R2 Score | MSE |
| ----- |:--------:|----:|
|Linear Regression Model | 0.505 | 26695878787.878784 |
|Polynomial Linear Regression | 0.980 | 1515662004.662002 |
