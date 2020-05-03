# Analytics
Hereby lies various insights as well as Visualizations of the Models

## Linear Model Visualization
### Training Data 
![Regression Line for Training Data](https://github.com/Lohomi/Analytics/blob/master/Training%20Data.png)

### Test Data
![TestData](https://github.com/Lohomi/Analytics/blob/master/Test%20Set.png)

We can see that the dataset we had in the above example had only a signle independent variable, which is rarely the case when you do any   hands on project/work, let us have a look at a Linear Mode that incorporates these multiple independent variables (check: MultLin.py)
  - **What is Dummy Variable Trap?**
  > The [Dummy Variable](https://www.algosome.com/articles/dummy-variable-trap-regression.html) trap is a scenario in which the independent variables are multicollinear - a scenario in which two or more        variables are highly correlated; in simple terms one variable can be predicted from the others. 
## Polynomial Linear Regression
First let us see how our simple linear model fits the data :

![](https://github.com/Lohomi/Analytics/blob/master/LinearReg(PolyModel).png)

We can see that the linear model does not fit the data at all, we get a R2 score of **0.50**, which is not acceptable as we want to make robust predictions, after that we fit the polynomial regression of degree 3 and hence see the visualization.

![](https://github.com/Lohomi/Analytics/blob/master/PolyRegVisualization(degree3).png)

So here we see a well fitting curve, now robust predictions can be made from this model, we can make the fitting even more accurate by increasing the Degree of the polynomia, but you will have to make sure that you are avoiding **Overfitting** of data, Overfitting leads to **high accuracy** in the **Training set** but very **poor predictions** in any new data, its often reffered to as High Variance.
