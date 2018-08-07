"""
A ML model built on the top of Polynomial Regressor in order to predict Salary of an employee on the basis of his Position and Level 
"""
# base imports 
import numpy as np # for matrix maths 
import pandas as pd # for data handeling 
import matplotlib.pyplot as plt # for data viz 

# sklearn imports 
from sklearn.preprocessing import PolynomialFeatures # for polynomial features 
from sklearn.model_selection import train_test_split # to split train test data 
from sklearn.pipeline import make_pipeline # to make pipe line of actions

from sklearn.linear_model import LinearRegression

# importing data set 
data = pd.read_csv('./Position_Salaries.csv')
# extracting features 
features = data[['Level']]
# output features 
output =  data.Salary

# breaking data into test and train 
X_train, X_test,y_train,y_test = train_test_split(features,output,random_state=0, test_size=1/3)

# making a linear regression model 
lig_reg = LinearRegression()
lig_reg.fit(X_train,y_train) # fitting lig_reg model 

# making a pipeline for polynomial features 
model = make_pipeline(PolynomialFeatures(degree=4),LinearRegression())
model.fit(X_train,y_train)

# making a viz for linear regression 
plt.scatter(features,output, color='red')
plt.plot(features,lig_reg.predict(features))
plt.title('Position Salaries')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show();

# making data viz for our polynomial model 
plt.scatter(features,output, color='red')
plt.plot(features,model.predict(features))
plt.title('Position Salaries')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show();

# printing score of each algo 
print("Score of Linear regression is {}".format(lig_reg.score(features,output)))
print("Score of Polynomial regression is {}".format(model.score(features,output)))