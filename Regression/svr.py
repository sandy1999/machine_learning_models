"""
A ML model built on the top of Support Vector Regressor in order to predict Salary of an employee on the basis of his Position and Level 
"""
# base import
import numpy as np # matrix maths 
import pandas as pd # for data handeling 
import matplotlib.pyplot as plt # for plotting data 

# sklearn imports 
from sklearn.svm import SVR # svm regression

# import data set 
data = pd.read_csv('./Position_Salaries.csv')
features = data[['Level']] # features (X) 
output = data[['Salary']] # output i.e. y

# model fitting for regressor
regressor = SVR(kernel='poly')
regressor.fit(features,output)

# making a prediction score 
regressor.score(features,output)

# making a data viz 
plt.scatter(features,output, color='red')
plt.plot(features, regressor.predict(features))
plt.title('Position vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()