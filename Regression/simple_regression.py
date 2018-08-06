"""
A script to perform simple linear regression on a data which includes a experience as features and salary as output  
"""

# base imports 
import numpy as mp # matrix maths 
import pandas as pd # for data handeling 
import matplotlib.pyplot as plt # for viz 

# sklearn imports 
from sklearn.model_selection import train_test_split # to split train test set
from sklearn.linear_model import LinearRegression # a simple linear regression 

# to extract data 
data = pd.read_csv('./Salary_Data.csv')
# features
features = data.iloc[:,:-1].values
# output
output = data.iloc[:,:1].values

# spliting data set 
X_train,X_test,y_train,y_test = train_test_split(features,output,test_size=1/3,random_state=0)

# making a regression model 
regressor = LinearRegression()
# fitting regression data 
regressor.fit(X_train,y_train)

# fitting our regression values 
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Experience v/s Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# printing accuracy of our data 
print("Accuracy is {}".format(regressor.score(X_test,y_test)))