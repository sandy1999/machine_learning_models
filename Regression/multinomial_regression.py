"""
A script to find out factors affecting profit in a start up 
"""

# base imports

import numpy as np # for matrix maths 
import pandas as pd # for data handelling 

# sklearns import
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load data set 
data = pd.read_csv('./50_Startups.csv')
# features
features = data[['R&D Spend','Administration','Marketing Spend','State']]
# output
output = data.Profit

# handlle categorial data 
features =  pd.get_dummies(features)
features =  features.iloc[:,:-1]
features.head()

# splitting our data set
X_train,X_test,y_train,y_test = train_test_split(features,output,test_size=1/3,random_state=0)

# to train our data set 
regressor = LinearRegression() # LR object 
regressor.fit(X_train,y_train) # fitting dataset 

# score of our model
print(regressor.score(X_test,y_test))