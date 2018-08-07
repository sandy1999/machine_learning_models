# base import 
import numpy as np # for martix maths 
import pandas as pd # for data handeling 
import matplotlib.pyplot as plt 

# sklearn imports 
from sklearn.tree import DecisionTreeRegressor

# loading data set 
data = pd.read_csv('./Position_Salaries.csv')
# feature extraction 
X = data[['Level']]
# output extraction 
y = data.Salary

# making a training model 
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y) # fitting model

# making a plot
X_grid = np.arange(1,10,0.01)
plt.scatter(X,y, color='red')
plt.plot(X_grid,regressor.predict(X_grid[:,np.newaxis]))
plt.title('Truth or Lie (Decision Tree)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show();

# testing score of descision tree 
print("Score of DT regressor is ",regressor.score(X,y))