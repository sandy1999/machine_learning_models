# base import 
import numpy as np  # for matrix maths 
import pandas as pd # for data handeling 
import matplotlib.pyplot as plt # for plotting 

# sklearn imports
from sklearn.ensemble import RandomForestRegressor

# loading data set 
data = pd.read_csv('./Position_Salaries.csv')
# feature extraction 
X  = data[['Level']]
# output extraction 
y  = data.Salary 

# making a model for training 
model = RandomForestRegressor(n_estimators=300, random_state=0)
model.fit(X,y)

# making a data viz
X_grid = np.arange(1,10,0.01)
plt.scatter(X,y,color='red')
plt.plot(X_grid,model.predict(X_grid[:,np.newaxis]))
plt.title('Truth or Lie (Random Forest Regressor)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# score of model
print("Score of Random Forest Model is ",model.score(X,y))