"""
A ML model built on the top of Logistic Regression in order to predict wether a person will buy or not a product from Social Network Ads 
"""

# base imports 
import numpy as np # matrix maths 
import pandas as pd # for data processing 
import matplotlib.pyplot as plt # for plotting curves 
import seaborn as sns  #for beatutiful plots 

# sklearn imports 
from sklearn.model_selection import train_test_split # for data spliting 
from sklearn.linear_model import LogisticRegression # model class 
from sklearn.preprocessing import StandardScaler # to make features scaled to standard values 
from sklearn.preprocessing import LabelEncoder # import Label encoder
from sklearn.pipeline import make_pipeline # making a pipeline 
from sklearn.metrics import confusion_matrix # to analyze our model performance

# import data set 
data = pd.read_csv('./Social_Network_Ads.csv')
# label encoding
data.Gender = LabelEncoder().fit_transform(data.Gender)
# feature extraction 
X = data.iloc[:,2:3].values
# output determination 
y = data.Purchased.values 

# spliting data set 
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.25)

# making a model for logistic regression 
regressor = make_pipeline(StandardScaler(),LogisticRegression(random_state=0))
regressor.fit(X_train,y_train)
pred = regressor.predict(X_test)

# score of our model
print(regressor.score(X_test,y_test))

# making heat map of confusion matrix 
mat = confusion_matrix(y_test,pred)
sns.heatmap(mat.T, square=True, annot= True, fmt='d', cbar=False)
plt.xlabel('True Labels')
plt.ylabel('Predicted labels')
plt.show()