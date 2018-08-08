"""
A ML model built on the top of Decision Tree classifier in order to predict wether a person will buy or not a product from Social Network Ads 
"""

# base import 
import numpy as np # matrix maths 
import pandas as pd # data handeling

# viz imports 
import matplotlib.pyplot as plt # for plotting 
from matplotlib.colors import ListedColormap # colour map 
import seaborn as sns; sns.set() # for better plots 

# sklearn import 
from sklearn.preprocessing import StandardScaler # make data standard 
from sklearn.metrics import confusion_matrix # performance metrics 
from sklearn.model_selection import train_test_split  # dataset spltting 
from sklearn.tree import DecisionTreeClassifier # model class 

# Loading data set 
data = pd.read_csv('./Social_Network_Ads.csv')
# extract data features 
X = data.iloc[:,[2,3]].values
# extract output class 
y = data.iloc[:,-1].values

# makind dataset standard 
X = StandardScaler().fit_transform(X)

# spliting dataset 
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.25)

# creating a model 
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# print score of model 
print("Score of Decision tree classifier is {}".format(classifier.score(X_test,y_test)))

# viz confusion matrix 
mat = confusion_matrix(y_test,classifier.predict(X_test))
sns.heatmap(mat.T, square=True, annot=True, cbar=False, fmt='d')
plt.show()

# visualizing a train case
X_set,y_set = X_train,y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()