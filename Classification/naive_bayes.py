"""
A ML model built on the top of Naive Bayes in order to predict wether a person will buy or not a product from Social Network Ads 
"""

# base import 
import numpy as np # matrix maths 
import pandas as pd # data handeling 

# viz imports 
import matplotlib.pyplot as plt # plotting 
from matplotlib.colors import ListedColormap # colour maps 
from seaborn import heatmap # for confusion matrix viz 

# sklearn imports 
from sklearn.preprocessing import StandardScaler # standardization of data set 
from sklearn.model_selection import train_test_split # split data set 
from sklearn.metrics import confusion_matrix # performance evaluation 
from sklearn.naive_bayes import GaussianNB # classifier class 

# load data set
data = pd.read_csv('./Social_Network_Ads.csv')
# feature extraction 
X = data.iloc[:,[2,3]].values 
# output extraction 
y = data.iloc[:,-1].values 

# standardization of features 
X = StandardScaler().fit_transform(X)

# spliting data set 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=0)

# making a classifier model 
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# score of classifier 
print("Score of classifier is {}".format(classifier.score(X_test,y_test)))

# viz confusion matrix 
mat = confusion_matrix(y_test,classifier.predict(X_test))
heatmap(mat.T, square=True, annot=True,cbar=False, fmt='d')
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
plt.title('SVM (Training set)')
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
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()