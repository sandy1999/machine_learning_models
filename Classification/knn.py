# base import 
import numpy as np  # for matrix maths
import pandas as pd # for data handeling 

# viz imports
import matplotlib.pyplot as plt # for plotting
import seaborn as sns; sns.set() # for beautiful plotting of data set 
from matplotlib.colors import ListedColormap # colour maps 

# sklearn imports 
from sklearn.model_selection import train_test_split # to split data set 
from sklearn.preprocessing import StandardScaler # to make standard features 
from sklearn.pipeline import make_pipeline  # to make pipeline 
from sklearn.neighbors import KNeighborsClassifier # model class 
from sklearn.preprocessing import LabelEncoder # to encode categorial data 
from sklearn.metrics import confusion_matrix # evaluation matrix 

# importing data set 
data = pd.read_csv('./Social_Network_Ads.csv')
# to encode data 
data.Gender = LabelEncoder().fit_transform(data.Gender)
# extracting features 
X = data.iloc[:,[2,3]].values
# output 
y = data.iloc[:,-1].values
# making a standard data set 
sca = StandardScaler()
X = sca.fit_transform(X)

# spliting data set 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=0)

# making a model class classifier
classifier = KNeighborsClassifier(n_neighbors=5) # model object 
classifier.fit(X_train,y_train) # model fitting 

# score of our model 
print("Score of our model is {}".format(classifier.score(X_test,y_test)))

# visualizing a confusion matrix 
ypred = classifier.predict(X_test)
mat = confusion_matrix(y_test,ypred)
sns.heatmap(mat.T, fmt='d',annot=True, cbar=False,square=True,cmap='plasma')
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
plt.title('K-NN (Training set)')
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
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()