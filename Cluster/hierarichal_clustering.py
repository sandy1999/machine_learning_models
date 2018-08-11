"""
A Machine Learning Hierarichal Cluster model template to make clusters of people spending in mall  
"""

# base import 
import numpy as np # matrix maths 
import pandas as pd # data handeling 

# data viz 
import matplotlib.pyplot as plt # python plotting 
import scipy.cluster.hierarchy as sch # to viz dendogram 

# sklearn imports 
from sklearn.cluster import AgglomerativeClustering

# loading data set 
data = pd.read_csv('./Mall_Customers.csv')
# extracting features 
X = data.iloc[:,3:].values

# plotting dendogram 
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendograms')
plt.xlabel('Customers')
plt.ylabel('Eculidian Distance')
plt.show()  # showing dendogram 

# making a hac model 
hac = AgglomerativeClustering(n_clusters=5,linkage='ward')
hac.fit(X)

# making a viz for our distance 
labels = hac.fit_predict(X) # predicting the labels 
plt.scatter(X[:,0],X[:,1],s=10, c=labels, cmap='viridis')
plt.title('Hierarchial Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()