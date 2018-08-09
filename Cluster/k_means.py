# base import 
import numpy as np # matrix maths 
import pandas as pd # data handeling 

# viz import 
import matplotlib.pyplot as plt # python plotting 

# sklearn imports 
from sklearn.cluster import KMeans # model class 

# importing data set 
data = pd.read_csv('./Mall_Customers.csv')
# extracting features 
X = data.iloc[:,3:].values

# viz elbow for the data set 
wcss = []
for i in range(1,11):
    cluster = KMeans(n_clusters=i, random_state=0)
    cluster.fit(X)
    wcss.append(cluster.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# making a prediction model 
cluster = KMeans(n_clusters=5, random_state=0)
y_means = cluster.fit_predict(X)

# viz our model prediction 
plt.scatter(X[y_means == 0,0], X[y_means == 0,1], s=100, color='red', label='Cluster 1')
plt.scatter(X[y_means == 1,0], X[y_means == 1,1], s=100, color='blue', label='Cluster 2')
plt.scatter(X[y_means == 2,0], X[y_means == 2,1], s=100, color='green', label='Cluster 3')
plt.scatter(X[y_means == 3,0], X[y_means == 3,1], s=100, color='cyan', label='Cluster 4')
plt.scatter(X[y_means == 4,0], X[y_means == 4,1], s=100, color='magenta', label='Cluster 5')
plt.scatter(cluster.cluster_centers_[:,0], cluster.cluster_centers_[:,1] ,s=300, color='yellow', label='Centroid')
plt.show()