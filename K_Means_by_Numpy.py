''' K Means Algorithm using Numpy '''

import numpy as np
from scipy.spatial import distance
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


'''

Assign k means paramters
K, max_iter, tol , max iter -> for loop

Assign random K points as centroid - initialization

calculate disatance and label each data point

calculate mean of data points for respective clusters -> gives new centroid -> for loop

check old centroid = new centroid -> if yes -> stop

old centroid = new centroid

'''


class KMeans:
  
  def __init__(self,n_clusters,max_iter,tol):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.centroids = None

  def fit(self, data):
    # centroid initialization
    self.centroids = data[np.random.choice(len(data),size=self.n_clusters,replace=False)]
    
    # cluster assigning to each data point
    for _ in range(self.max_iter):
      clusters = self._assign_clusters(data)

      # create new centroids
      new_centroids = self._update_centroids(data,clusters)

      if np.allclose(new_centroids,self.centroids,atol=self.tol):
        break

      self.centroids = new_centroids

  def _assign_clusters(self,data):
    # calculate distance of clusters and data points
    distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

  def _update_centroids(self,data,clusters):
    # create new centroids & update new centroids
    new_centroids = np.zeros_like(self.centroids)
    
    for idx in range(self.n_clusters):
      new_centroids[idx] = data[clusters == idx].mean(axis=0)

    return new_centroids


if __name__ == "__main__":
  # create random data points

  X, _ = make_blobs(n_samples = 1000
           , centers = 4,cluster_std=0.6,random_state=42)
  
  # Initialize the k parameters

  k= 4; max_iter = 600; tol = 1e-4

  kmeans = KMeans(k,max_iter,tol)
  kmeans.fit(X)

  # Get the centroids
  centroids = kmeans.centroids
  print("Centroids:")
  print(centroids)

  # Plot the data points and centroids
  plt.scatter(X[:, 0], X[:, 1], c=kmeans._assign_clusters(X), cmap='viridis', alpha=0.5)
  plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
  plt.title('K-means Clustering')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.show()


