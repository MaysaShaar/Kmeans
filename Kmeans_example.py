import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from KmeansClass import Kmeans


k = 3
load_data = load_iris()
dataT = load_data["data"]
targets = load_data["target"]
data = np.empty((len(dataT), 2))
index = 0
for point in dataT:
    x, y, z, w = point
    data[index] = [x, y]
    index += 1

kmeans = Kmeans(data, targets, k)
centroids, clusters = kmeans.fit()

print(centroids)
x, y = zip(*clusters[str(centroids[0])])
plt.scatter(x, y,alpha= 0.75, c='red', label ='Cluster 1')
x, y = zip(*clusters[str(centroids[1])])
plt.scatter(x, y,alpha= 0.75, c='blue', label ='Cluster 2')
x, y = zip(*clusters[str(centroids[2])])
plt.scatter(x, y,alpha= 0.75, c='green', label ='Cluster 3')


plt.scatter([row[0] for row in centroids], [row[1] for row in centroids],  s=1000, alpha= 0.55)

plt.show()

