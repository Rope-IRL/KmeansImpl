import numpy as np
import matplotlib.pyplot as pyl



class KMeansClustering:
    def __init__(self, k=3, acc = 0.00001) -> None:
        self.k = k
        self.centroids = None
        self.acc = acc
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids-data_point)**2, axis = 1))

    def fit(self, X, max_iterations = 200):
        self.centroids = np.random.uniform(np.amin(X, axis = 0), np.amax(X, axis = 0), size = (self.k, X.shape[1]))
        
        for _ in range(max_iterations):
            y = []

            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            y = np.array(y)


            cluster_indecies = []
            
            print(y)
            
            for i in range(self.k):
                cluster_indecies.append(np.argwhere(y == i))

            cluster_centres = []

            for i, indecies in enumerate(cluster_indecies):
                if len(indecies == 0):
                    cluster_centres.append(self.centroids[i])

                else:
                    cluster_centres.append(np.mean(X[indecies], axis = 0)[0])


            if np.max(self.centroids - np.array(cluster_centres)) < self.acc:
                break
            else:
                self.centroids = np.array(cluster_centres)

        return y
