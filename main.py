from KMeans import KMeansClustering
import numpy as np
import matplotlib.pyplot as plt



def main(k, acc):
    kmeans = KMeansClustering(k, acc)
    random_points = np.random.randint(0, 255, (100, 2))
    #print(random_points)
    labels = kmeans.fit(random_points)
    print(labels)
    print(kmeans.centroids)
    plt.scatter(random_points[:, 0], random_points[:, 1], c = labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c = range(len(kmeans.centroids)), marker = "*", s = 200)

    plt.show()
    


if __name__ == "__main__":
    main(5, 0.0001)
