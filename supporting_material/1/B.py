import numpy

from FileHandler import read_from_file
from Kmeans import kmeans
from Plot import plot
from scipy.spatial.distance import cdist


def get_error(dataSet, centroids, labels):
    errors = [0 for i in range(len(centroids))]
    for i in range(len(labels)):
        errors[labels[i]] += numpy.math.sqrt(
            (dataSet[i][0] - centroids[labels[i]][0]) ** 2 + (dataSet[i][1] - centroids[labels[i]][1]) ** 2)/labels.count(labels[i])
    return errors


if __name__ == '__main__':
    dataSet = read_from_file("Dataset1.csv")
    min_iteration = 15
    k = 4
    centroids, labels = kmeans(dataSet, k, min_iteration)
    plot(labels, dataSet, k)
    print("for k=" + str(k) + " the average distance between the cluster center and the data points in that cluster : " + str(get_error(dataSet, centroids, labels)))
