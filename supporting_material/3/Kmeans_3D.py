import random
from scipy.spatial.distance import cdist
import numpy


def InitializeCentroids(dataSet, k):
    # return [[numpy.random.random_sample(),numpy.random.random_sample(),numpy.random.random_sample()] for i in random.sample(range(0, len(dataSet) - 1), k)]
    return [dataSet[i] for i in random.sample(range(0, len(dataSet) - 1), k)]


def kmeans(dataSet, k, iterations):
    centroids = InitializeCentroids(dataSet, k)
    xcentroids = None
    labels = None
    while not convergence(centroids, xcentroids) and iterations != 0:
        xcentroids = centroids
        labels = label(dataSet, centroids)
        centroids = updateCentroids(dataSet, labels, k)
        iterations -= 1

    return centroids, labels


def convergence(centroids, xcentroids):
    if xcentroids is None:
        return False
    return centroids == xcentroids


def label(dataSet, centroids):
    dist = cdist(dataSet, centroids)
    labels = [numpy.argmin(dist[i]) for i in range(len(dist))]
    return labels


def unique(list1):
    x = numpy.array(list1)
    return numpy.unique(x)


def updateCentroids(dataSet, labels, k):
    centroids=[]
    for i in range(k):
        points = [dataSet[j] for j in range(len(dataSet)) if labels[j] == i]
        if points:
            centroids.append(numpy.mean(points, axis=0).tolist())
    return centroids


def eucl_dist(a, b, axis=1):
    return numpy.linalg.norm(a - b, axis=axis)
