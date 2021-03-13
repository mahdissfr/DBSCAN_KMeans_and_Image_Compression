import random
from scipy.spatial.distance import cdist
import numpy
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image

import io
from google.colab import files

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

k = 16
min_iteration = 4

uploaded = files.upload()

img = image.imread(io.BytesIO(uploaded['imageSmall.png']))
plt.title("before")
plt.imshow(img)
plt.show()

dim = numpy.shape(img)
dataSet = [] #3rd dimention
for i in range(dim[0]):
    for j in range(dim[1]):
        dataSet.append(img[i, j].tolist())

centroids, labels = kmeans(dataSet, k, min_iteration)

