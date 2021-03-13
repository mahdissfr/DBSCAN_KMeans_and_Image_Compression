import random
import os
from scipy.spatial.distance import cdist
import numpy
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
import tensorflow as tf
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
new_img16 = numpy.array(img, copy=True)
cntr = 0
for i in range(dim[0]):
    for j in range(dim[1]):
        new_img16[i, j] = centroids[labels[cntr]]
        cntr+=1

k = 256
min_iteration = 16
centroids, labels = kmeans(dataSet, k, min_iteration)
new_img256 = numpy.array(img, copy=True)
cntr = 0
for i in range(dim[0]):
    for j in range(dim[1]):
        new_img256[i, j] = centroids[labels[cntr]]
        cntr+=1

new_img16=numpy.array(new_img16)
rescaled = (255.0 / new_img16.max() * (new_img16 - new_img16.min())).astype(numpy.uint8)
im = Image.fromarray(rescaled)
saver=tf.compat.v1.train.Saver(im)
session = tf.compat.v1.Session()
save_path = saver.save(session, "data/dm.ckpt")
print('done saving at',save_path)
files.download( "data/dm.ckpt.meta" )

new_img256=numpy.array(new_img256)
rescaled = (255.0 / new_img256.max() * (new_img256 - new_img256.min())).astype(numpy.uint8)
im = Image.fromarray(rescaled)
saver=tf.compat.v1.train.Saver(im)
session = tf.compat.v1.Session()
save_path = saver.save(session, "data/dm.ckpt")
print('done saving at',save_path)
files.download( "data/dm.ckpt.meta" )

plt.subplot(132)
plt.imshow(new_img16)
plt.title('k=16')

plt.subplot(133)
plt.imshow(new_img256)
plt.title('k=256')
# plt.title("after")
# plt.imshow(new_img)
plt.show()
