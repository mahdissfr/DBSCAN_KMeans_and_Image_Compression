import numpy

from FileHandler import read_from_file
from Kmeans import kmeans
from Plot import plot

if __name__ == '__main__':
    dataSet = read_from_file("Dataset1.csv")
    min_iteration = 15
    for k in range(2,5):
        centroids, labels = kmeans(dataSet, k, min_iteration)
        plot(labels,dataSet,k)