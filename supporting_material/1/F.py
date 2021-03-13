from FileHandler import read_from_file
from Kmeans import kmeans
from Plot import plot

if __name__ == '__main__':
    dataSet = read_from_file("Dataset2.csv")
    min_iteration = 15
    k = 4
    centroids, labels = kmeans(dataSet, k, min_iteration)
    plot(labels, dataSet, k)
