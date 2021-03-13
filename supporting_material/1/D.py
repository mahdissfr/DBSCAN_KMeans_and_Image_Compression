from B import get_error
from FileHandler import read_from_file
from Kmeans import kmeans
import matplotlib.pyplot as plt
from Plot import plot

if __name__ == '__main__':
    dataSet = read_from_file("Dataset1.csv")
    min_iteration = 15
    errors = []
    ks = []
    for k in range(2,15):
        centroids, labels = kmeans(dataSet, k, min_iteration)
        errors.append(sum(get_error(dataSet, centroids, labels)) / len(get_error(dataSet, centroids, labels)))
        ks.append(k)

    plt.plot(ks, errors)
    plt.show()