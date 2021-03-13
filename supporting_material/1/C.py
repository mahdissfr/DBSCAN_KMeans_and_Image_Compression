from B import get_error
from FileHandler import read_from_file
from Kmeans import kmeans
from Plot import plot

if __name__ == '__main__':
    dataSet = read_from_file("Dataset1.csv")
    min_iteration = 15
    k = 4
    centroids, labels = kmeans(dataSet, k, min_iteration)
    # plot(labels, dataSet, k)
    errors = get_error(dataSet, centroids, labels)
    cluster_error = sum(errors) / len(errors)
    print("for k=" + str(k) + " cluster error : " + str(cluster_error))
