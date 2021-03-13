import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy


def plot(labels, dataset, k):
    x = numpy.arange(10)
    ys = [i + x + (i * x) ** 2 for i in range(k)]

    colors = cm.rainbow(numpy.linspace(0, 1, len(ys)))
    LABEL_COLOR_MAP = {}
    for i in range(k):
        LABEL_COLOR_MAP[i] = colors[i]
    ds = numpy.array(dataset)
    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    plt.scatter(ds[:,0], ds[:,1], c=label_color)
    plt.show()



