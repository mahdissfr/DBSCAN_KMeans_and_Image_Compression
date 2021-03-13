from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import folium
import io
from google.colab import files
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

uploaded = files.upload()
df2 = pd.read_csv(io.BytesIO(uploaded['covid.csv']), header=None)
dataset = df2.to_numpy()
clustering = DBSCAN(eps=0.1, min_samples=30).fit(dataset)
clustering.labels_