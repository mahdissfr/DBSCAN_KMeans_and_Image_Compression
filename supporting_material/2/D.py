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
labels = clustering.labels_.tolist()
x = np.array(labels)
unique_labels = np.unique(x)
k=len(unique_labels)
colors = []
for i in range(k):
  random_number = random.randint(0,16777215)
  hex_number = str(hex(random_number))
  hex_number ='#'+ hex_number[2:]
  colors.append(hex_number)
LABEL_COLOR_MAP = {}
for i in range(len(unique_labels)):
  if unique_labels[i] == -1:
    LABEL_COLOR_MAP[unique_labels[i]] = "#000000"
  LABEL_COLOR_MAP[unique_labels[i]] = colors[i]
label_color = [LABEL_COLOR_MAP[l] for l in labels]
m=folium.Map(location=[32.427910,53.688046],zoom_start=5)
for i in range(len(dataset)):
  loc = [dataset[i,0],dataset[i,1]]
  folium.Circle(location=loc,radius=1,color=LABEL_COLOR_MAP[labels[i]],fill=True).add_to(m)
m