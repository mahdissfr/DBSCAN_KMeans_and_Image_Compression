import folium
import pandas as pd
import io
from google.colab import files
uploaded = files.upload()
df2 = pd.read_csv(io.BytesIO(uploaded['covid.csv']), header=None)
m=folium.Map(location=[32.427910,53.688046],zoom_start=5)
loc = [float(df2.iloc[0][0]),float(df2.iloc[0][1])]
for i in range(len(df2)):
  loc = [df2.iloc[i][0],df2.iloc[i][1]]
  folium.Circle(location=loc,radius=1,color='red',fill=True).add_to(m)
m

