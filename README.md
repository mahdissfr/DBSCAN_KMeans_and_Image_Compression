Implementation 1: KMeans
In this part, I implemented and used the K-means clustering algorithm. First I clustered a simple 2D dataset, next I evaluated the performance of clustering and finally checked out the restrictions of KMeans by running it on a complex dataset.

Implementation 2: DBSCAN
I used DBSCAN library in sklearn package in this part. In this assignment, I used geographical distribution of COVID-19 patients inside Iran (covid.csv).
In this section I should:
	A) Load the dataset from csv file and plot it on map using folium. 
	B) Run DBSCAN algorithm with arbitrary values for eps and minPts.
	C) Fine-tune eps and minPts parameters so that each cluster only includes patients from heavily infected areas. these clusters must exclude outlier locations.
	D) Plot each cluster on map using a different color. Plot outliers with same color.

Implementation 3: Image Compression
In this part, I used K-means algorithm to reduce the number of colors to 16 (or 256) so that the color of each pixel can be represented by only 4 bits (or 8 bits). Since reducing the number of the colors result in lower quality for the image, we use K-means to find the 16 (or 256) colors that best group pixels in the image.