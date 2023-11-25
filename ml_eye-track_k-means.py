# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:57:00 2023

@author: DELL
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/DELL/Downloads/eye_tracking-heatmap.csv")

selected_rows = data.iloc[127:181] 

features = selected_rows[['fixation_point_x', 'fixation_point_y']]
num_clusters = 7

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)
cluster_labels = kmeans.labels_

centroids = kmeans.cluster_centers_
print("Centroids of the clusters:")
print(centroids)

plt.scatter(selected_rows['fixation_point_x'], selected_rows['fixation_point_y'], c=cluster_labels, cmap='viridis')
plt.title('KMeans Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# cluster_0_data = data[data['Cluster'] == 0]
# print("Data points in Cluster 0:")
# print(cluster_0_data.head())