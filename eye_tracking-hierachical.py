# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:09:15 2023

@author: DELL
"""

import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

data = pd.read_csv("C:/Users/DELL/Downloads/eye_tracking-heatmap.csv")
selected_rows = data.iloc[255:]

features = selected_rows[['fixation_point_x', 'fixation_point_y']]


# Perform hierarchical clustering using single linkage
# You can change 'single' to 'complete', 'average', etc., for different linkage methods
# Adjust the metric parameter (e.g., 'euclidean', 'cosine') based on your data
linkage_matrix = linkage(features, method='complete', metric='euclidean')

num_clusters = 7
cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

#data['Cluster'] = cluster_labels

plt.scatter(selected_rows['fixation_point_x'], selected_rows['fixation_point_y'], c=cluster_labels, cmap='viridis')
plt.title('Hierarchical Clustering (Complete Linkage, 7 clusters Euclidean Individual)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Calculate silhouette score (a higher silhouette score indicates better clustering)
silhouette_avg = silhouette_score(features, cluster_labels)
# Davies-Bouldin Index
db_index = davies_bouldin_score(features, cluster_labels)

# Calinski-Harabasz Index
ch_index = calinski_harabasz_score(features, cluster_labels)

print(f"Silhouette Score Complete: {silhouette_avg}")
print(f"davies bouldin Complete:{db_index}")
print(f"Calinski-Harabasz Complete: {ch_index}")