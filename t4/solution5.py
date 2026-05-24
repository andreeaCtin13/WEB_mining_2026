## Exercise 5 (10 minutes): Evaluating Clusters with Silhouette Scores

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load or assume you have a preprocessed dataset (df_scaled)
#    For demonstration, we'll again load & scale the Iris dataset
iris = load_iris()
X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit each clustering method

# K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42
).fit(X_scaled)

# DBSCAN
dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
).fit(X_scaled)

# Agglomerative Clustering
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
).fit(X_scaled)

# 3. Get the cluster labels from each method

# Fiecare algoritm atribuie fiecărui punct un cluster.
kmeans_labels = kmeans.labels_

dbscan_labels = dbscan.labels_

agg_labels = agg.labels_

# 4. Compute silhouette scores (only if more than one cluster exists)
#    DBSCAN might produce a single cluster or no clusters if parameters are not well-tuned,
#    so we check to avoid an error in silhouette_score.

# Silhouette Score măsoară cât de bine sunt separate clusterele.
# Valori apropiate de:
# 1  -> foarte bine separate
# 0  -> clustere suprapuse
# -1 -> punctele sunt probabil în clusterul greșit

# K-Means
kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

# Agglomerative
agg_score = silhouette_score(
    X_scaled,
    agg_labels
)

# Pentru DBSCAN verificăm întâi dacă există cel puțin 2 clustere.
# Altfel silhouette_score ar genera eroare.
if len(set(dbscan_labels)) > 1:
    dbscan_score = silhouette_score(
        X_scaled,
        dbscan_labels
    )
else:
    dbscan_score = "Cannot compute silhouette score"

# 5. Print the scores

print("K-Means Silhouette Score:")
print(kmeans_score)

print("\nAgglomerative Silhouette Score:")
print(agg_score)

print("\nDBSCAN Silhouette Score:")
print(dbscan_score)