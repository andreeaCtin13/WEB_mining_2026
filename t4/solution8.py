## Exercise 8 (10 minutes): Visual Summary & Report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# 1. Load or assume you have a preprocessed dataset
#    Here, we load & scale the Iris dataset for demonstration
iris = load_iris()
X = iris.data
y = iris.target  # Not used for clustering, but sometimes nice for reference

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply three clustering algorithms
# Aplicăm trei metode diferite de clustering pe aceleași date scalate.
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)
# de ce linkage=ward?
# Unește cele două clustere care cresc cel mai puțin variația internă (variance).
# Încearcă să formeze grupuri cât mai compacte și cât mai omogene.

# 3. Reduce to 2D with PCA for visualization
# PCA reduce datele de la mai multe dimensiuni la 2 dimensiuni,
# ca să le putem reprezenta într-un grafic 2D.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 4. Create a combined DataFrame for plotting & reporting
# Punem coordonatele PCA și etichetele clusterelor într-un DataFrame.
df_final = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_final['KMeans'] = kmeans.labels_
df_final['DBSCAN'] = dbscan.labels_
df_final['Agglo'] = agg.labels_

# 5. Plot side-by-side scatter plots of the clustering results in PCA space
# Facem 3 grafice, câte unul pentru fiecare algoritm.
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(
    df_final["PC1"],
    df_final["PC2"],
    c=df_final["KMeans"]
)
axes[0].set_title("KMeans Clustering")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

axes[1].scatter(
    df_final["PC1"],
    df_final["PC2"],
    c=df_final["DBSCAN"]
)
axes[1].set_title("DBSCAN Clustering")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

axes[2].scatter(
    df_final["PC1"],
    df_final["PC2"],
    c=df_final["Agglo"]
)
axes[2].set_title("Agglomerative Clustering")
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")

plt.tight_layout()
plt.show()

# 6. Print a short cluster distribution report
# Afișăm câte puncte a pus fiecare algoritm în fiecare cluster.
print("=== Cluster Distribution Report ===")

print("\nKMeans:")
print(df_final["KMeans"].value_counts())

print("\nDBSCAN:")
print(df_final["DBSCAN"].value_counts())
print("Note: In DBSCAN, label -1 means outlier/noise.")

print("\nAgglomerative:")
print(df_final["Agglo"].value_counts())