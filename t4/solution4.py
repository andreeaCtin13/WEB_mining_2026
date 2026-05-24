## Exercise 4 (10 minutes): Agglomerative Clustering & Dendrogram

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    For demonstration, we simulate df_scaled by loading and scaling the Iris dataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# ------------------------------------------------------------------------------------

# 2. Perform Agglomerative Clustering
# Agglomerative Clustering pornește cu fiecare punct separat
# și apoi unește treptat punctele/grupurile cele mai apropiate.
# n_clusters=3 înseamnă că vrem la final 3 grupuri.
agg = AgglomerativeClustering(n_clusters=3)

# fit_predict face două lucruri:
# - antrenează/găsește grupurile
# - returnează eticheta clusterului pentru fiecare rând
cluster_labels = agg.fit_predict(df_scaled)

# 3. Add the cluster labels to the DataFrame
# Adăugăm clusterul găsit ca o coloană nouă.
df_scaled["Cluster"] = cluster_labels

# 4. Print a quick summary of how many points were assigned to each cluster
# Vedem câte puncte au fost puse în fiecare cluster.
print("Cluster counts:")
print(df_scaled["Cluster"].value_counts())

# 5. Create a linkage matrix for plotting a dendrogram
#    Note: We exclude the 'cluster' column when computing the linkage
# Dendrograma arată vizual cum s-au unit punctele/grupurile.
# Excludem coloana Cluster deoarece ea este rezultatul, nu feature original.
linked = linkage(df_scaled.drop("Cluster", axis=1), method="ward")

# 6. Plot the dendrogram
# Desenăm dendrograma.
plt.figure(figsize=(10, 6))

dendrogram(linked)

plt.title("Dendrogram - Agglomerative Clustering")
plt.xlabel("Data points")
plt.ylabel("Distance")

plt.show()