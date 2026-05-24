## Exercise 2 (10 minutes): K-Means Clustering

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    (containing numeric, imputed, and scaled features).
#    For demonstration, let's simulate df_scaled with the Iris dataset's features.
from sklearn.datasets import load_iris
import numpy as np

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate K-Means with a chosen number of clusters, say 3
# KMeans este un algoritm de clustering.
# El grupează datele în clustere pe baza asemănării dintre puncte.
# n_clusters=3 înseamnă că vrem 3 grupuri.
kmeans = KMeans(n_clusters=3, random_state=42)

# 3. Fit the model to the data
# Modelul caută automat centrele celor 3 clustere.
# Aici NU avem y, pentru că este învățare nesupervizată.
kmeans.fit(df_scaled)

# 4. Extract cluster labels
# labels_ conține clusterul atribuit fiecărui rând.
# De exemplu: 0, 1 sau 2.
cluster_labels = kmeans.labels_

# 5. (Optional) Add the cluster labels to the DataFrame
# Adăugăm rezultatul clusteringului ca o coloană nouă.
df_scaled["Cluster"] = cluster_labels

# 6. Print or visualize the results
# Afișăm primele rânduri ca să vedem clusterul atribuit fiecărei flori.
print(df_scaled.head())

# Afișăm câte exemple sunt în fiecare cluster.
print("\nNumber of points in each cluster:")
print(df_scaled["Cluster"].value_counts())

# 7. Optional quick visualization (for 2D only)
#    If you'd like a scatter plot, choose two features to plot.
# Alegem două coloane pentru a putea face grafic 2D.
plt.scatter(
    df_scaled["sepal length (cm)"],
    df_scaled["sepal width (cm)"],
    c=df_scaled["Cluster"]
)

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("K-Means Clustering")
plt.show()