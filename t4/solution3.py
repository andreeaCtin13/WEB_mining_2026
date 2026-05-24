## Exercise 3 (10 minutes): DBSCAN Clustering

import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    For demonstration, we'll again simulate df_scaled with the Iris dataset's features.
from sklearn.datasets import load_iris

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate DBSCAN with chosen parameters
#    eps defines the neighborhood radius, min_samples is the minimum number of points
#    for a region to be considered dense.
# eps = cât de aproape trebuie să fie punctele ca să fie considerate vecine.
# min_samples = câte puncte trebuie să existe într-o zonă ca să formeze un cluster.
dbscan = DBSCAN(eps=0.8, min_samples=5)
# DBSCAN E UTILIZAT PT A ELIMINA OUTLIERS

# 3. Fit the model to the data
# DBSCAN caută zone dense de puncte.
# Nu folosim y, pentru că este clustering nesupervizat.
dbscan.fit(df_scaled)

# 4. Extract cluster labels
# labels_ conține clusterul fiecărui punct.
# -1 înseamnă outlier/zgomot.
cluster_labels = dbscan.labels_

# 5. Identify outliers (DBSCAN labels outliers as -1)
# Selectăm punctele care au label -1.
outliers = df_scaled[cluster_labels == -1]

# 6. (Optional) Add the labels to the DataFrame
# Adăugăm clusterul într-o coloană nouă.
df_scaled["Cluster"] = cluster_labels

# 7. Print the cluster label counts
# Vedem câte puncte sunt în fiecare cluster.
# Dacă apare -1, acelea sunt outliers.
print("Cluster label counts:")
print(df_scaled["Cluster"].value_counts())

print("\nNumber of outliers:")
print(len(outliers))

# 8. Optional quick visualization (for 2D only)
#    Choose two features to plot, coloring by DBSCAN labels
plt.scatter(
    df_scaled["sepal length (cm)"],
    df_scaled["sepal width (cm)"],
    c=df_scaled["Cluster"]
)

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("DBSCAN Clustering")
plt.show()