## Exercise 7 (10 minutes): Anomaly Detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Generate synthetic "normal" data
#    E.g., two features representing normal operating ranges (e.g., purchase amounts, usage rates, etc.)
np.random.seed(42)

# Generăm 200 de puncte normale, în jurul valorii 50.
# scale=10 înseamnă că valorile variază în jurul mediei.
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))

# 2. Generate synthetic "anomalous" data
#    Points that deviate significantly from the normal distribution
# Aceste puncte sunt departe de zona normală, deci le considerăm anomalii.
outliers = np.array([
    [100, 100],
    [10, 90],
    [90, 10],
    [120, 40],
    [40, 120]
])

# 3. Combine the datasets
# Combinăm datele normale cu anomaliile într-un singur dataset.
X = np.vstack((normal_data, outliers))

# 4. Apply DBSCAN
#    eps controls the neighborhood radius; min_samples is how many samples must be within eps to form a cluster
# DBSCAN caută zone dense de puncte.
# Punctele care nu aparțin unei zone dense primesc eticheta -1.
dbscan = DBSCAN(eps=8, min_samples=5)
dbscan.fit(X)

# 5. Identify outliers (DBSCAN labels them as -1)
# labels_ conține clusterul fiecărui punct.
# -1 înseamnă anomalie/outlier.
labels = dbscan.labels_

# Selectăm punctele marcate ca outliers.
detected_outliers = X[labels == -1]

# 6. Visualization
# Vizualizăm toate punctele.
# Culoarea este dată de eticheta DBSCAN.
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

# Evidențiem outlierii detectați cu un marker diferit.
plt.scatter(
    detected_outliers[:, 0],
    detected_outliers[:, 1],
    marker="x",
    s=100
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Anomaly Detection with DBSCAN")
plt.show()

# 7. Reporting
# Afișăm câte anomalii a detectat DBSCAN.
print("Number of detected outliers:", len(detected_outliers))

print("\nDetected outliers:")
print(detected_outliers)