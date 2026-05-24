## Exercise 6 (10 minutes): Customer Segmentation Use Case

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Generate synthetic customer data
#    For example:
#    - 'purchase_frequency': how many purchases per month
#    - 'average_spent': average amount spent per purchase
#    - 'loyalty_score': a simple 1–5 rating

np.random.seed(42)
num_customers = 50

df_customers = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 15, num_customers),
    'average_spent': np.random.randint(10, 500, num_customers),
    'loyalty_score': np.random.randint(1, 6, num_customers)
})

print("=== Raw Customer Data (first 5 rows) ===")
print(df_customers.head(), "\n")

# 2. Scale the data
# Scalăm datele pentru ca toate coloanele să aibă importanță comparabilă.
# average_spent are valori mari, iar loyalty_score are valori 1-5.
# Fără scalare, average_spent ar domina clusteringul.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_customers)

# 3. K-Means clustering
# Alegem 3 segmente de clienți.
# KMeans va grupa clienții pe baza asemănării dintre comportamente.
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# 4. Add cluster labels to the DataFrame
# Fiecare client primește un label de cluster: 0, 1 sau 2.
df_customers["cluster"] = kmeans.labels_

# 5. Inspect each segment
# Calculăm media valorilor pentru fiecare cluster.
# Asta ne ajută să înțelegem ce tip de clienți sunt în fiecare segment.
segment_summary = df_customers.groupby("cluster").mean()

print("=== Customer Segments Summary ===")
print(segment_summary)

print("\n=== Number of customers in each segment ===")
print(df_customers["cluster"].value_counts())

# 6. (Optional) Quick interpretation or marketing actions
#    For example, cluster 0 may represent "frequent, high-spending customers" etc.
# Interpretarea se face uitându-ne la mediile fiecărui cluster:
# - purchase_frequency mare + average_spent mare = clienți valoroși
# - purchase_frequency mic + average_spent mic = clienți mai puțin activi
# - loyalty_score mare = clienți loiali