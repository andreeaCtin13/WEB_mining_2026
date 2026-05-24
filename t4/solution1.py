## Exercise 1 (10 minutes): Load & Preprocess Your Dataset

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load the Iris dataset from scikit-learn
# Încărcăm datasetul Iris și îl transformăm într-un DataFrame.
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Introduce some artificial missing values (optional, for demonstration)
#    Here, we'll set a few entries to NaN in the 'petal length (cm)' column
# Simulăm valori lipsă pentru a vedea cum pot fi tratate.
df.iloc[5:10, 2] = np.nan

# 3. Handle missing values
#    We'll use SimpleImputer to replace NaNs with the mean of each column
# SimpleImputer înlocuiește valorile lipsă (NaN).
# strategy="mean" => folosește media coloanei respective.
imputer = SimpleImputer(strategy="mean")

# fit_transform:
# - calculează media fiecărei coloane
# - înlocuiește valorile lipsă cu acea medie
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# 4. Scale the data
#    StandardScaler transforms each feature to have mean=0 and std=1
# StandardScaler normalizează valorile.
# După transformare:
# media ≈ 0
# deviația standard ≈ 1
scaler = StandardScaler()

# fit_transform:
# - învață media și deviația standard
# - transformă datele
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_imputed),
    columns=df.columns
)

# 5. Check the results
# Verificăm dacă mai există valori lipsă.
print("Missing values after imputation:")
print(df_imputed.isnull().sum())

print("\nScaled dataset shape:")
print(df_scaled.shape)

# 6. (Optional) Print the first few rows to confirm preprocessing
print("\nFirst 5 rows after preprocessing:")
print(df_scaled.head())