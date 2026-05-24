## Exercise 6 (10 minutes): kNN for Regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset
np.random.seed(42)
num_samples = 30

# Let's generate two features (e.g., Feature1, Feature2) and a target
X = np.random.rand(num_samples, 2) * 10

# Define a "true" relationship for the target: y = 3*X1 + 2*X2 + noise
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=num_samples)

# Convert to a DataFrame for clarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Target"] = y

# 2. Separate features and target
# X = coloanele pe baza cărora prezicem
# y = valoarea pe care vrem să o prezicem
X = df[["Feature1", "Feature2"]]
y = df["Target"]

# 3. Split into training and test sets
# Împărțim datele în training și testing.
# Modelul învață pe training și este testat pe date nevăzute.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Feature scaling (recommended for distance-based methods like kNN)
# kNN folosește distanțe între puncte.
# De aceea, scalarea este importantă: aduce feature-urile la o scară comparabilă.
scaler = StandardScaler()

# Învățăm media și deviația standard doar din training.
X_train_scaled = scaler.fit_transform(X_train)

# Aplicăm aceeași scalare pe test.
X_test_scaled = scaler.transform(X_test)

# 5. Create and train the kNN Regressor
#    We'll start with n_neighbors=3 (can try different values)
# n_neighbors=3 înseamnă că modelul se uită la cei mai apropiați 3 vecini.
model = KNeighborsRegressor(n_neighbors=3)

# Antrenăm modelul pe datele scalate.
model.fit(X_train_scaled, y_train)

# 6. Evaluate on the test set
# Facem predicții pe datele de test scalate.
y_pred = model.predict(X_test_scaled)

# Comparăm valorile reale cu predicțiile.
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")

# 7. (Optional) Explore the effect of different k values
#    You can loop over various values of k and compare performance.