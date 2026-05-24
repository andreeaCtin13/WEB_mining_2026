import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a small synthetic dataset
#    For simplicity, let's assume we have only numeric features.
np.random.seed(42)  # Pentru rezultate identice la fiecare rulare

num_samples = 20

# Generăm 20 exemple cu 2 caracteristici numerice
X = np.random.rand(num_samples, 2) * 100

# True relationship (just as an example):
# price = 3.0*(feature1) + 2.0*(feature2) + some_noise
true_coeffs = np.array([3.0, 2.0])

# Construim prețul pe baza formulei de mai sus și adăugăm puțin zgomot
y = X.dot(true_coeffs) + np.random.normal(0, 10, size=num_samples)

# Convert to a pandas DataFrame for familiarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Price"] = y

# 2. Separate features (X) and target (y)
# X = datele de intrare (ce folosim pentru predicție)
# y = valoarea pe care vrem să o prezicem (Price)

X = df[["Feature1", "Feature2"]]
y = df["Price"]

# 3. Split the dataset into training and test sets
# Împărțim datele:
# - training = modelul învață
# - testing = verificăm cât de bine a învățat

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Create and train the Linear Regression model
# Creăm modelul de regresie liniară.
# Acesta încearcă să găsească formula:
# y = a*Feature1 + b*Feature2 + c

model = LinearRegression()

# Modelul învață coeficienții folosind datele de training.
model.fit(X_train, y_train)

# 5. Use the model to predict on the test set
# Pe baza valorilor din X_test,
# modelul estimează prețurile (y_pred).

y_pred = model.predict(X_test)

# 6. Evaluate the model
# Comparăm valorile reale (y_test)
# cu valorile estimate de model (y_pred).

# R² arată cât de bine explică modelul variația datelor.
# Mai aproape de 1 = mai bine.
r2 = r2_score(y_test, y_pred)

# MSE = media pătratelor erorilor.
# Mai mic = mai bine.
mse = mean_squared_error(y_test, y_pred)

# MAE = media valorilor absolute ale erorilor.
# Mai mic = mai bine.
mae = mean_absolute_error(y_test, y_pred)

print("Coefficients:", model.coef_)
# Coeficienții învățați pentru fiecare caracteristică.

print("Intercept:", model.intercept_)
# Constanta din ecuația regresiei.

print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")