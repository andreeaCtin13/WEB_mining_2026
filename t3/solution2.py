## Exercise 2 (10 minutes): Polynomial Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic non-linear dataset
np.random.seed(42)
num_samples = 30

# Single feature for clarity (e.g., 'sqft' or just X)
X = np.linspace(0, 10, num_samples).reshape(-1, 1)

# True relationship: y = 2 * X^2 - 3 * X + noise
y_true = 2 * (X**2) - 3 * X
noise = np.random.normal(0, 3, size=(num_samples, 1))
y = (y_true + noise).flatten()

# Convert to DataFrame
df = pd.DataFrame({"Feature": X.flatten(), "Target": y})

print(df.head())
# 2. Separate features and target

X = df["Feature"]
Y = df["Target"]

# 3. Split into training and test sets

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Transform features to polynomial (degree=2 or 3 for illustration)
# 5. Create and train a Linear Regression model on the polynomial features
model = LinearRegression()

poly_degree = 2
poly = PolynomialFeatures(degree=poly_degree)

x_train_poly = poly.fit_transform(x_train.values.reshape(-1, 1))
x_test_poly = poly.transform(x_test.values.reshape(-1, 1))

model.fit(x_train_poly, y_train)

# 6. Evaluate the model on the test set

y_pred = model.predict(x_test_poly)
poly_degree = 2
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Polynomial Degree:", poly_degree)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"R² on test set: {r2:.3f}")
print(f"MSE on test set: {mse:.3f}")
print(f"MAE on test set: {mae:.3f}")

# 7. Optional: Plot to visualize the fit
#    Generate a smooth curve for plotting
