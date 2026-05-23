# Exercițiul 3: Descărcați setul de date despre automobile de la URL-ul de mai jos.
# Folosiți regresie polinomială (gradul 2) pentru a prezice consumul de combustibil
# pe baza puterii motorului. Afișați R² pe setul de test.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

url = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
df = pd.read_csv(url)

# === Your code starts here ===
df = df.dropna()

X = df[["hp"]]      # puterea motorului
y = df["mpg"]       # consumul de combustibil

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

poly_pipeline = make_pipeline(PolynomialFeatures(degree=2),model)

poly_pipeline.fit(X_train, y_train)

y_pred = poly_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)

print(r2)
# === Your code ends here ===
