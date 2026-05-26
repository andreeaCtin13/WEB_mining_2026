# Exercițiul 3: Folosiți setul de date despre prețuri imobiliare din California pentru a
# antrena un model de regresie Ridge. Tratați valorile lipsă, folosiți GridSearchCV
# pentru a căuta parametrul optim alpha și afișați eroarea MSE pe setul de test.

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# === START ===
df = df.dropna()

X = df.drop("median_house_value", axis=1)
# axis=1 înseamnă coloană (axis=0 ar însemna rând).

y = df["median_house_value"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Ridge()

param_grid = {
    "alpha": [0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Best alpha:", grid_search.best_params_)
print("MSE:", mse)
# === END ===
