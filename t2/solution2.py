import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load the Wine dataset
# Încărcăm setul de date Wine.
# X conține caracteristicile (features), iar y conține clasele vinurilor.
wine = load_wine()
X = wine.data # features
y = wine.target # clasele

# 2. Split the data into training (80%) and testing (20%) sets
# Împărțim datele:
# - 80% pentru antrenarea modelului
# - 20% pentru testarea modelului
# random_state=42 asigură aceeași împărțire la fiecare rulare.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Train a Naïve Bayes classifier (from Exercise 1)
# Creăm modelul Gaussian Naive Bayes și îl antrenăm folosind datele de training.
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 4. Train a Logistic Regression classifier
# Creăm și antrenăm un model Logistic Regression pentru comparație.
# max_iter=1000 oferă suficiente iterații pentru convergență.
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

# 5. Predict on the test set
# Folosim ambele modele pentru a prezice clasele datelor de test.
y_pred_nb = nb_model.predict(X_test)
y_pred_logreg = logreg_model.predict(X_test)

# 6. Compare metrics: accuracy, precision, and recall for each model
# Accuracy = procentul predicțiilor corecte.
# Precision = cât de precise sunt predicțiile făcute.
# Recall = cât de bine identifică modelul exemplele reale.
#
# Wine este un dataset cu 3 clase, deci folosim average="macro"
# pentru a calcula media scorurilor tuturor claselor.
metrics = {
    # Creez o structură (dicționar) în care voi salva rezultatele fiecărui model.

    "Naive Bayes": {
        # Cheia este numele modelului pentru care calculez metricile.

        "Accuracy": accuracy_score(y_test, y_pred_nb),
        # Compar valorile reale (y_test) cu predicțiile modelului (y_pred_nb)
        # și obțin procentul de clasificări corecte.

        "Precision": precision_score(y_test, y_pred_nb, average="macro"),
        # Calculez precizia modelului.
        # Mă interesează cât de corecte sunt predicțiile făcute pentru fiecare clasă.
        # average="macro" face media scorurilor tuturor claselor.

        "Recall": recall_score(y_test, y_pred_nb, average="macro")
        # Calculez recall-ul modelului.
        # Verific cât de bine identifică exemplele reale din fiecare clasă.
        # average="macro" calculează media pentru toate clasele.
    },

    "Logistic Regression": {
        # Repet exact aceiași pași pentru al doilea model.

        "Accuracy": accuracy_score(y_test, y_pred_logreg),
        # Calculez acuratețea Logistic Regression.

        "Precision": precision_score(y_test, y_pred_logreg, average="macro"),
        # Calculez precizia Logistic Regression.

        "Recall": recall_score(y_test, y_pred_logreg, average="macro")
        # Calculez recall-ul Logistic Regression.
    }
}

# 7. Print results
# Afișăm valorile obținute pentru fiecare model.
for model_name, scores in metrics.items():
    print(f"=== {model_name} ===")
    print(f"Accuracy:  {scores['Accuracy']:.2f}")
    print(f"Precision: {scores['Precision']:.2f}")
    print(f"Recall:    {scores['Recall']:.2f}")
    print()