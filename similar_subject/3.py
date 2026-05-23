# Exercițiul 3: Descărcați setul de date despre vinuri din sklearn. Folosiți regresie logistică
# pentru clasificarea vinurilor. Standardizați datele, împărțiți în train/test și afișați
# raportul de clasificare pe setul de test.

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

wine = load_wine()
X = wine.data
y = wine.target

# === Your code starts here ===

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

report = classification_report(y_test, y_pred)

print(report)
# === Your code ends here ===