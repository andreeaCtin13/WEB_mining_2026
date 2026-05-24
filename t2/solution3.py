import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load the Wine dataset
# Încărcăm datasetul Wine.
# X = caracteristicile vinurilor
# y = clasele/etichetele vinurilor
wine = load_wine()
X = wine.data # caracteristici
y = wine.target # clasele

# 2. Split into training (80%) and testing (20%) sets
# Împărțim datele în:
# - training: datele pe care modelul învață
# - testing: datele pe care verificăm cât de bine a învățat
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Train a Decision Tree Classifier
#    max_depth=3 to control overfitting a bit
# Creăm modelul Decision Tree.
# max_depth=3 limitează cât de adânc poate crește arborele.
# Asta ajută ca modelul să nu memoreze prea mult datele de training.
# evit procesul de suprainvatare avand max_depth=3
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Antrenăm modelul pe datele de training.
model.fit(X_train, y_train)

# 4. Check accuracy on the test set
# Facem predicții pe datele de test.
y_pred = model.predict(X_test)

# Comparăm valorile reale cu predicțiile modelului.
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# (Optional) Visualize the tree structure
# Desenăm arborele pentru a vedea cum ia decizii modelul.
plt.figure(figsize=(15, 8))

plot_tree(
    model,
    feature_names=wine.feature_names, # numele caracteristilor
    class_names=wine.target_names, # numele claselor
    filled=True
)

plt.show()

# (Optional) Feature importances
# Vedem ce caracteristici au fost cele mai importante pentru deciziile modelului.
feature_importances = pd.DataFrame({
    "Feature": wine.feature_names, # asa accesezi caracteristicile
    "Importance": model.feature_importances_ # asa accesezi importanta lor
})

print(feature_importances)