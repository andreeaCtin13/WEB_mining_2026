import numpy as np                      # Bibliotecă pentru lucrul cu vectori și matrici.
from sklearn.metrics import accuracy_score  # Funcție care calculează acuratețea modelului.


# Definim clasa Perceptron (modelul nostru).
class Perceptron:

    # Constructorul clasei.
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr                    # Learning rate = cât de mult modificăm greutățile când greșim.
        self.epochs = epochs            # De câte ori trecem prin toate datele.

    # Funcția de antrenare.
    def fit(self, X, y):

        # Adaugă o coloană de 1 la final pentru bias.
        # Dacă X era [[0,1],[1,0]]
        # devine [[0,1,1],[1,0,1]]
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Inițializăm greutățile cu valori aleatoare.
        # Numărul de greutăți = numărul de coloane din X.
        self.weights = np.random.randn(X.shape[1])

        # Repetăm antrenarea de "epochs" ori.
        for _ in range(self.epochs):

            # Luăm fiecare exemplu și răspunsul lui corect.
            for xi, yi in zip(X, y):

                # Modelul face o predicție.
                pred = self.predict_single(xi)

                # Calculăm eroarea.
                # exemplu:
                # răspuns real = 1
                # predicție = 0
                # eroare = 1
                error = yi - pred

                # Actualizăm greutățile.
                # Dacă eroarea este 0 nu se schimbă nimic.
                self.weights += self.lr * error * xi

    # Predicție pentru un singur exemplu.
    def predict_single(self, x):

        # np.dot = produs scalar.
        # face:
        # x1*w1 + x2*w2 + ...
        #
        # dacă rezultatul >= 0 => clasa 1
        # altfel => clasa 0
        return 1 if np.dot(x, self.weights) >= 0 else 0

    # Predicție pentru mai multe exemple.
    def predict(self, X):

        # Adăugăm iar coloana de bias.
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Facem predict pentru fiecare rând.
        return np.array([self.predict_single(xi) for xi in X])


# --------------------------------------------------
# 1. AND gate dataset (linearly separable)
# --------------------------------------------------

# Toate combinațiile posibile de 0 și 1.
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Rezultatele operației AND.
#
# 0 AND 0 = 0
# 0 AND 1 = 0
# 1 AND 0 = 0
# 1 AND 1 = 1
y_and = np.array([0, 0, 0, 1])

# Creăm modelul.
perceptron_and = Perceptron(
    lr=0.1,
    epochs=10
)

# Modelul învață pe date.
perceptron_and.fit(
    X_and,
    y_and
)

# Modelul face predicții.
pred_and = perceptron_and.predict(
    X_and
)

# Comparăm răspunsurile reale cu predicțiile.
print(
    "AND gate accuracy:",
    accuracy_score(y_and, pred_and)
)


# --------------------------------------------------
# 2. XOR gate dataset (non-linearly separable)
# --------------------------------------------------

# Aceleași intrări.
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Rezultatele operației XOR.
#
# 0 XOR 0 = 0
# 0 XOR 1 = 1
# 1 XOR 0 = 1
# 1 XOR 1 = 0
y_xor = np.array([0, 1, 1, 0])

# Creăm un nou Perceptron.
perceptron_xor = Perceptron(
    lr=0.1,
    epochs=10
)

# Îl antrenăm pe datele XOR.
perceptron_xor.fit(
    X_xor,
    y_xor
)

# Facem predicții.
pred_xor = perceptron_xor.predict(
    X_xor
)

# Calculăm acuratețea.
print(
    "XOR gate accuracy:",
    accuracy_score(y_xor, pred_xor)
)