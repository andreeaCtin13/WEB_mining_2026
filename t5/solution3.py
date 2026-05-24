# Exercise 3 (10 minutes): Exploring Activation Functions on the Iris Dataset

import torch                                # PyTorch pentru rețele neuronale.
import torch.nn as nn                       # Module pentru layers, activări, loss.
import torch.optim as optim                 # Optimizatori, ex: Adam.
from sklearn.datasets import load_iris       # Datasetul Iris.
from sklearn.preprocessing import StandardScaler  # Pentru scalarea datelor.
from sklearn.model_selection import train_test_split  # Pentru împărțire train/test.
import matplotlib.pyplot as plt             # Pentru grafic.

# 1. Load and prepare the Iris dataset (binary classification)

iris = load_iris()                           # Încărcăm datasetul Iris.
X = iris['data']                             # X = features: cele 4 măsurători ale florilor.
y = (iris['target'] == 0).astype(float)      # y = 1 dacă floarea este Setosa, altfel 0.

# Train/test split and normalization

X_train, X_test, y_train, y_test = train_test_split(
    X,                                       # Datele de intrare.
    y,                                       # Etichetele/răspunsurile corecte.
    test_size=0.3,                           # 30% test, 70% train.
    random_state=42                          # Aceeași împărțire la fiecare rulare.
)

scaler = StandardScaler()                    # Creez scalerul.

X_train = scaler.fit_transform(X_train)      # Învăț scalarea pe train și o aplic.
X_test = scaler.transform(X_test)            # Aplic aceeași scalare pe test.

X_train = torch.tensor(X_train, dtype=torch.float32)
# Transform X_train din numpy array în tensor PyTorch.
# Rețeaua neuronală PyTorch lucrează cu tensori, nu direct cu pandas/numpy.

y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
# Transform y_train în tensor.
# reshape(-1, 1) îl face coloană, adică formă compatibilă cu output-ul modelului.

# 2. Build model factory with pluggable activation

def build_model(activation_fn):
    # Funcția primește o funcție de activare.
    # Asta ne permite să testăm ReLU, Sigmoid, Tanh etc. cu aceeași arhitectură.

    return nn.Sequential(
        nn.Linear(4, 8),                     # Layer 1: primește 4 features și produce 8 valori.
        activation_fn,                       # Funcția de activare pe care vrem să o testăm.
        nn.Linear(8, 1),                     # Layer final: din 8 valori produce 1 output.
        nn.Sigmoid()                         # Output între 0 și 1, potrivit pentru clasificare binară.
    )

# 3. Training function

def train(model):
    # Funcția primește un model și îl antrenează.

    criterion = nn.BCELoss()
    # Binary Cross Entropy Loss.
    # Se folosește când avem clasificare binară: 0 sau 1.

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Adam modifică greutățile modelului ca loss-ul să scadă.
    # model.parameters() = greutățile modelului.
    # lr=0.01 = learning rate.

    losses = []
    # Aici salvăm loss-ul de la fiecare epocă pentru grafic.

    for _ in range(50):
        # Antrenăm timp de 50 de epoci.

        optimizer.zero_grad()
        # Ștergem gradientele vechi.
        # În PyTorch, gradientele se acumulează dacă nu le resetăm.

        y_pred = model(X_train)
        # Modelul face predicții pe datele de training.

        loss = criterion(y_pred, y_train)
        # Calculăm cât de mult greșește modelul.

        loss.backward()
        # Calculăm gradientele.
        # Adică aflăm cum trebuie schimbate greutățile.

        optimizer.step()
        # Optimizerul actualizează greutățile modelului.

        losses.append(loss.item())
        # Salvăm valoarea loss-ului ca număr normal Python.

    return losses
    # Returnăm lista cu loss-uri pentru a o putea desena.

# 4. Try different activations

activations = {
    "ReLU": nn.ReLU(),                       # ReLU: transformă valorile negative în 0.
    "Tanh": nn.Tanh(),                       # Tanh: comprimă valorile între -1 și 1.
    "Sigmoid": nn.Sigmoid()                  # Sigmoid: comprimă valorile între 0 și 1.
}

results = {}
# Aici vom salva loss-urile pentru fiecare funcție de activare.

for name, activation_fn in activations.items():
    # Luăm pe rând fiecare activare.

    model = build_model(activation_fn)
    # Construim un model cu activarea curentă.

    losses = train(model)
    # Antrenăm modelul și obținem loss-urile.

    results[name] = losses
    # Salvăm loss-urile în dicționar, sub numele activării.

# 5. Plot loss curves

plt.figure(figsize=(8, 5))

for name, loss_curve in results.items():
    # Parcurgem rezultatele pentru fiecare activare.

    plt.plot(loss_curve, label=name)
    # Desenăm curba loss-ului pentru activarea respectivă.

plt.title("Activation Function Comparison (Iris - Binary Classification)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()