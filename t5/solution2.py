# Exercise 2 (15 minutes): Building an MLP Using PyTorch or TensorFlow

import torch                         # Biblioteca principală PyTorch.
import torch.nn as nn                # Modul pentru rețele neuronale.
import torch.optim as optim          # Modul pentru optimizatori.
import matplotlib.pyplot as plt      # Pentru graficul pierderii/loss-ului.

# 1. Create dataset: 2D points, label = 1 if y > x else 0
torch.manual_seed(42)                # Fixăm random-ul ca să avem aceleași rezultate la fiecare rulare.

X = torch.rand(200, 2)               # Generăm 200 de puncte, fiecare cu 2 valori: x și y.

y = (X[:, 1] > X[:, 0]).float().unsqueeze(1)
# X[:, 1] = a doua coloană, adică valoarea y a punctului.
# X[:, 0] = prima coloană, adică valoarea x a punctului.
# X[:, 1] > X[:, 0] verifică dacă y > x.
# Dacă y > x, label-ul este 1.
# Dacă y <= x, label-ul este 0.
# .float() transformă True/False în 1.0/0.0.
# .unsqueeze(1) face forma din (200,) în (200, 1), ca să se potrivească cu output-ul modelului.

# 2. Define a simple MLP model

class MLP(nn.Module):
    # Definim o rețea neuronală simplă.

    def __init__(self):
        super().__init__()
        # Apelăm constructorul clasei părinte nn.Module.

        self.model = nn.Sequential(
            nn.Linear(2, 8),
            # Primul layer:
            # primește 2 inputuri: x și y
            # produce 8 valori interne.

            nn.ReLU(),
            # Funcție de activare.
            # Ajută modelul să învețe relații mai complexe.

            nn.Linear(8, 1),
            # Ultimul layer:
            # primește 8 valori
            # produce 1 singur output.

            nn.Sigmoid()
            # Transformă output-ul într-o valoare între 0 și 1.
            # Aproape de 1 = clasa 1
            # Aproape de 0 = clasa 0
        )

    def forward(self, x):
        # Definește cum trec datele prin model.
        return self.model(x)


model = MLP()
# Creăm modelul.

# 3. Loss and optimizer

criterion = nn.BCELoss()
# Binary Cross Entropy Loss.
# Se folosește pentru clasificare binară: 0 sau 1.

optimizer = optim.Adam(model.parameters(), lr=0.01)
# Adam modifică greutățile modelului ca loss-ul să scadă.
# lr=0.01 este learning rate-ul.

# 4. Training loop

losses = []
# Salvăm loss-ul la fiecare epocă, ca să îl putem desena la final.

epochs = 100
# Modelul va trece de 100 de ori prin date.

for epoch in range(epochs):

    y_pred = model(X)
    # Modelul face predicții pentru toate punctele.

    loss = criterion(y_pred, y)
    # Comparăm predicțiile cu răspunsurile reale.
    # Rezultatul este loss-ul.

    optimizer.zero_grad()
    # Resetăm gradientele vechi.
    # Altfel, PyTorch le-ar aduna de la o epocă la alta.

    loss.backward()
    # Calculează cât trebuie schimbate greutățile ca loss-ul să scadă.

    optimizer.step()
    # Aplică efectiv modificarea greutăților.

    losses.append(loss.item())
    # Salvăm valoarea numerică a loss-ului.

# 5. Evaluation

with torch.no_grad():
    # Nu mai antrenăm modelul, doar evaluăm.
    # Deci nu avem nevoie de gradient.

    predictions = model(X)
    # Obținem predicții între 0 și 1.

    predicted_classes = (predictions >= 0.5).float()
    # Dacă predicția este >= 0.5, clasa devine 1.
    # Dacă este sub 0.5, clasa devine 0.

    accuracy = (predicted_classes == y).float().mean()
    # Comparăm clasele prezise cu y real.
    # mean() calculează procentul de predicții corecte.

print("Accuracy:", accuracy.item())

# 6. Plot loss over time

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True)
plt.show()