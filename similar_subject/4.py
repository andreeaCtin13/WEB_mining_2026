# Exercițiul 4: Antrenați o rețea neuronală simplă (cu un strat ascuns) în PyTorch
# pentru clasificarea vinurilor folosind setul de date load_wine din sklearn.
# Afișați acuratețea pe setul de test după 50 de epoci.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X = wine.data
y = wine.target

# === Your code starts here ===

print(X.shape)
print(y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(13, 16)
        self.layer2 = nn.Linear(16, 178)

        # nn.Linear(in_features, out_features)
        # Astfel, o regulă generală este:
        # self.layer1 = nn.Linear(X.shape, 16 - neuroni alesi de mine)
        # self.layer2 = nn.Linear(nr de neuroni alesi de mine, nr clase - aici 10)

        # dimensiunea primului Linear = nr features(X)
        # dimensiunea ultimului Linear = numărul clase(Y)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = DigitModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()

print("Accuracy:", accuracy.item())


# === Your code ends here ===