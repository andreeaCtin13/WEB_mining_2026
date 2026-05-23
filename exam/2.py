# Exercițiul 2: Creați un model de clasificare text folosind un subset din 20 Newsgroups,
# cu două clase: 'sci.space' și 'rec.sport.hockey'. Vectorizați textul, antrenați modelul și
# calculați acuratețea pe setul de test.


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === START ===
print("1. Incep download dataset...")

categories = ['sci.space', 'rec.sport.hockey']

data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

print("2. Dataset descarcat")
print(data)

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42
)

print("3. Split facut")

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("4. Vectorizare gata")

model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

print("5. Model antrenat")

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
# === END ===
