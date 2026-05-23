# Exercițiul 2: Creați un model de clasificare text pentru categoriile 'alt.atheism'
# și 'soc.religion.christian' din 20 Newsgroups. Folosiți CountVectorizer, antrenați
# un model Naive Bayes și afișați raportul de clasificare pe setul de test.

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("1. Incep download dataset...")

categories = ['alt.atheism', 'soc.religion.christian']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# === Your code starts here ===

print("2. Dataset descarcat")
print(data)

# PASI PENTRU ACEST EXERCITIU:
# TRAIN_TEST_SPLIT
# TE FOLOSESTI DE CE TI SE DA PT A FACE FIT_TRANSFORM + TRANSFORM
# => FIT_TRANSFORM => X_train_vectorized PE BAZA X_TRAIN
# => TRANSFORM => X_test_vectorized PE BAZA X_TEST
# INITIALIZEZI MODEL
# MODEL.FIT

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

report = classification_report(y_test, y_pred)

print("REPORT:", report)

# === Your code ends here ===
