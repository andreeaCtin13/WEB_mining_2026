# Exercițiul 1: Accesați pagina http://quotes.toscrape.com și extrageți toate citatele,
# autorii și tag-urile de pe prima pagină. Salvați rezultatul într-un DataFrame pandas
# cu coloanele: 'quote', 'author', 'tags' (tags este o listă de șiruri).

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://quotes.toscrape.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# === Your code starts here ===

quotes = soup.find_all("div", class_="quote")

data = []

for quote in quotes:
    quote_text = quote.find("span", class_="text").text
    author = quote.find("small", class_="author").text
    tags = quote.find("div", class_="tags")
    tags_text = ""
    for tag in tags.find_all("a"):
        tags_text = tags_text + tag.text.strip() + ","

    data.append({
        "quote_text": quote_text,
        "author": author,
        "tags": tags_text
    })

df = pd.DataFrame(data)

print(df)

# === Your code ends here ===
