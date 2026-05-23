# Exercițiul 1: Extrageți titlul, prețul și disponibilitatea cărților de pe pagina principală
# a site-ului http://books.toscrape.com. Normalizați prețurile (float) și disponibilitatea
# (1 pentru "In stock", 0 altfel), apoi stocați-le într-un DataFrame pandas.

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://books.toscrape.com/"

response = requests.get(url)
response.encoding = "utf-8"

# === START ===
soup = BeautifulSoup(response.text, "html.parser")

books = soup.find_all("article", class_="product_pod")

data = []

for book in books:
    print("book:",book)
    title = book.h3.a["title"]

    price_text = book.find("p", class_="price_color").text
    price = float(price_text.replace("£", ""))

    availability_text = book.find("p", class_="instock availability").text.strip()
    availability = 1 if "In stock" in availability_text else 0

    data.append({
        "title": title,
        "price": price,
        "availability": availability
    })

df = pd.DataFrame(data)

print(df)
# === END ===
