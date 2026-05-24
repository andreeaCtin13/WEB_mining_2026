### **Exercise 2: Web Scraping a Product Listings Page**
import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops"

response = requests.get(URL)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

products = []

for item in soup.select(".thumbnail"):
    title = item.select_one(".title").get_text(strip=True)
    price = item.select_one(".price").get_text(strip=True)
    description = item.select_one(".description").get_text(strip=True)

    rating_tag = item.select_one("p[data-rating]")
    rating = rating_tag["data-rating"] if rating_tag else None

    reviews = item.select_one(".ratings .pull-right").get_text(strip=True)

    products.append({
        "title": title,
        "price": price,
        "description": description,
        "rating": rating,
        "reviews": reviews
    })

df = pd.DataFrame(products)

df_cleaned = df.copy()
df_cleaned["price"] = df_cleaned["price"].str.replace("$", "", regex=False).astype(float)
df_cleaned["reviews"] = df_cleaned["reviews"].str.extract(r"(\d+)").astype(int)
df_cleaned["rating"] = df_cleaned["rating"].astype(int)

df_cleaned.to_csv("scraped_products.csv", index=False)

print(df_cleaned.head())
