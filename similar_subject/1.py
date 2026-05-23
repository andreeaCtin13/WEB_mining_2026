# Exercițiul 1: Extrageți numele țărilor, capitalele și populația din tabelul de pe pagina
# https://www.scrapethissite.com/pages/simple/. Normalizați populația ca întreg și
# stocați rezultatele într-un DataFrame pandas.

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.scrapethissite.com/pages/simple/"
response = requests.get(url)
response.encoding = "utf-8"
soup = BeautifulSoup(response.text, 'html.parser')

# === START ===

countries = soup.find_all("div", class_="col-md-4 country")

data = []

for country in countries:
    country_name = country.find("h3", class_="country-name").text
    capital = country.find("span", class_="country-capital").text
    population = int(country.find("span", class_="country-population").text)
    area = country.find("span", class_="country-area").text

    data.append([country_name, capital, population, area])

df = pd.DataFrame(data)
print(df)

# === END ===