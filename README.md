# ml-product-classification-final

# Automatska klasifikacija proizvoda

Ovaj projekat razvija inteligentan sistem za automatsko predlaganje kategorije proizvoda na osnovu njegovog naziva, koristeći mašinsko učenje. Cilj je ubrzati i olakšati unos proizvoda na online platformi, smanjiti greške i poboljšati korisničko iskustvo.

## Struktura projekta
- `data/products.csv` — skup podataka za treniranje
- `notebooks/products_analysis_categories.ipynb` — Jupyter sveska sa analizom i razvojem modela
- `src/train_model.py` — skripta za treniranje i čuvanje modela
- `src/predict_category.py` — skripta za interaktivno testiranje modela
- `model/product_category_model.pkl` — sačuvani trenirani model

## Pokretanje projekta
1. Pokreni Jupyter svesku za analizu i razvoj modela.
2. Pokreni `src/train_model.py` za treniranje i čuvanje modela.
3. Pokreni `src/predict_category.py` za interaktivno testiranje modela.

## Testiranje
Unesi naziv proizvoda i model će predložiti odgovarajuću kategoriju. Primeri za testiranje:
- "iphone 7 32gb gold" → Mobile Phones
- "olympus e m10 mark iii geh use silber" → Digital Cameras

## Za tim
Projekat je dokumentovan i spreman za dalji razvoj ili korištenje u timu. 