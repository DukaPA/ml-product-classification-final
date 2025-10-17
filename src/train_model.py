import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Učitaj podatke
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/products.csv'))
df.columns = df.columns.str.strip()

# Osnovno čišćenje podataka
df = df.dropna(subset=['Product Title', 'Category Label'])

# Feature engineering: samo naziv proizvoda
X = df['Product Title']
y = df['Category Label']

# TF-IDF vektorizacija
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Podjela na trening i test
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# Treniraj model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluacija
y_pred = model.predict(X_test)
print('Tačnost:', accuracy_score(y_test, y_pred))
print('Klasifikacijski izveštaj:')
print(classification_report(y_test, y_pred))
print('Matrica zabune:')
print(confusion_matrix(y_test, y_pred))

# Sačuvaj model i vektorizator
joblib.dump(model, os.path.join(os.path.dirname(__file__), '../model/product_category_model.pkl'))
joblib.dump(vectorizer, os.path.join(os.path.dirname(__file__), '../model/tfidf_vectorizer.pkl'))
print('Model i vektorizator su sačuvani.')
