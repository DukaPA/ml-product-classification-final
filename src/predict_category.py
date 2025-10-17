
import joblib
import os
import pandas as pd
import sys

model_path = os.path.join(os.path.dirname(__file__), '../model/product_category_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '../model/tfidf_vectorizer.pkl')

# Učitavanje model i TF-IDF vektorizator
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict(title):
    # Pretvoranje u DataFrame
    X = pd.Series([title]).astype(str)
    X_vec = vectorizer.transform(X)  # koristi isti TF-IDF kao u treniranju
    prediction = model.predict(X_vec)[0]
    return prediction

if __name__ == "__main__":
    if len(sys.argv) > 1:
        title = " ".join(sys.argv[1:])
        print(predict(title))
    else:
        while True:
            t = input("Enter product title (or exit): ")
            if t.lower() in ("exit", "quit"):
                break
            print("Predicted category:", predict(t))