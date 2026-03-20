import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("health_data.csv")

X = df[["diarrhea", "fever", "vomiting", "rainfall", "temperature"]]
y = df["risk"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
