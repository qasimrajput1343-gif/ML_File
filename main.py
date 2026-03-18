from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")
label_encoder = joblib.load("label.pkl")

@app.get("/")
def home():
    return {"message": "Model is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    
    df = pd.DataFrame([data])
    pred = model.predict(df)
    result = label_encoder.inverse_transform(pred)
    
    return {"prediction": result[0]}