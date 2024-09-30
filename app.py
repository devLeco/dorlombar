import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File


app = FastAPI(docs_url='/',title='Deploy DM 2023.3 BI Master')

# carregar modelo treinado
model = joblib.load(r'model\spine_model.pkl')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """Endpoint para inferência de dor lombar"""
    
    data = pd.read_csv(file.file)
    inferences = model.predict(data)
    return {'inferences': inferences.tolist()}