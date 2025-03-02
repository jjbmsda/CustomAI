from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil
import os
from model_handler import train_model, predict

app = FastAPI()

UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Custom AI Model Service is Running"}

@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File uploaded successfully", "file_path": file_path}

@app.post("/train-model/")
def train():
    model_path = train_model()
    return {"message": "Model trained successfully", "model_path": model_path}

@app.post("/predict/")
def make_prediction(features: list):
    prediction = predict(features)
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
