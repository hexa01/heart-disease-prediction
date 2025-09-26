from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import pandas as pd

# Load trained model and scaler
with open("checkpoints/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("checkpoints/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

# Mount static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: float = Form(...),
    exang: int = Form(...),
    oldpeak: str = Form(...),  # dropdown returns string, will convert
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):
    # Convert oldpeak dropdown value to float
    try:
        oldpeak = float(oldpeak)
    except ValueError:
        oldpeak = 0.0  # fallback to default

    # Feature names must match what scaler/model were trained on
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal']

    # Convert input to DataFrame
    features_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]], columns=columns)

    # Scale features
    features_scaled = scaler.transform(features_df)

    # Predict
    prediction = model.predict(features_scaled)[0]

    result = "Heart Disease" if prediction == 1 else "No Heart Disease"

    return templates.TemplateResponse("index.html", {"request": request, "result": result})
