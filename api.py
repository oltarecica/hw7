from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import numpy as np
import json

model = joblib.load("model.pkl")

app = FastAPI(title="Iris Model API")

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "API is running"}

#The existing endpoint
@app.post("/predict")
def predict(data: Features):
    X_new = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(X_new)[0]
    return {"prediction": int(prediction)}

#The new endpoint
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    contents = await file.read()
    data_list = json.loads(contents)

    if isinstance(data_list, dict):
        data_list = [data_list]

    X_new = np.array([
        [d["sepal_length"], d["sepal_width"], d["petal_length"], d["petal_width"]]
        for d in data_list
    ])

    predictions = model.predict(X_new).tolist()
    return {"predictions": predictions}
