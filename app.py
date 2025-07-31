from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI()

model = tf.keras.models.load_model("energy_model.h5")
scaler = joblib.load('scaler.save')


class Features(BaseModel):
    data : list

@app.post('/predict')
def predict(features : Features):
    try:
        x = np.array(features.data).reshape(1,-1)
        x_Scaled = scaler.transform(x)
        pred = model.predict(x_Scaled)
        return {"Prediction": float(pred[0][0])}
    except Exception as e:
        raise HTTPException(status_code = 400 , detail = str(e))