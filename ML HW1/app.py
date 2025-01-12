from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from joblib import load
import pandas as pd
from typing import List, Optional

app = FastAPI()

# Класс для одного объекта
class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float

# Класс для ответа с предсказанием
class PredictionResponse(BaseModel):
    prediction: float

# Загрузка модели
model = load('model.joblib')

# Функция для преобразования модели pydantic в DataFrame
def pydantic_model_to_df(model_instance):
    return pd.DataFrame([jsonable_encoder(model_instance)])


@app.post("/predict_item/", response_model=PredictionResponse)
def predict_item(item: Item) -> PredictionResponse:
    # Преобразуем входной объект в DataFrame
    df_instance = pydantic_model_to_df(item)
    
    # Предсказание с использованием модели
    prediction = model.predict(df_instance).tolist()[0]

    # Возвращаем только предсказание
    return PredictionResponse(prediction=prediction)
