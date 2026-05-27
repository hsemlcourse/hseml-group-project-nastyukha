import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Bike Sharing Demand Prediction API",
    description="API для прогнозирования почасового спроса на аренду велосипедов",
    version="1.0.0"
)

MODEL_PATH = os.path.join("models", "best_model.pkl")
DATA_PATH = os.path.join("data", "processed", "final_dataset.csv")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f" Успешно загружена финальная модель из {MODEL_PATH}")
else:
    model = None
    print(f" ВНИМАНИЕ: Файл модели не найден по пути {MODEL_PATH}")

if os.path.exists(DATA_PATH):
    df_history = pd.read_csv(DATA_PATH)
    print(" Исторические данные для лагов успешно загружены.")
else:
    df_history = None
    print(" ВНИМАНИЕ: final_dataset.csv не найден. Будет использоваться среднее значение.")


class PredictionInput(BaseModel):
    temp: float = Field(..., description="Температура воздуха (°C)", example=22.5)
    precip: float = Field(..., description="Количество осадков (мм)", example=0.0)
    wind: float = Field(..., description="Скорость ветра (км/ч)", example=12.3)
    hour: int = Field(..., ge=0, le=23, description="Час суток (0-23)", example=18)
    day_of_week: int = Field(..., ge=0, le=6, description="День недели (0-6, где 0 - Понедельник)", example=2)
    month: int = Field(..., ge=1, le=12, description="Месяц (1-12)", example=6)


@app.get("/health", summary="Проверка работоспособности сервиса")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена на сервере")
    return {"status": "healthy", "model": "XGBoost (best_model.pkl)"}


@app.post("/predict", summary="Получить прогноз спроса")
def predict_demand(payload: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель недоступна на сервере")

    lag_value = 100.0

    if df_history is not None:
        matched_rows = df_history[
            (df_history['hour'] == payload.hour) &
            (df_history['day_of_week'] == payload.day_of_week) &
            (df_history['month'] == payload.month)
            ]
        if not matched_rows.empty:
            lag_value = float(matched_rows['target'].mean())

    try:
        # Превращаем входные данные в DataFrame с правильным порядком колонок
        input_data = pd.DataFrame([{
            "temp": payload.temp,
            "precip": payload.precip,
            "wind": payload.wind,
            "hour": payload.hour,
            "day_of_week": payload.day_of_week,
            "month": payload.month,
            "cnt_lag_24h": lag_value
        }])

        prediction = model.predict(input_data)[0]

        # Спрос не может быть отрицательным, поэтому страхуемся
        final_prediction = max(0, float(prediction))

        return {
            "prediction": round(final_prediction, 1)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при инференсе модели: {str(e)}")