import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry


def fetch_historical_weather(lat, lon, start_date, end_date):
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"]
    }

    responses = openmeteo.weather_api(url, params=params)
    hourly = responses[0].Hourly()

    data = {
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temp": hourly.Variables(0).ValuesAsNumpy(),
        "precip": hourly.Variables(1).ValuesAsNumpy(),
        "wind": hourly.Variables(2).ValuesAsNumpy()
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Проверка парсера...")
    print(fetch_historical_weather(41.87, -87.62, "2024-05-01", "2024-05-02").head())