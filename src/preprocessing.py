import pandas as pd
import os
import glob
from data_parser import fetch_historical_weather


def prepare_full_dataset(data_raw_dir):
    print("Агрегация поездок")

    # Ищем все CSV файлы в папке
    all_files = glob.glob(os.path.join(data_raw_dir, "*.csv"))

    df_list = []
    for file in all_files:
        print(f"Загрузка файла: {os.path.basename(file)}...")
        df_temp = pd.read_csv(file, usecols=['started_at'])
        df_list.append(df_temp)

    # Объединяем все месяцы в один массив
    df_raw = pd.concat(df_list, ignore_index=True)
    print(f"Всего загружено сырых строк поездок: {len(df_raw)}")

    df_raw['timestamp'] = pd.to_datetime(df_raw['started_at'], format='mixed').dt.round('h')

    # Группируем поездки по часам
    df_bikes = df_raw.groupby('timestamp').size().reset_index(name='target')

    # Определяем границы для погоды
    start = df_bikes['timestamp'].min().strftime('%Y-%m-%d')
    end = df_bikes['timestamp'].max().strftime('%Y-%m-%d')

    print(f"Парсинг погоды ({start} - {end})")
    df_weather = fetch_historical_weather(41.87, -87.62, start, end)

    df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp']).dt.tz_localize(None)

    print("Мердж и фичи")
    df = pd.merge(df_bikes, df_weather, on='timestamp', how='inner')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['cnt_lag_24h'] = df['target'].shift(24)

    df = df.dropna()
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/final_dataset.csv', index=False)
    return df


def get_splits(df):
    df = df.sort_values('timestamp')
    X = df.drop(columns=['target', 'timestamp'])
    y = df['target']
    split = int(len(df) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]