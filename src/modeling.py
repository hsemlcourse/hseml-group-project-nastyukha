import pandas as pd
import joblib
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprocessing import prepare_full_dataset, get_splits

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def run_modeling():
    input_dir = os.path.join('data', 'raw')

    df = prepare_full_dataset(input_dir)
    X_train, X_test, y_train, y_test = get_splits(df)

    models = {
        "Baseline (Ridge)": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }

    results = []
    best_model = None
    best_mae = float('inf')
    best_name = ""

    print("\n--- Сравнение моделей ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({"Model": name, "MAE": mae, "R2": r2})
        print(f" Модель {name} обучена. MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    res_df = pd.DataFrame(results)

    try:
        print("\n", res_df.to_markdown(index=False))
    except ImportError:
        print("\n", res_df.to_string(index=False))

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\n Победитель: {best_name} с MAE: {best_mae:.2f}. Сохранено!")


if __name__ == "__main__":
    run_modeling()