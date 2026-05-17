import pandas as pd
import joblib
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprocessing import prepare_full_dataset, get_splits

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def run_modeling():
    input_dir = os.path.join('data', 'raw')

    df = prepare_full_dataset(input_dir)
    X_train, X_test, y_train, y_test = get_splits(df)

    models = {
        "Baseline (Ridge)": make_pipeline(StandardScaler(), Ridge()),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    }

    results = []
    best_mae = 200
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

    print("\nТаблица экспериментов:")
    print("\n", res_df.to_markdown(index=False))

    print("\n--- Перебор гиперпараметров для LightGBM ---")
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 10, -1]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    lgb = LGBMRegressor(random_state=42, verbose=-1)
    grid_search = GridSearchCV(
        estimator=lgb,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_tuned_model = grid_search.best_estimator_

    tuned_preds = best_tuned_model.predict(X_test)
    tuned_mae = mean_absolute_error(y_test, tuned_preds)
    tuned_r2 = r2_score(y_test, tuned_preds)

    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Метрики модели на тесте - MAE: {tuned_mae:.2f}, R2: {tuned_r2:.3f}")

    if tuned_mae < best_mae:
        best_mae = tuned_mae
        best_model = best_tuned_model
        best_name = "LightGBM (boosting)"

    print(f"\n Найдена лучшая модель: {best_name} с MAE: {best_mae:.2f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\n Победитель: {best_name} с MAE: {best_mae:.2f}. Сохранено!")


if __name__ == "__main__":
    run_modeling()