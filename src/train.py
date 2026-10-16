"""Train and compare models, then save the best one and its predictions.

Run from the project root:  python -m src.train
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save charts to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from src.data_prep import add_test_rul, add_train_rul, find_useless_columns, load_raw
from src.features import add_rolling_features, get_feature_columns

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
FIGURES = Path("reports/figures")
MODELS = Path("models")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    for folder in (PROCESSED, FIGURES, MODELS):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Load and label
    train = add_train_rul(load_raw(RAW / "train_FD001.txt"))
    test = add_test_rul(load_raw(RAW / "test_FD001.txt"), RAW / "RUL_FD001.txt")

    # 2. Drop sensors that never change, then engineer features
    useless = find_useless_columns(train)
    train, test = train.drop(columns=useless), test.drop(columns=useless)
    sensors = [c for c in train.columns if c.startswith("s")]
    train = add_rolling_features(train, sensors)
    test = add_rolling_features(test, sensors)
    features = get_feature_columns(train)
    print(f"Dropped {len(useless)} constant columns; using {len(features)} features")

    # 3. Validation split BY ENGINE, so no engine appears in both sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(train, groups=train["unit"]))
    X_tr, y_tr = train.iloc[train_idx][features], train.iloc[train_idx]["RUL"]
    X_val, y_val = train.iloc[val_idx][features], train.iloc[val_idx]["RUL"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42),
    }

    # 4. Train each model and score it on the validation engines
    results = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        results.append({"model": name, "val_RMSE": rmse(y_val, pred), "val_MAE": mean_absolute_error(y_val, pred)})
    results = pd.DataFrame(results).sort_values("val_RMSE")
    print("\nValidation results:\n", results.round(2).to_string(index=False))

    # 5. Retrain the best model on ALL training data, score on the official test set
    best_name = results.iloc[0]["model"]
    best = models[best_name].fit(train[features], train["RUL"])
    last = test.groupby("unit").tail(1).copy()  # official scoring uses each engine's last cycle
    last["predicted_RUL"] = best.predict(last[features]).clip(min=0).round(1)
    print(f"\nBest model: {best_name}")
    print(f"Test RMSE: {rmse(last['RUL'], last['predicted_RUL']):.2f} cycles")
    print(f"Test MAE:  {mean_absolute_error(last['RUL'], last['predicted_RUL']):.2f} cycles")

    # 6. Save the model, predictions for the dashboard, and a chart
    joblib.dump(best, MODELS / "best_model.joblib")
    last["status"] = pd.cut(last["predicted_RUL"], bins=[-1, 30, 75, np.inf],
                            labels=["Critical", "Warning", "Healthy"])
    last[["unit", "cycle", "RUL", "predicted_RUL", "status"]].to_csv(PROCESSED / "test_predictions.csv", index=False)
    results.to_csv(PROCESSED / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(last["RUL"], last["predicted_RUL"], alpha=0.7)
    ax.plot([0, 125], [0, 125], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{best_name}: predicted vs actual RUL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "predicted_vs_actual.png", dpi=150)
    print("\nSaved model, predictions, and chart.")


if __name__ == "__main__":
    main()