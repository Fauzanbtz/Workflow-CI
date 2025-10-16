# modelling.py
# ====================================================
# Training Model untuk Prediksi Harga Rumah + Tracking MLflow
# ====================================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os


# ====================================================
# 1️⃣ Load & Persiapan Data
# ====================================================
def load_and_prepare_data(base_path: str):
    train_path = os.path.join(base_path, "train_preprocessed.csv")
    test_path = os.path.join(base_path, "test_preprocessed.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    target_column = "Price (in rupees)"

    # Hapus baris tanpa target
    train_data = train_data.dropna(subset=[target_column])
    test_data = test_data.dropna(subset=[target_column])

    # Pisahkan fitur dan target
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Encode kolom kategorikal
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            encoder = LabelEncoder()
            X_train[col] = encoder.fit_transform(X_train[col].astype(str))
            X_test[col] = encoder.transform(X_test[col].astype(str))

    print(f"✅ Data siap digunakan. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ====================================================
# 2️⃣ Training Model + MLflow Tracking
# ====================================================
def train_and_log_model(X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Regression_Model_Tracking")

    with mlflow.start_run(run_name="RandomForest_HousingPrice", nested=True):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\n📊 HASIL EVALUASI MODEL")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Log hasil dan model
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print("\n✅ Model berhasil dilatih dan dicatat di MLflow!")


# ====================================================
# 3️⃣ Entry Point (untuk MLflow CLI)
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for house price prediction")
    parser.add_argument(
        "--data_path",
        type=str,
        default="houseprice_preprocessing",
        help="Path ke folder preprocessing (berisi train_preprocessed.csv dan test_preprocessed.csv)",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_and_prepare_data(args.data_path)
    train_and_log_model(X_train, X_test, y_train, y_test)
