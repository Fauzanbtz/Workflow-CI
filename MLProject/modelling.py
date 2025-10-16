import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import LabelEncoder

# ====================================================
# 1. Load dan Persiapan Data
# ====================================================
train_data = pd.read_csv('houseprice_preprocessing/train_preprocessed.csv')
test_data = pd.read_csv('houseprice_preprocessing/test_preprocessed.csv')

target_column = 'Price (in rupees)'

# Drop baris tanpa nilai target
train_data = train_data.dropna(subset=[target_column])
test_data = test_data.dropna(subset=[target_column])

# Pisahkan fitur dan target
X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# Encode kolom kategorikal (jika masih ada object)
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col].astype(str))
        X_test[col] = encoder.transform(X_test[col].astype(str))

# ====================================================
# 2. Inisialisasi MLflow
# ====================================================
mlflow.set_experiment("Regression_Model_Tracking")
mlflow.sklearn.autolog()

# ====================================================
# 3. Training Model + Tracking
# ====================================================
with mlflow.start_run(run_name="RandomForest_HousingPrice"):
    start_time = time.time()

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    end_time = time.time()

    # ====================================================
    # 4. Evaluation Lengkap
    # ====================================================
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    evs = explained_variance_score(y_test, preds)
    runtime = end_time - start_time

    # Toleransi 10% (semacam “recall” untuk regresi)
    tolerance = 0.10
    within_tolerance = np.mean(np.abs(preds - y_test) <= tolerance * y_test)

    print("\n=== 📊 Evaluation Metrics ===")
    print(f"MAE   : {mae:.3f}")
    print(f"MSE   : {mse:.3f}")
    print(f"RMSE  : {rmse:.3f}")
    print(f"R²    : {r2:.3f}")
    print(f"MAPE  : {mape:.3f}")
    print(f"EVS   : {evs:.3f}")
    print(f"Tolerance(10%) Accuracy : {within_tolerance:.3f}")
    print(f"Runtime (s) : {runtime:.2f}")

    # ====================================================
    # 5. Logging Metric ke MLflow
    # ====================================================
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("ExplainedVariance", evs)
    mlflow.log_metric("Tolerance_Accuracy", within_tolerance)
    mlflow.log_metric("Runtime_sec", runtime)

    # ====================================================
    # 6. Visualisasi & Logging Artifacts
    # ====================================================

    # --- Residual Plot ---
    residuals = y_test - preds
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=40)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (y_test - y_pred)")
    residual_path = "residual_plot.png"
    plt.savefig(residual_path)
    mlflow.log_artifact(residual_path)
    plt.close()

    # --- Prediksi vs Aktual ---
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=preds, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Price")
    pred_plot_path = "actual_vs_pred.png"
    plt.savefig(pred_plot_path)
    mlflow.log_artifact(pred_plot_path)
    plt.close()

    # --- Feature Importance ---
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(15))
    plt.title("Top 15 Important Features")
    plt.tight_layout()
    feat_imp_path = "feature_importance.png"
    plt.savefig(feat_imp_path)
    mlflow.log_artifact(feat_imp_path)
    plt.close()

    # Simpan model
    mlflow.sklearn.log_model(model, "model")

# ====================================================
# 7. Tutup MLflow
# ====================================================
mlflow.end_run()

print("\n✅ Semua metric, plot, dan model berhasil dilog ke MLflow!")
