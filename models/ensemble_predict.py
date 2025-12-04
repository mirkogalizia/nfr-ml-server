import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
import joblib
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--variant_id', type=str, required=True)
args = parser.parse_args()

variant_id = args.variant_id

# ==================== LOAD MODELS ====================
try:
    # LSTM
    lstm_model = keras.models.load_model('models/artifacts/lstm_demand_forecast.h5')
    scaler_X = joblib.load('models/artifacts/scaler_X.pkl')
    scaler_y = joblib.load('models/artifacts/scaler_y.pkl')
    
    # XGBoost
    xgb_7d = xgb.XGBRegressor()
    xgb_7d.load_model('models/artifacts/xgboost_target_7d.json')
    xgb_14d = xgb.XGBRegressor()
    xgb_14d.load_model('models/artifacts/xgboost_target_14d.json')
    xgb_30d = xgb.XGBRegressor()
    xgb_30d.load_model('models/artifacts/xgboost_target_30d.json')
    
    # Random Forest
    rf_7d = joblib.load('models/artifacts/random_forest_target_7d.pkl')
    rf_14d = joblib.load('models/artifacts/random_forest_target_14d.pkl')
    rf_30d = joblib.load('models/artifacts/random_forest_target_30d.pkl')
    
except Exception as e:
    print(json.dumps({"error": f"Models not found: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

# ==================== LOAD RECENT DATA ====================
try:
    train_df = pd.read_csv('data/train.csv')
    variant_data = train_df[train_df['variant_id'] == variant_id].tail(1)
    
    if len(variant_data) == 0:
        print(json.dumps({"error": f"Variant {variant_id} not found"}), file=sys.stderr)
        sys.exit(1)
    
    feature_cols = [
        'month', 'day_of_week', 'week_of_year', 'is_weekend',
        'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std',
        'trend_7d', 'is_month_start', 'is_month_end', 'quarter', 'avg_order_value'
    ]
    
    X = variant_data[feature_cols].values
    
    # ==================== PREDICTIONS ====================
    
    # LSTM
    X_scaled = scaler_X.transform(X)
    X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    lstm_pred_scaled = lstm_model.predict(X_reshaped, verbose=0)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)[0]
    
    # XGBoost
    xgb_pred_7d = xgb_7d.predict(X)[0]
    xgb_pred_14d = xgb_14d.predict(X)[0]
    xgb_pred_30d = xgb_30d.predict(X)[0]
    
    # Random Forest
    rf_pred_7d = rf_7d.predict(X)[0]
    rf_pred_14d = rf_14d.predict(X)[0]
    rf_pred_30d = rf_30d.predict(X)[0]
    
    # ==================== ENSEMBLE (weighted average) ====================
    # Pesi basati su performance storica (puoi ottimizzarli)
    weights = {
        'lstm': 0.35,
        'xgboost': 0.40,
        'random_forest': 0.25
    }
    
    forecast_7d = (
        lstm_pred[0] * weights['lstm'] +
        xgb_pred_7d * weights['xgboost'] +
        rf_pred_7d * weights['random_forest']
    )
    
    forecast_14d = (
        lstm_pred[1] * weights['lstm'] +
        xgb_pred_14d * weights['xgboost'] +
        rf_pred_14d * weights['random_forest']
    )
    
    forecast_30d = (
        lstm_pred[2] * weights['lstm'] +
        xgb_pred_30d * weights['xgboost'] +
        rf_pred_30d * weights['random_forest']
    )
    
    # Calcola agreement (quanto i modelli sono d'accordo)
    agreement_7d = 1 - (np.std([lstm_pred[0], xgb_pred_7d, rf_pred_7d]) / (np.mean([lstm_pred[0], xgb_pred_7d, rf_pred_7d]) + 1))
    agreement_14d = 1 - (np.std([lstm_pred[1], xgb_pred_14d, rf_pred_14d]) / (np.mean([lstm_pred[1], xgb_pred_14d, rf_pred_14d]) + 1))
    agreement_30d = 1 - (np.std([lstm_pred[2], xgb_pred_30d, rf_pred_30d]) / (np.mean([lstm_pred[2], xgb_pred_30d, rf_pred_30d]) + 1))
    
    confidence = float(np.mean([agreement_7d, agreement_14d, agreement_30d]))
    
    # Output JSON
    result = {
        "variant_id": variant_id,
        "forecast_7d": float(max(0, forecast_7d)),
        "forecast_14d": float(max(0, forecast_14d)),
        "forecast_30d": float(max(0, forecast_30d)),
        "confidence": min(1.0, max(0.0, confidence)),
        "model_predictions": {
            "lstm": [float(lstm_pred[0]), float(lstm_pred[1]), float(lstm_pred[2])],
            "xgboost": [float(xgb_pred_7d), float(xgb_pred_14d), float(xgb_pred_30d)],
            "random_forest": [float(rf_pred_7d), float(rf_pred_14d), float(rf_pred_30d)]
        },
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    print(json.dumps(result))
    
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
