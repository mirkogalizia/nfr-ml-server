import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--variant_id', type=str, required=True)
args = parser.parse_args()

variant_id = args.variant_id

# ==================== LOAD MODEL & SCALERS ====================
try:
    model = keras.models.load_model('models/artifacts/lstm_demand_forecast.h5')
    scaler_X = joblib.load('models/artifacts/scaler_X.pkl')
    scaler_y = joblib.load('models/artifacts/scaler_y.pkl')
except Exception as e:
    print(json.dumps({"error": f"Model not found: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

# ==================== LOAD RECENT DATA ====================
try:
    # Carica ultimi dati del variant da train.csv (in produzione useresti dati real-time)
    train_df = pd.read_csv('data/train.csv')
    variant_data = train_df[train_df['variant_id'] == variant_id].tail(1)
    
    if len(variant_data) == 0:
        print(json.dumps({"error": f"Variant {variant_id} not found in training data"}), file=sys.stderr)
        sys.exit(1)
    
    feature_cols = [
        'month', 'day_of_week', 'week_of_year', 'is_weekend',
        'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std'
    ]
    
    X = variant_data[feature_cols].values
    
    # Normalize
    X_scaled = scaler_X.transform(X)
    X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    
    # Predict
    y_pred_scaled = model.predict(X_reshaped, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
    
    # Output JSON
    result = {
        "variant_id": variant_id,
        "forecast_7d": float(y_pred[0]),
        "forecast_14d": float(y_pred[1]),
        "forecast_30d": float(y_pred[2]),
        "confidence": 0.85,  # Placeholder, puoi calcolare confidence interval
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    print(json.dumps(result))
    
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
