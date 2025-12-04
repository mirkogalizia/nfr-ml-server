import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse
import json
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--variant_id', type=str, required=True)
args = parser.parse_args()

variant_id = str(args.variant_id)  # Forza a stringa

# ==================== LOAD MODEL & SCALERS ====================
try:
    model_path = 'models/artifacts/lstm_demand_forecast.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Carica il modello ignorando metriche custom
    model = keras.models.load_model(
        model_path,
        compile=False  # evita di deserializzare metriche/optimizer
    )
    
    scaler_X = joblib.load('models/artifacts/scaler_X.pkl')
    scaler_y = joblib.load('models/artifacts/scaler_y.pkl')
    
except Exception as e:
    print(json.dumps({"error": f"Model loading failed: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

# ==================== LOAD RECENT DATA ====================
try:
    # Leggi CSV
    train_df = pd.read_csv('data/train.csv')
    
    # Converti variant_id a stringa per matching consistente
    train_df['variant_id'] = train_df['variant_id'].astype(str)
    
    # Filtra per il variant richiesto
    variant_data = train_df[train_df['variant_id'] == variant_id]
    
    if len(variant_data) == 0:
        print(json.dumps({
            "error": f"Variant {variant_id} not found in training data",
            "tip": "Use a variant_id from your Firebase ml_sales_data collection"
        }), file=sys.stderr)
        sys.exit(1)
    
    # Prendi l'ultima riga (dati più recenti)
    variant_data = variant_data.tail(1)
    
    # Feature columns (ordine deve corrispondere al training)
    feature_cols = [
        'month', 'day_of_week', 'week_of_year', 'is_weekend',
        'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std',
        'trend_7d', 'is_month_start', 'is_month_end', 'quarter', 'avg_order_value'
    ]
    
    # Verifica che tutte le feature esistano
    missing_features = [f for f in feature_cols if f not in variant_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    X = variant_data[feature_cols].values
    
    # Normalize
    X_scaled = scaler_X.transform(X)
    
    # Reshape per LSTM (samples, timesteps, features)
    X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    
    # ==================== PREDICT ====================
    y_pred_scaled = model.predict(X_reshaped, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
    
    # ==================== OUTPUT JSON ====================
    result = {
        "variant_id": variant_id,
        "forecast_7d": float(max(0, y_pred[0])),
        "forecast_14d": float(max(0, y_pred[1])),
        "forecast_30d": float(max(0, y_pred[2])),
        "confidence": 0.85,
        "last_7d_avg": float(variant_data['sales_7d_avg'].values[0]),
        "last_30d_avg": float(variant_data['sales_30d_avg'].values[0]),
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    # Stampa su stdout (FastAPI lo cattura)
    print(json.dumps(result))
    
except Exception as e:
    error_response = {
        "error": f"Prediction failed: {str(e)}",
        "variant_id": variant_id
    }
    print(json.dumps(error_response), file=sys.stderr)
    sys.exit(1)

