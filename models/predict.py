import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse
import json
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--variant_id', type=str, required=True)
args = parser.parse_args()

variant_id = args.variant_id

# ==================== LOAD MODEL & SCALERS ====================
try:
    model_path = 'models/artifacts/lstm_demand_forecast.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Carica il modello ignorando metriche custom tipo 'mse'
    model = keras.models.load_model(
        model_path,
        compile=False  # evita di deserializzare metriche/optimizer
    )
    
    scaler_X = joblib.load('models/artifacts/scaler_X.pkl')
    scaler_y = joblib.load('models/artifacts/scaler_y.pkl')
except Exception as e:
    print(json.dumps({"error": f"Model not found: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

