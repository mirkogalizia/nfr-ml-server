import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================== LOAD MODEL & SCALERS ====================
try:
    model = keras.models.load_model(
        'models/artifacts/lstm_demand_forecast.h5',
        compile=False
    )
    scaler_X = joblib.load('models/artifacts/scaler_X.pkl')
    scaler_y = joblib.load('models/artifacts/scaler_y.pkl')
except Exception as e:
    print(json.dumps({"error": f"Model loading failed: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

# ==================== LOAD DATA ====================
try:
    train_df = pd.read_csv('data/train.csv')
    train_df['variant_id'] = train_df['variant_id'].astype(str)
    
    # Trova top N variants per total quantity
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    top_variants = train_df.groupby('variant_id')['quantity'].sum().sort_values(ascending=False).head(top_n)
    
    print(f"Generating forecasts for top {top_n} variants...", file=sys.stderr)
    
    feature_cols = [
        'month', 'day_of_week', 'week_of_year', 'is_weekend',
        'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std',
        'trend_7d', 'is_month_start', 'is_month_end', 'quarter', 'avg_order_value'
    ]
    
    results = []
    
    for variant_id in top_variants.index:
        try:
            variant_data = train_df[train_df['variant_id'] == variant_id].tail(1)
            
            if len(variant_data) == 0:
                continue
            
            X = variant_data[feature_cols].values
            X_scaled = scaler_X.transform(X)
            X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
            
            y_pred_scaled = model.predict(X_reshaped, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
            
            result = {
                "variant_id": variant_id,
                "forecast_7d": float(max(0, y_pred[0])),
                "forecast_14d": float(max(0, y_pred[1])),
                "forecast_30d": float(max(0, y_pred[2])),
                "total_sales": float(top_variants[variant_id]),
                "last_7d_avg": float(variant_data['sales_7d_avg'].values[0]),
                "last_30d_avg": float(variant_data['sales_30d_avg'].values[0])
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing variant {variant_id}: {str(e)}", file=sys.stderr)
            continue
    
    # Output JSON
    output = {
        "status": "success",
        "count": len(results),
        "forecasts": results,
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    print(json.dumps(output))
    
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
