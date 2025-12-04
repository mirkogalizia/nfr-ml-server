import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import argparse
import json
from datetime import datetime
import os

print("=" * 60)
print("NFR ML - Random Forest Training")
print("=" * 60)

parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
args = parser.parse_args()

N_ESTIMATORS = args.n_estimators

# ==================== LOAD DATA ====================
print("\n📥 Loading training data...")

train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/validation.csv')

print(f"   Train: {len(train_df)} records")
print(f"   Validation: {len(val_df)} records")

# ==================== FEATURES ====================
feature_cols = [
    'month', 'day_of_week', 'week_of_year', 'is_weekend',
    'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std',
    'trend_7d', 'is_month_start', 'is_month_end', 'quarter', 'avg_order_value'
]

X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values

targets = ['target_7d', 'target_14d', 'target_30d']
models = {}
results = {}

os.makedirs('models/artifacts', exist_ok=True)

for target in targets:
    print(f"\n🌲 Training Random Forest for {target}...")
    
    y_train = train_df[target].values
    y_val = val_df[target].values
    
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Usa tutti i core CPU
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"   ✅ {target} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Save model
    model_path = f'models/artifacts/random_forest_{target}.pkl'
    joblib.dump(model, model_path)
    
    models[target] = model
    results[target] = {
        'mae': float(mae),
        'rmse': float(rmse),
        'n_estimators': N_ESTIMATORS
    }

# ==================== SAVE METADATA ====================
metadata = {
    'model_type': 'random_forest',
    'n_estimators': N_ESTIMATORS,
    'results': results,
    'feature_importance': {
        target: {
            feature: float(importance)
            for feature, importance in zip(feature_cols, models[target].feature_importances_)
        }
        for target in targets
    },
    'trained_at': datetime.now().isoformat()
}

with open('models/artifacts/random_forest_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n💾 Models saved:")
for target in targets:
    print(f"   ✅ models/artifacts/random_forest_{target}.pkl")

print("\n✅ Random Forest training complete!")
print("=" * 60)
