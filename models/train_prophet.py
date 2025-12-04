import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import argparse
import json
from datetime import datetime
import os

print("=" * 60)
print("NFR ML - Prophet Training")
print("=" * 60)

parser = argparse.ArgumentParser()
parser.add_argument('--top_variants', type=int, default=50)
args = parser.parse_args()

TOP_VARIANTS = args.top_variants

# ==================== LOAD DATA ====================
print("\n📥 Loading training data...")

train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/validation.csv')

print(f"   Train: {len(train_df)} records")
print(f"   Validation: {len(val_df)} records")

# ==================== TRAIN PROPHET PER TOP VARIANTS ====================
# Prophet funziona meglio con serie temporali lunghe
# Alleniamo solo sui top N variant per vendite

print(f"\n📊 Selecting top {TOP_VARIANTS} variants by sales volume...")

top_variants = train_df.groupby('variant_id')['quantity'].sum().nlargest(TOP_VARIANTS).index.tolist()
print(f"   Selected {len(top_variants)} variants")

os.makedirs('models/artifacts/prophet', exist_ok=True)

models = {}
results = {}

for variant_id in top_variants[:10]:  # Train primi 10 per test veloce
    print(f"\n🔮 Training Prophet for variant {variant_id}...")
    
    # Prepara dati per Prophet (serve formato 'ds' e 'y')
    variant_train = train_df[train_df['variant_id'] == variant_id][['date', 'quantity']].copy()
    variant_train.columns = ['ds', 'y']
    
    variant_val = val_df[val_df['variant_id'] == variant_id][['date', 'quantity']].copy()
    variant_val.columns = ['ds', 'y']
    
    if len(variant_train) < 30:  # Skip se pochi dati
        print(f"   ⚠️  Skipped (not enough data)")
        continue
    
    # Train
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    model.fit(variant_train)
    
    # Predict
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Evaluate on validation
    if len(variant_val) > 0:
        val_dates = variant_val['ds'].values
        forecast_val = forecast[forecast['ds'].isin(val_dates)]
        
        if len(forecast_val) > 0:
            mae = mean_absolute_error(variant_val['y'], forecast_val['yhat'])
            print(f"   ✅ MAE: {mae:.2f}")
            
            results[variant_id] = {'mae': float(mae)}
    
    # Save model
    model_path = f'models/artifacts/prophet/prophet_{variant_id}.pkl'
    joblib.dump(model, model_path)
    models[variant_id] = model

# ==================== SAVE METADATA ====================
metadata = {
    'model_type': 'prophet',
    'top_variants': TOP_VARIANTS,
    'trained_variants': list(models.keys()),
    'results': results,
    'trained_at': datetime.now().isoformat()
}

with open('models/artifacts/prophet_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Saved {len(models)} Prophet models")
print("\n✅ Prophet training complete!")
print("=" * 60)
