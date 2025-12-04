import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import json
import subprocess
import os

print("=" * 70)
print("FORECAST DEMAND BLANKS (basato su ML grafiche)")
print("=" * 70)

# Init Firebase
cred = credentials.Certificate("serviceAccountKey.json")
try:
    firebase_admin.initialize_app(cred)
except:
    pass

db = firestore.client()

# 1. Carica mapping graphics → blanks
print("\n📋 Caricamento mapping grafiche → blanks...")
graphics_blanks_ref = db.collection("graphics_blanks").stream()

mapping = {}
for doc in graphics_blanks_ref:
    data = doc.to_dict()
    variant_id_grafica = str(data.get('variant_id_grafica'))
    
    mapping[variant_id_grafica] = {
        'blank_key': data.get('blank_key'),
        'size': data.get('size'),
        'color': data.get('color')
    }

print(f"✅ Mappati {len(mapping)} variant_id grafiche → blanks")

# 2. Genera forecast per top variants usando il modello LSTM
print("\n🤖 Generazione forecast ML per top variants...")

# Leggi train.csv per trovare i top variants
train_df = pd.read_csv('data/train.csv')
train_df['variant_id'] = train_df['variant_id'].astype(str)

# Top 100 variants per volume
top_variants = train_df.groupby('variant_id')['quantity'].sum().sort_values(ascending=False).head(100)

print(f"   Processando {len(top_variants)} top variants...")

# Carica modello e genera forecast
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import joblib
import numpy as np

model = keras.models.load_model('models/artifacts/lstm_demand_forecast.h5', compile=False)
scaler_X = joblib.load('models/artifacts/scaler_X.pkl')
scaler_y = joblib.load('models/artifacts/scaler_y.pkl')

feature_cols = [
    'month', 'day_of_week', 'week_of_year', 'is_weekend',
    'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std',
    'trend_7d', 'is_month_start', 'is_month_end', 'quarter', 'avg_order_value'
]

# 3. Genera forecast e mappa ai blanks
blanks_forecast = []

for variant_id in top_variants.index:
    # Forecast per questa grafica
    variant_data = train_df[train_df['variant_id'] == variant_id].tail(1)
    
    if len(variant_data) == 0:
        continue
    
    try:
        X = variant_data[feature_cols].values
        X_scaled = scaler_X.transform(X)
        X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
        
        y_pred_scaled = model.predict(X_reshaped, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
        
        forecast_7d = max(0, y_pred[0])
        forecast_14d = max(0, y_pred[1])
        forecast_30d = max(0, y_pred[2])
        
        # Mappa al blank
        if variant_id in mapping:
            blank_info = mapping[variant_id]
            
            blanks_forecast.append({
                'variant_id_grafica': variant_id,
                'blank_key': blank_info['blank_key'],
                'size': blank_info['size'],
                'color': blank_info['color'],
                'forecast_7d': forecast_7d,
                'forecast_14d': forecast_14d,
                'forecast_30d': forecast_30d
            })
    except Exception as e:
        print(f"   Errore variant {variant_id}: {e}")
        continue

print(f"✅ Generati {len(blanks_forecast)} forecast mappati ai blanks")

# 4. Aggrega per blank + taglia + colore
df_forecast = pd.DataFrame(blanks_forecast)

if len(df_forecast) == 0:
    print("\n❌ Nessun forecast generato")
    exit(1)

# Aggregazione
blanks_agg = df_forecast.groupby(['blank_key', 'size', 'color']).agg({
    'forecast_7d': 'sum',
    'forecast_14d': 'sum',
    'forecast_30d': 'sum'
}).reset_index()

blanks_agg = blanks_agg.sort_values('forecast_30d', ascending=False)

# 5. Output risultati
print("\n" + "=" * 70)
print("🎯 FORECAST DEMAND BLANKS - PROSSIMI 30 GIORNI")
print("=" * 70)

print("\nTop 20 combinazioni con maggiore domanda prevista:")
for idx, row in blanks_agg.head(20).iterrows():
    print(f"\n{row['blank_key']} - {row['size']} - {row['color']}")
    print(f"  • Prossimi 7 giorni: {row['forecast_7d']:.1f} pz")
    print(f"  • Prossimi 14 giorni: {row['forecast_14d']:.1f} pz")
    print(f"  • Prossimi 30 giorni: {row['forecast_30d']:.1f} pz")

# Aggregazione per tipologia
by_type = df_forecast.groupby('blank_key').agg({
    'forecast_7d': 'sum',
    'forecast_14d': 'sum',
    'forecast_30d': 'sum'
}).reset_index().sort_values('forecast_30d', ascending=False)

print("\n" + "=" * 70)
print("📦 FORECAST PER TIPOLOGIA BLANK")
print("=" * 70)

for _, row in by_type.iterrows():
    print(f"\n{row['blank_key']}")
    print(f"  • 7 giorni: {row['forecast_7d']:.0f} pz")
    print(f"  • 14 giorni: {row['forecast_14d']:.0f} pz")
    print(f"  • 30 giorni: {row['forecast_30d']:.0f} pz")

# Aggregazione per taglia
by_size = df_forecast.groupby('size').agg({
    'forecast_7d': 'sum',
    'forecast_14d': 'sum',
    'forecast_30d': 'sum'
}).reset_index().sort_values('forecast_30d', ascending=False)

print("\n" + "=" * 70)
print("📏 FORECAST PER TAGLIA")
print("=" * 70)

for _, row in by_size.iterrows():
    print(f"\n{row['size']}")
    print(f"  • 7 giorni: {row['forecast_7d']:.0f} pz")
    print(f"  • 14 giorni: {row['forecast_14d']:.0f} pz")
    print(f"  • 30 giorni: {row['forecast_30d']:.0f} pz")

# 6. Salva JSON
output = {
    "generated_at": pd.Timestamp.now().isoformat(),
    "forecast_period": {
        "7_days": "Next 7 days",
        "14_days": "Next 14 days",
        "30_days": "Next 30 days"
    },
    "by_combination": [
        {
            "blank_key": row['blank_key'],
            "size": row['size'],
            "color": row['color'],
            "forecast_7d": float(row['forecast_7d']),
            "forecast_14d": float(row['forecast_14d']),
            "forecast_30d": float(row['forecast_30d'])
        }
        for _, row in blanks_agg.iterrows()
    ],
    "by_type": [
        {
            "blank_key": row['blank_key'],
            "forecast_7d": float(row['forecast_7d']),
            "forecast_14d": float(row['forecast_14d']),
            "forecast_30d": float(row['forecast_30d'])
        }
        for _, row in by_type.iterrows()
    ],
    "by_size": [
        {
            "size": row['size'],
            "forecast_7d": float(row['forecast_7d']),
            "forecast_14d": float(row['forecast_14d']),
            "forecast_30d": float(row['forecast_30d'])
        }
        for _, row in by_size.iterrows()
    ],
    "totals": {
        "forecast_7d": float(blanks_agg['forecast_7d'].sum()),
        "forecast_14d": float(blanks_agg['forecast_14d'].sum()),
        "forecast_30d": float(blanks_agg['forecast_30d'].sum())
    }
}

with open('data/blanks_forecast.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Forecast salvato in: data/blanks_forecast.json")
print("\n" + "=" * 70)
