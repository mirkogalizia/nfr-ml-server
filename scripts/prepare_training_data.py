import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

print("🔥 Initializing Firebase...")
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

print("📥 Fetching sales data from Firestore...")
sales_ref = db.collection("ml_sales_data").stream()

data = []
for doc in sales_ref:
    data.append(doc.to_dict())

df = pd.DataFrame(data)
print(f"✅ Loaded {len(df)} sales records")

# ==== DATA CLEANING ====
print("\n🧹 Cleaning data...")

# Converti date in datetime
df['date'] = pd.to_datetime(df['date'])
df['created_at'] = pd.to_datetime(df['created_at'])

# Rimuovi record senza variant_id o quantity
df = df.dropna(subset=['variant_id', 'quantity'])

# Converti quantity in numeric
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df = df.dropna(subset=['quantity'])

print(f"   → {len(df)} records after cleaning")

# ==== FEATURE ENGINEERING ====
print("\n🔧 Creating features...")

# Time features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0=lunedì, 6=domenica
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Aggrega vendite giornaliere per variant
daily_sales = df.groupby(['variant_id', 'date']).agg({
    'quantity': 'sum',
    'revenue': 'sum',
    'year': 'first',
    'month': 'first',
    'day_of_week': 'first',
    'week_of_year': 'first',
    'is_weekend': 'first'
}).reset_index()

print(f"   → {len(daily_sales)} daily aggregated records")

# ==== ROLLING FEATURES (trend) ====
print("\n📈 Computing rolling statistics...")

def compute_rolling_features(group):
    group = group.sort_values('date')
    group['sales_7d_avg'] = group['quantity'].rolling(7, min_periods=1).mean()
    group['sales_14d_avg'] = group['quantity'].rolling(14, min_periods=1).mean()
    group['sales_30d_avg'] = group['quantity'].rolling(30, min_periods=1).mean()
    group['sales_7d_std'] = group['quantity'].rolling(7, min_periods=1).std().fillna(0)
    return group

daily_sales = daily_sales.groupby('variant_id', group_keys=False).apply(compute_rolling_features)

# ==== TARGET: vendite prossimi 7/14/30 giorni ====
print("\n🎯 Creating target variables (future sales)...")

def create_targets(group):
    group = group.sort_values('date')
    group['target_7d'] = group['quantity'].rolling(7, min_periods=1).sum().shift(-7)
    group['target_14d'] = group['quantity'].rolling(14, min_periods=1).sum().shift(-14)
    group['target_30d'] = group['quantity'].rolling(30, min_periods=1).sum().shift(-30)
    return group

daily_sales = daily_sales.groupby('variant_id', group_keys=False).apply(create_targets)

# Rimuovi righe senza target (ultime date)
daily_sales = daily_sales.dropna(subset=['target_7d', 'target_14d', 'target_30d'])

print(f"   → {len(daily_sales)} records with targets")

# ==== SPLIT TRAIN/VALIDATION/TEST ====
print("\n✂️ Splitting dataset...")

# Ordina per data
daily_sales = daily_sales.sort_values('date')

# Split temporale (no shuffle per serie temporali!)
train_size = int(len(daily_sales) * 0.7)
val_size = int(len(daily_sales) * 0.15)

train_df = daily_sales.iloc[:train_size]
val_df = daily_sales.iloc[train_size:train_size + val_size]
test_df = daily_sales.iloc[train_size + val_size:]

print(f"   Train: {len(train_df)} records ({train_df['date'].min()} → {train_df['date'].max()})")
print(f"   Validation: {len(val_df)} records ({val_df['date'].min()} → {val_df['date'].max()})")
print(f"   Test: {len(test_df)} records ({test_df['date'].min()} → {test_df['date'].max()})")

# ==== SAVE DATASETS ====
print("\n💾 Saving processed datasets...")

os.makedirs('data', exist_ok=True)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/validation.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Salva anche metadata
metadata = {
    'total_records': len(daily_sales),
    'train_records': len(train_df),
    'val_records': len(val_df),
    'test_records': len(test_df),
    'date_range': {
        'min': str(daily_sales['date'].min()),
        'max': str(daily_sales['date'].max())
    },
    'unique_variants': int(daily_sales['variant_id'].nunique()),
    'features': list(train_df.columns),
    'created_at': datetime.now().isoformat()
}

with open('data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Data preparation complete!")
print(f"   Files saved in data/:")
print(f"   - train.csv ({len(train_df)} rows)")
print(f"   - validation.csv ({len(val_df)} rows)")
print(f"   - test.csv ({len(test_df)} rows)")
print(f"   - metadata.json")

