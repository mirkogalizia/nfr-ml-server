import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

print("=" * 60)
print("NFR ML - Data Preparation")
print("=" * 60)

print("\n🔥 Initializing Firebase...")
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
print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
print(f"   Unique variants: {df['variant_id'].nunique()}")

# ==== DATA CLEANING ====
print("\n🧹 Cleaning data...")

# Converti date
df['date'] = pd.to_datetime(df['date'])
df['created_at'] = pd.to_datetime(df['created_at'])

# Rimuovi record senza dati essenziali
initial_len = len(df)
df = df.dropna(subset=['variant_id', 'quantity', 'date'])
df = df[df['quantity'] > 0]  # Solo ordini validi
print(f"   Removed {initial_len - len(df)} invalid records")

# Converti tipi
df['quantity'] = df['quantity'].astype(int)
df['revenue'] = df['revenue'].astype(float)
df['variant_id'] = df['variant_id'].astype(str)

print(f"   → {len(df)} valid records")

# ==== AGGREGATE DAILY SALES ====
print("\n📊 Aggregating daily sales per variant...")

daily_sales = df.groupby(['variant_id', 'date']).agg({
    'quantity': 'sum',
    'revenue': 'sum',
    'order_id': 'count',  # numero ordini
    'year': 'first',
    'month': 'first',
    'day_of_week': 'first',
    'week_of_year': 'first',
}).reset_index()

daily_sales.rename(columns={'order_id': 'num_orders'}, inplace=True)

# Aggiungi is_weekend
daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)

print(f"   → {len(daily_sales)} daily records")
print(f"   → Variants with data: {daily_sales['variant_id'].nunique()}")

# ==== ROLLING FEATURES ====
print("\n📈 Computing rolling statistics (trend indicators)...")

def compute_rolling_features(group):
    group = group.sort_values('date')
    
    # Medie mobili
    group['sales_7d_avg'] = group['quantity'].rolling(7, min_periods=1).mean()
    group['sales_14d_avg'] = group['quantity'].rolling(14, min_periods=1).mean()
    group['sales_30d_avg'] = group['quantity'].rolling(30, min_periods=1).mean()
    
    # Deviazione standard (volatilità)
    group['sales_7d_std'] = group['quantity'].rolling(7, min_periods=1).std().fillna(0)
    
    # Trend (differenza % rispetto a media precedente)
    group['trend_7d'] = (group['sales_7d_avg'] - group['sales_7d_avg'].shift(7)) / (group['sales_7d_avg'].shift(7) + 1)
    group['trend_7d'] = group['trend_7d'].fillna(0)
    
    return group

daily_sales = daily_sales.groupby('variant_id', group_keys=False).apply(compute_rolling_features)

# ==== CREATE TARGETS ====
print("\n🎯 Creating target variables (future demand)...")

def create_targets(group):
    group = group.sort_values('date')
    
    # Target = somma vendite nei prossimi N giorni
    group['target_7d'] = group['quantity'].rolling(7, min_periods=1).sum().shift(-7)
    group['target_14d'] = group['quantity'].rolling(14, min_periods=1).sum().shift(-14)
    group['target_30d'] = group['quantity'].rolling(30, min_periods=1).sum().shift(-30)
    
    return group

daily_sales = daily_sales.groupby('variant_id', group_keys=False).apply(create_targets)

# Rimuovi righe senza target (ultime date per cui non abbiamo futuro)
initial_len = len(daily_sales)
daily_sales = daily_sales.dropna(subset=['target_7d', 'target_14d', 'target_30d'])
print(f"   Removed {initial_len - len(daily_sales)} records without future data")
print(f"   → {len(daily_sales)} records with targets")

# ==== FEATURE ENGINEERING AVANZATO ====
print("\n🔧 Advanced feature engineering...")

# Seasonality indicators
daily_sales['is_month_start'] = (daily_sales['date'].dt.day <= 7).astype(int)
daily_sales['is_month_end'] = (daily_sales['date'].dt.day >= 24).astype(int)
daily_sales['quarter'] = daily_sales['date'].dt.quarter

# Interaction features
daily_sales['avg_order_value'] = daily_sales['revenue'] / (daily_sales['quantity'] + 1)

print("   ✅ Feature engineering complete")

# ==== SPLIT TRAIN/VALIDATION/TEST ====
print("\n✂️  Splitting dataset (temporal split)...")

# Ordina per data
daily_sales = daily_sales.sort_values('date')

# Split temporale: 70% train, 15% val, 15% test
train_size = int(len(daily_sales) * 0.7)
val_size = int(len(daily_sales) * 0.15)

train_df = daily_sales.iloc[:train_size]
val_df = daily_sales.iloc[train_size:train_size + val_size]
test_df = daily_sales.iloc[train_size + val_size:]

print(f"   Train: {len(train_df)} records")
print(f"      → Date range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"   Validation: {len(val_df)} records")
print(f"      → Date range: {val_df['date'].min()} to {val_df['date'].max()}")
print(f"   Test: {len(test_df)} records")
print(f"      → Date range: {test_df['date'].min()} to {test_df['date'].max()}")

# ==== SAVE DATASETS ====
print("\n💾 Saving processed datasets...")

os.makedirs('data', exist_ok=True)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/validation.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Save metadata
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
    'feature_columns': [
        'month', 'day_of_week', 'week_of_year', 'is_weekend',
        'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std',
        'trend_7d', 'is_month_start', 'is_month_end', 'quarter', 'avg_order_value'
    ],
    'target_columns': ['target_7d', 'target_14d', 'target_30d'],
    'created_at': datetime.now().isoformat()
}

with open('data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Data preparation complete!")
print("=" * 60)
print(f"\nFiles saved:")
print(f"   📄 data/train.csv ({len(train_df):,} rows)")
print(f"   📄 data/validation.csv ({len(val_df):,} rows)")
print(f"   📄 data/test.csv ({len(test_df):,} rows)")
print(f"   📄 data/metadata.json")
print("\nNext step: Run training with /train/lstm")
print("=" * 60)

