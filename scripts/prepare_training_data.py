import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

print("=" * 60)
print("NFR ML - Data Preparation (Fixed for short history)")
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

df['date'] = pd.to_datetime(df['date'])
df['created_at'] = pd.to_datetime(df['created_at'])

initial_len = len(df)
df = df.dropna(subset=['variant_id', 'quantity', 'date'])
df = df[df['quantity'] > 0]
print(f"   Removed {initial_len - len(df)} invalid records")

df['quantity'] = df['quantity'].astype(int)
df['revenue'] = df['revenue'].astype(float)
df['variant_id'] = df['variant_id'].astype(str)

print(f"   → {len(df)} valid records")

# ==== AGGREGATE DAILY SALES ====
print("\n📊 Aggregating daily sales per variant...")

daily_sales = df.groupby(['variant_id', 'date']).agg({
    'quantity': 'sum',
    'revenue': 'sum',
    'order_id': 'count',
    'year': 'first',
    'month': 'first',
    'day_of_week': 'first',
    'week_of_year': 'first',
}).reset_index()

daily_sales.rename(columns={'order_id': 'num_orders'}, inplace=True)
daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)

print(f"   → {len(daily_sales)} daily records")
print(f"   → Variants with data: {daily_sales['variant_id'].nunique()}")

# ==== FILL MISSING DATES (importante per serie temporali) ====
print("\n📅 Filling missing dates for each variant...")

# Per ogni variant, crea serie temporale completa
all_dates = pd.date_range(daily_sales['date'].min(), daily_sales['date'].max(), freq='D')
complete_data = []

for variant_id in daily_sales['variant_id'].unique():
    variant_data = daily_sales[daily_sales['variant_id'] == variant_id].copy()
    
    # Crea dataframe con tutte le date
    complete_dates = pd.DataFrame({'date': all_dates})
    complete_dates['variant_id'] = variant_id
    
    # Merge con dati esistenti
    merged = complete_dates.merge(variant_data, on=['variant_id', 'date'], how='left')
    
    # Riempi giorni senza vendite con 0
    merged['quantity'] = merged['quantity'].fillna(0)
    merged['revenue'] = merged['revenue'].fillna(0)
    merged['num_orders'] = merged['num_orders'].fillna(0)
    
    # Riempi feature temporali
    merged['year'] = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month
    merged['day_of_week'] = merged['date'].dt.dayofweek
    merged['week_of_year'] = merged['date'].dt.isocalendar().week
    merged['is_weekend'] = merged['day_of_week'].isin([5, 6]).astype(int)
    
    complete_data.append(merged)

daily_sales = pd.concat(complete_data, ignore_index=True)
print(f"   → {len(daily_sales)} records after filling gaps")

# ==== ROLLING FEATURES ====
print("\n📈 Computing rolling statistics...")

def compute_rolling_features(group):
    group = group.sort_values('date')
    
    group['sales_7d_avg'] = group['quantity'].rolling(7, min_periods=1).mean()
    group['sales_14d_avg'] = group['quantity'].rolling(14, min_periods=1).mean()
    group['sales_30d_avg'] = group['quantity'].rolling(30, min_periods=1).mean()
    group['sales_7d_std'] = group['quantity'].rolling(7, min_periods=1).std().fillna(0)
    
    # Trend
    group['trend_7d'] = (group['sales_7d_avg'] - group['sales_7d_avg'].shift(7)) / (group['sales_7d_avg'].shift(7) + 1)
    group['trend_7d'] = group['trend_7d'].fillna(0)
    
    return group

daily_sales = daily_sales.groupby('variant_id', group_keys=False).apply(compute_rolling_features)

# ==== ADVANCED FEATURES ====
print("\n🔧 Creating advanced features...")

daily_sales['is_month_start'] = (daily_sales['date'].dt.day <= 7).astype(int)
daily_sales['is_month_end'] = (daily_sales['date'].dt.day >= 24).astype(int)
daily_sales['quarter'] = daily_sales['date'].dt.quarter
daily_sales['avg_order_value'] = daily_sales['revenue'] / (daily_sales['quantity'] + 1)

# ==== NUOVO APPROCCIO: SUPERVISED LEARNING CON LAG ====
print("\n🎯 Creating supervised learning dataset with lags...")

# Invece di target futuro, creiamo feature "passate" e target "domani"
def create_supervised_dataset(group):
    group = group.sort_values('date')
    
    # Target = vendite domani (shift -1)
    group['target_next_day'] = group['quantity'].shift(-1)
    
    # Target aggregati per periodo
    # Per 7 giorni: media dei prossimi 7 giorni
    group['target_7d_avg'] = group['quantity'].rolling(7, min_periods=1).mean().shift(-7)
    group['target_14d_avg'] = group['quantity'].rolling(14, min_periods=1).mean().shift(-14)
    group['target_30d_avg'] = group['quantity'].rolling(30, min_periods=1).mean().shift(-30)
    
    return group

daily_sales = daily_sales.groupby('variant_id', group_keys=False).apply(create_supervised_dataset)

# Rimuovi solo le righe senza target_next_day (ultimo giorno per variant)
initial_len = len(daily_sales)
daily_sales = daily_sales.dropna(subset=['target_next_day'])
print(f"   Removed {initial_len - len(daily_sales)} records (last day per variant)")
print(f"   → {len(daily_sales)} records with targets")

# Per i target aggregati, riempi NaN con la media dei prossimi disponibili
daily_sales['target_7d_avg'] = daily_sales.groupby('variant_id')['target_7d_avg'].transform(
    lambda x: x.fillna(x.mean())
)
daily_sales['target_14d_avg'] = daily_sales.groupby('variant_id')['target_14d_avg'].transform(
    lambda x: x.fillna(x.mean())
)
daily_sales['target_30d_avg'] = daily_sales.groupby('variant_id')['target_30d_avg'].transform(
    lambda x: x.fillna(x.mean())
)

# ==== TEMPORAL SPLIT ====
print("\n✂️  Splitting dataset (temporal split)...")

# Ordina per data
daily_sales = daily_sales.sort_values('date')

# Usa ultimi 15% per test, 15% prima per validation, resto per train
split_date_test = daily_sales['date'].max() - timedelta(days=int(len(all_dates) * 0.15))
split_date_val = split_date_test - timedelta(days=int(len(all_dates) * 0.15))

train_df = daily_sales[daily_sales['date'] < split_date_val]
val_df = daily_sales[(daily_sales['date'] >= split_date_val) & (daily_sales['date'] < split_date_test)]
test_df = daily_sales[daily_sales['date'] >= split_date_test]

print(f"   Train: {len(train_df)} records ({train_df['date'].min()} → {train_df['date'].max()})")
print(f"   Validation: {len(val_df)} records ({val_df['date'].min()} → {val_df['date'].max()})")
print(f"   Test: {len(test_df)} records ({test_df['date'].min()} → {test_df['date'].max()})")

# ==== SAVE DATASETS ====
print("\n💾 Saving processed datasets...")

os.makedirs('data', exist_ok=True)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/validation.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Metadata
metadata = {
    'total_records': len(daily_sales),
    'train_records': len(train_df),
    'val_records': len(val_df),
    'test_records': len(test_df),
    'date_range': {
        'min': str(daily_sales['date'].min()),
        'max': str(daily_sales['date'].max()),
        'days_covered': (daily_sales['date'].max() - daily_sales['date'].min()).days
    },
    'unique_variants': int(daily_sales['variant_id'].nunique()),
    'features': list(train_df.columns),
    'feature_columns': [
        'month', 'day_of_week', 'week_of_year', 'is_weekend',
        'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std',
        'trend_7d', 'is_month_start', 'is_month_end', 'quarter', 'avg_order_value'
    ],
    'target_columns': ['target_next_day', 'target_7d_avg', 'target_14d_avg', 'target_30d_avg'],
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
print(f"\n📊 Dataset covers {(daily_sales['date'].max() - daily_sales['date'].min()).days} days")
print(f"   {daily_sales['variant_id'].nunique()} unique variants")
print("\nNext step: Run training with /train/lstm")
print("=" * 60)


