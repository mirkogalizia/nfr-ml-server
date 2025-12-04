import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import json

print("=" * 60)
print("NFR ML - Blanks Data Preparation")
print("=" * 60)

# Init Firebase
cred = credentials.Certificate("serviceAccountKey.json")
try:
    firebase_admin.initialize_app(cred)
except:
    pass

db = firestore.client()

print("\n📥 Fetching sales data from Firestore...")
sales_ref = db.collection("ml_sales_data")
docs = sales_ref.stream()

sales_data = []
for doc in docs:
    data = doc.to_dict()
    sales_data.append(data)

print(f"✅ Loaded {len(sales_data)} sales records")

# Converti in DataFrame
df = pd.DataFrame(sales_data)

# Data range
print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
print(f"   Unique variants: {df['variant_id'].nunique()}")

# Load blanks mapping
print("\n🗺️  Loading blanks mapping...")
with open('data/blanks_mapping.json', 'r', encoding='utf-8') as f:
    blanks_mapping = json.load(f)

print(f"✅ Loaded mapping for {len(blanks_mapping)} variants")

# Mappa ogni vendita al blank
print("\n🔗 Mapping sales to blanks...")

def map_to_blank(variant_id):
    """Mappa variant_id al blank (tipo, taglia, colore)"""
    variant_str = str(variant_id)
    if variant_str in blanks_mapping:
        blank_info = blanks_mapping[variant_str]
        return pd.Series({
            'blank_type': blank_info.get('blank_key', 'unknown'),
            'size': blank_info.get('size', 'unknown'),
            'color': blank_info.get('color', 'unknown')
        })
    else:
        return pd.Series({
            'blank_type': 'unknown',
            'size': 'unknown',
            'color': 'unknown'
        })

# Applica mapping
df[['blank_type', 'size', 'color']] = df['variant_id'].apply(map_to_blank)

# Rimuovi unmapped
df_mapped = df[df['blank_type'] != 'unknown'].copy()
print(f"✅ Mapped {len(df_mapped)} records ({len(df) - len(df_mapped)} unmapped)")

# Converti date
df_mapped['date'] = pd.to_datetime(df_mapped['date'])

# Aggrega per blank + data
print("\n📊 Aggregating daily sales per blank combination...")
df_agg = df_mapped.groupby(['date', 'blank_type', 'size', 'color']).agg({
    'quantity': 'sum',
    'revenue': 'sum'
}).reset_index()

print(f"✅ {len(df_agg)} daily blank records")

# Crea chiave univoca per ogni combinazione blank
df_agg['blank_id'] = (
    df_agg['blank_type'] + '_' + 
    df_agg['size'] + '_' + 
    df_agg['color']
)

print(f"   Unique blank combinations: {df_agg['blank_id'].nunique()}")

# Riempi date mancanti per ogni blank_id
print("\n📅 Filling missing dates for each blank...")
date_range = pd.date_range(
    start=df_agg['date'].min(),
    end=df_agg['date'].max(),
    freq='D'
)

blank_ids = df_agg['blank_id'].unique()
full_data = []

for blank_id in blank_ids:
    # Prendi info blank
    blank_data = df_agg[df_agg['blank_id'] == blank_id].iloc[0]
    
    for date in date_range:
        existing = df_agg[(df_agg['blank_id'] == blank_id) & (df_agg['date'] == date)]
        
        if len(existing) > 0:
            full_data.append(existing.iloc[0].to_dict())
        else:
            # Giorno senza vendite = 0
            full_data.append({
                'date': date,
                'blank_type': blank_data['blank_type'],
                'size': blank_data['size'],
                'color': blank_data['color'],
                'blank_id': blank_id,
                'quantity': 0,
                'revenue': 0.0
            })

df_full = pd.DataFrame(full_data)
df_full = df_full.sort_values(['blank_id', 'date']).reset_index(drop=True)

print(f"✅ {len(df_full)} records after filling gaps")

# Feature engineering
print("\n📈 Computing features...")

# Rolling statistics (7, 14, 30 giorni)
for window in [7, 14, 30]:
    df_full[f'quantity_rolling_mean_{window}d'] = (
        df_full.groupby('blank_id')['quantity']
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    df_full[f'quantity_rolling_std_{window}d'] = (
        df_full.groupby('blank_id')['quantity']
        .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    )

# Lag features
for lag in [1, 7, 14, 30]:
    df_full[f'quantity_lag_{lag}d'] = (
        df_full.groupby('blank_id')['quantity'].shift(lag).fillna(0)
    )

# Date features
df_full['day_of_week'] = df_full['date'].dt.dayofweek
df_full['day_of_month'] = df_full['date'].dt.day
df_full['month'] = df_full['date'].dt.month
df_full['is_weekend'] = (df_full['day_of_week'] >= 5).astype(int)

# Target: prossime vendite (7, 14, 30 giorni)
print("\n🎯 Creating targets...")
for horizon in [7, 14, 30]:
    df_full[f'target_{horizon}d'] = (
        df_full.groupby('blank_id')['quantity']
        .transform(lambda x: x.shift(-horizon))
    )

# Rimuovi righe senza target
df_train = df_full.dropna(subset=['target_7d', 'target_14d', 'target_30d']).copy()

print(f"✅ {len(df_train)} records with targets")

# Split temporale
print("\n✂️  Splitting dataset (temporal split)...")
days_total = (df_train['date'].max() - df_train['date'].min()).days

train_end = df_train['date'].min() + timedelta(days=int(days_total * 0.7))
val_end = df_train['date'].min() + timedelta(days=int(days_total * 0.85))

train_df = df_train[df_train['date'] <= train_end]
val_df = df_train[(df_train['date'] > train_end) & (df_train['date'] <= val_end)]
test_df = df_train[df_train['date'] > val_end]

print(f"   Train: {len(train_df)} records ({train_df['date'].min()} → {train_df['date'].max()})")
print(f"   Validation: {len(val_df)} records ({val_df['date'].min()} → {val_df['date'].max()})")
print(f"   Test: {len(test_df)} records ({test_df['date'].min()} → {test_df['date'].max()})")

# Salva
print("\n💾 Saving datasets...")
train_df.to_csv('data/blanks_train.csv', index=False)
val_df.to_csv('data/blanks_validation.csv', index=False)
test_df.to_csv('data/blanks_test.csv', index=False)

# Metadata
metadata = {
    'created_at': datetime.now().isoformat(),
    'total_records': len(df_train),
    'train_records': len(train_df),
    'val_records': len(val_df),
    'test_records': len(test_df),
    'unique_blanks': int(df_train['blank_id'].nunique()),
    'date_range': {
        'start': str(df_train['date'].min()),
        'end': str(df_train['date'].max()),
        'days': days_total
    }
}

with open('data/blanks_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Blanks data preparation complete!")
print("=" * 60)
print(f"\nFiles saved:")
print(f"   📄 data/blanks_train.csv ({len(train_df):,} rows)")
print(f"   📄 data/blanks_validation.csv ({len(val_df):,} rows)")
print(f"   📄 data/blanks_test.csv ({len(test_df):,} rows)")
print(f"   📄 data/blanks_metadata.json")
print(f"\n📊 Dataset covers {days_total} days")
print(f"   {df_train['blank_id'].nunique()} unique blank combinations")
print(f"\nNext step: Run training with /train/blanks-lstm")
print("=" * 60)

