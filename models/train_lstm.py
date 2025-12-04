import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import json
from datetime import datetime
import os

print("=" * 50)
print("NFR ML - LSTM Training")
print("=" * 50)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

EPOCHS = args.epochs

# ==================== LOAD DATA ====================
print("\n📥 Loading training data...")

train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/validation.csv')

print(f"   Train: {len(train_df)} records")
print(f"   Validation: {len(val_df)} records")

# ==================== FEATURE SELECTION ====================
feature_cols = [
    'month', 'day_of_week', 'week_of_year', 'is_weekend',
    'sales_7d_avg', 'sales_14d_avg', 'sales_30d_avg', 'sales_7d_std',
    'trend_7d', 'is_month_start', 'is_month_end', 'quarter', 'avg_order_value'
]

# Usa i target corretti
target_cols = ['target_7d_avg', 'target_14d_avg', 'target_30d_avg']

X_train = train_df[feature_cols].values
y_train = train_df[target_cols].values

X_val = val_df[feature_cols].values
y_val = val_df[target_cols].values

print(f"\n   Features: {len(feature_cols)} columns")
print(f"   Targets: {target_cols}")

# ==================== NORMALIZATION ====================
print("\n🔧 Normalizing features...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)

# Save scalers
os.makedirs('models/artifacts', exist_ok=True)
joblib.dump(scaler_X, 'models/artifacts/scaler_X.pkl')
joblib.dump(scaler_y, 'models/artifacts/scaler_y.pkl')

print("   ✅ Scalers saved")

# ==================== MODEL ARCHITECTURE ====================
print("\n🏗️  Building LSTM model...")

# Reshape for LSTM (samples, timesteps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))

model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(1, len(feature_cols))),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3)  # 3 output: 7d, 14d, 30d avg
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(model.summary())

# ==================== GPU CHECK ====================
print("\n🎮 GPU Info:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"   ✅ {gpu}")
else:
    print("   ⚠️  No GPU detected, using CPU")

# ==================== TRAINING ====================
print(f"\n🚀 Training for {EPOCHS} epochs...")

history = model.fit(
    X_train_reshaped, y_train_scaled,
    validation_data=(X_val_reshaped, y_val_scaled),
    epochs=EPOCHS,
    batch_size=32,
    verbose=1
)

# ==================== SAVE MODEL ====================
print("\n💾 Saving model...")

model.save('models/artifacts/lstm_demand_forecast.h5')
print("   ✅ Model saved: models/artifacts/lstm_demand_forecast.h5")

# Save training history
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'mae': [float(x) for x in history.history['mae']],
    'val_mae': [float(x) for x in history.history['val_mae']],
    'epochs': EPOCHS,
    'trained_at': datetime.now().isoformat()
}

with open('models/artifacts/training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)

print("   ✅ Training history saved")

# ==================== EVALUATION ====================
print("\n📊 Evaluation on validation set:")
val_loss, val_mae = model.evaluate(X_val_reshaped, y_val_scaled, verbose=0)
print(f"   Validation Loss (MSE): {val_loss:.4f}")
print(f"   Validation MAE: {val_mae:.4f}")

print("\n✅ Training complete!")
print("=" * 50)
