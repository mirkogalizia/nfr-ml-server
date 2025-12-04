import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

print("=" * 60)
print("TEST: Lettura dati da Firebase ml_sales_data")
print("=" * 60)

# Init Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

print("\n🔥 Fetching data from Firestore...")
sales_ref = db.collection("ml_sales_data").stream()

data = []
for doc in sales_ref:
    data.append(doc.to_dict())

df = pd.DataFrame(data)

print(f"\n✅ Loaded {len(df)} records from Firebase")
print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
print(f"   Unique variants: {df['variant_id'].nunique()}")
print(f"   Total quantity sold: {df['quantity'].sum()}")

# Mostra primi 10 record
print("\n📋 First 10 records:")
print(df.head(10)[['date', 'variant_id', 'quantity', 'revenue', 'product_title']])

# Record con vendite > 0
sales = df[df['quantity'] > 0]
print(f"\n💰 Records with sales > 0: {len(sales)}")
print("\n📋 First 10 records WITH sales:")
print(sales.head(10)[['date', 'variant_id', 'quantity', 'revenue', 'product_title']])

# Verifica variant specifico
test_variant = "56148396081535"
variant_data = df[df['variant_id'] == test_variant]
print(f"\n🔍 Checking variant {test_variant}:")
if len(variant_data) > 0:
    print(f"   ✅ Found {len(variant_data)} records for this variant")
    print(variant_data[['date', 'quantity', 'revenue']])
else:
    print(f"   ❌ Variant {test_variant} NOT found in Firebase data")

# Top 10 variant per vendite
print("\n🏆 Top 10 variants by total quantity sold:")
top_variants = df.groupby('variant_id')['quantity'].sum().sort_values(ascending=False).head(10)
for vid, qty in top_variants.items():
    print(f"   {vid}: {qty} units")

print("\n" + "=" * 60)
