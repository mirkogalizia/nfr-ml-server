import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import json
from collections import defaultdict

print("=" * 70)
print("ANALISI VENDITE BLANKS (con mapping grafiche → blanks)")
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
        'blank_variant_id': data.get('blank_variant_id'),
        'size': data.get('size'),
        'color': data.get('color')
    }

print(f"✅ Mappati {len(mapping)} variant_id grafiche → blanks")

# 2. Carica vendite da ml_sales_data
print("\n📊 Caricamento vendite...")
sales_ref = db.collection("ml_sales_data").stream()

sales_data = []
unmapped_variants = set()

for doc in sales_ref:
    sale = doc.to_dict()
    variant_id = str(sale.get('variant_id'))
    quantity = sale.get('quantity', 0)
    date = sale.get('date')
    
    # Trova blank corrispondente
    if variant_id in mapping:
        blank_info = mapping[variant_id]
        
        sales_data.append({
            'date': date,
            'variant_id_grafica': variant_id,
            'blank_key': blank_info['blank_key'],
            'size': blank_info['size'],
            'color': blank_info['color'],
            'quantity': quantity,
            'revenue': sale.get('revenue', 0)
        })
    else:
        unmapped_variants.add(variant_id)

print(f"✅ {len(sales_data)} vendite con mapping blanks")
print(f"⚠️  {len(unmapped_variants)} variant_id senza mapping blanks")

# 3. Converti a DataFrame per analisi
df = pd.DataFrame(sales_data)

if len(df) == 0:
    print("\n❌ Nessuna vendita trovata con mapping blanks valido")
    exit(1)

df['date'] = pd.to_datetime(df['date'])

# 4. Aggregazione per blank_key (tipologia)
print("\n" + "=" * 70)
print("📦 VENDITE PER TIPOLOGIA BLANK")
print("=" * 70)

by_blank = df.groupby('blank_key').agg({
    'quantity': 'sum',
    'revenue': 'sum',
    'date': ['min', 'max']
}).reset_index()

by_blank.columns = ['blank_key', 'total_quantity', 'total_revenue', 'first_sale', 'last_sale']
by_blank['days_active'] = (by_blank['last_sale'] - by_blank['first_sale']).dt.days + 1
by_blank['avg_daily_sales'] = by_blank['total_quantity'] / by_blank['days_active']
by_blank = by_blank.sort_values('total_quantity', ascending=False)

for _, row in by_blank.iterrows():
    print(f"\n{row['blank_key']}")
    print(f"  • Pezzi totali: {row['total_quantity']:.0f}")
    print(f"  • Media: {row['avg_daily_sales']:.2f} pz/giorno")
    print(f"  • Revenue: €{row['total_revenue']:.2f}")

# 5. Aggregazione per taglia
print("\n" + "=" * 70)
print("📏 VENDITE PER TAGLIA")
print("=" * 70)

by_size = df.groupby('size').agg({
    'quantity': 'sum',
    'revenue': 'sum'
}).reset_index().sort_values('quantity', ascending=False)

total_pieces = by_size['quantity'].sum()
for _, row in by_size.iterrows():
    pct = (row['quantity'] / total_pieces) * 100
    print(f"\n{row['size']}")
    print(f"  • Pezzi: {row['quantity']:.0f} ({pct:.1f}%)")
    print(f"  • Revenue: €{row['revenue']:.2f}")

# 6. Aggregazione per colore
print("\n" + "=" * 70)
print("🎨 VENDITE PER COLORE")
print("=" * 70)

by_color = df.groupby('color').agg({
    'quantity': 'sum',
    'revenue': 'sum'
}).reset_index().sort_values('quantity', ascending=False)

for _, row in by_color.iterrows():
    pct = (row['quantity'] / total_pieces) * 100
    print(f"\n{row['color']}")
    print(f"  • Pezzi: {row['quantity']:.0f} ({pct:.1f}%)")
    print(f"  • Revenue: €{row['revenue']:.2f}")

# 7. Aggregazione per blank + taglia + colore (combinazione completa)
print("\n" + "=" * 70)
print("🎯 TOP 20 COMBINAZIONI BLANK + TAGLIA + COLORE")
print("=" * 70)

by_combo = df.groupby(['blank_key', 'size', 'color']).agg({
    'quantity': 'sum',
    'revenue': 'sum',
    'date': ['min', 'max']
}).reset_index()

by_combo.columns = ['blank_key', 'size', 'color', 'total_quantity', 'total_revenue', 'first_sale', 'last_sale']
by_combo['days_active'] = (by_combo['last_sale'] - by_combo['first_sale']).dt.days + 1
by_combo['avg_daily_sales'] = by_combo['total_quantity'] / by_combo['days_active']
by_combo = by_combo.sort_values('total_quantity', ascending=False)

top_20_combo = by_combo.head(20)
for _, row in top_20_combo.iterrows():
    print(f"\n{row['blank_key']} - {row['size']} - {row['color']}")
    print(f"  • Pezzi totali: {row['total_quantity']:.0f}")
    print(f"  • Media: {row['avg_daily_sales']:.2f} pz/giorno")
    print(f"  • Revenue: €{row['total_revenue']:.2f}")

# 8. Salva output JSON
output = {
    "period": {
        "start": df['date'].min().isoformat(),
        "end": df['date'].max().isoformat(),
        "days": (df['date'].max() - df['date'].min()).days + 1
    },
    "totals": {
        "pieces_sold": float(df['quantity'].sum()),
        "revenue": float(df['revenue'].sum()),
        "mapped_variants": len(mapping),
        "unmapped_variants": len(unmapped_variants)
    },
    "by_blank_type": [
        {
            "blank_key": row['blank_key'],
            "total_quantity": float(row['total_quantity']),
            "avg_daily_sales": float(row['avg_daily_sales']),
            "revenue": float(row['total_revenue'])
        }
        for _, row in by_blank.iterrows()
    ],
    "by_size": [
        {
            "size": row['size'],
            "total_quantity": float(row['quantity']),
            "percentage": float((row['quantity'] / total_pieces) * 100),
            "revenue": float(row['revenue'])
        }
        for _, row in by_size.iterrows()
    ],
    "by_color": [
        {
            "color": row['color'],
            "total_quantity": float(row['quantity']),
            "percentage": float((row['quantity'] / total_pieces) * 100),
            "revenue": float(row['revenue'])
        }
        for _, row in by_color.iterrows()
    ],
    "top_20_combinations": [
        {
            "blank_key": row['blank_key'],
            "size": row['size'],
            "color": row['color'],
            "total_quantity": float(row['total_quantity']),
            "avg_daily_sales": float(row['avg_daily_sales']),
            "revenue": float(row['total_revenue'])
        }
        for _, row in top_20_combo.iterrows()
    ]
}

with open('data/blanks_sales_detailed.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Report salvato in: data/blanks_sales_detailed.json")
print("\n" + "=" * 70)
