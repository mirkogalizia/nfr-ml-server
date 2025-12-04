import requests
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import time
import os

print("=" * 70)
print("IMPORT VENDITE DA VECCHIO SHOPIFY STORE")
print("=" * 70)

# Configurazione Shopify da variabili d'ambiente
SHOP_URL = os.getenv("OLD_SHOPIFY_STORE_URL", "imjsqk-my.myshopify.com")
ACCESS_TOKEN = os.getenv("OLD_SHOPIFY_ACCESS_TOKEN")

if not ACCESS_TOKEN:
    print("❌ Errore: OLD_SHOPIFY_ACCESS_TOKEN non configurato")
    exit(1)

API_VERSION = "2024-10"

headers = {
    "X-Shopify-Access-Token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}

# Init Firebase
cred = credentials.Certificate("serviceAccountKey.json")
try:
    firebase_admin.initialize_app(cred)
except:
    pass

db = firestore.client()

print(f"\n📡 Connessione a: {SHOP_URL}")

# 1. Recupera ordini da Shopify
def get_all_orders():
    """Recupera tutti gli ordini dal vecchio store"""
    orders = []
    url = f"https://{SHOP_URL}/admin/api/{API_VERSION}/orders.json"
    params = {
        "status": "any",
        "limit": 250,
        "fields": "id,created_at,line_items,financial_status"
    }
    
    page = 1
    while url:
        print(f"   Caricamento pagina {page}...")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"❌ Errore API: {response.status_code}")
            print(response.text)
            break
        
        data = response.json()
        orders.extend(data.get("orders", []))
        
        # Pagination link
        link_header = response.headers.get("Link", "")
        if 'rel="next"' in link_header:
            next_link = [l for l in link_header.split(",") if 'rel="next"' in l]
            if next_link:
                url = next_link[0].split(";")[0].strip("<> ")
                params = None
            else:
                url = None
        else:
            url = None
        
        page += 1
        time.sleep(0.5)
    
    return orders

print("\n📦 Recupero ordini da Shopify...")
orders = get_all_orders()
print(f"✅ Recuperati {len(orders)} ordini")

# 2. Processa ordini e carica su Firebase
print("\n💾 Caricamento vendite su Firebase...")

total_items = 0
processed_orders = 0
batch = db.batch()
batch_count = 0

for order in orders:
    order_id = str(order["id"])
    created_at = order["created_at"]
    financial_status = order.get("financial_status", "")
    
    # Solo ordini pagati
    if financial_status not in ["paid", "partially_paid", "refunded"]:
        continue
    
    # Processa line items
    for item in order.get("line_items", []):
        variant_id = str(item.get("variant_id", ""))
        quantity = item.get("quantity", 0)
        price = float(item.get("price", 0))
        
        if not variant_id or quantity <= 0:
            continue
        
        # Crea documento su Firebase
        doc_id = f"old_{order_id}_{variant_id}"
        doc_ref = db.collection("ml_sales_data").document(doc_id)
        
        batch.set(doc_ref, {
            "order_id": order_id,
            "variant_id": variant_id,
            "date": created_at[:10],
            "quantity": quantity,
            "revenue": quantity * price,
            "source": "old_shopify_store",
            "imported_at": datetime.now().isoformat()
        })
        
        total_items += 1
        batch_count += 1
        
        # Commit batch ogni 450 documenti
        if batch_count >= 450:
            batch.commit()
            print(f"   Salvati {total_items} items...")
            batch = db.batch()
            batch_count = 0
    
    processed_orders += 1

# Commit ultimo batch
if batch_count > 0:
    batch.commit()

print(f"\n✅ Import completato!")
print(f"   • Ordini processati: {processed_orders}")
print(f"   • Line items salvati: {total_items}")

# 3. Statistiche periodo importato
print("\n" + "=" * 70)
print("📊 STATISTICHE DATI IMPORTATI")
print("=" * 70)

if orders:
    dates = [o["created_at"][:10] for o in orders]
    print(f"\nPeriodo: {min(dates)} → {max(dates)}")
    print(f"Ordini totali: {len(orders)}")
    print(f"Items venduti: {total_items}")

print("\n💡 Prossimi step:")
print("   1. Rilanciare /train/prepare-data")
print("   2. Rilanciare /train/lstm")
print("\n" + "=" * 70)
