import requests
import json
import os

print("=" * 60)
print("NFR ML - Generate Blanks Mapping")
print("=" * 60)

# Configurazione Shopify
SHOP_URL = os.getenv("SHOPIFY_STORE_URL", "nerofuturarestock.myshopify.com")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")

if not ACCESS_TOKEN:
    print("❌ Errore: SHOPIFY_ACCESS_TOKEN non configurato")
    exit(1)

API_VERSION = "2024-10"

headers = {
    "X-Shopify-Access-Token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}

print(f"\n📡 Connessione a: {SHOP_URL}")

# Recupera tutti i prodotti
def get_all_products():
    """Recupera tutti i prodotti con le loro varianti"""
    products = []
    url = f"https://{SHOP_URL}/admin/api/{API_VERSION}/products.json"
    params = {"limit": 250}
    
    page = 1
    while url:
        print(f"   Caricamento prodotti pagina {page}...")
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"❌ Errore API: {response.status_code}")
            print(response.text)
            break
        
        data = response.json()
        products.extend(data.get("products", []))
        
        # Pagination
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
    
    return products

print("\n📦 Recupero prodotti da Shopify...")
products = get_all_products()
print(f"✅ Recuperati {len(products)} prodotti")

# Mappa varianti a blanks
print("\n🗺️  Mappando varianti a blanks...")

blanks_mapping = {}
total_variants = 0

# Mapping dei nomi blanks
blank_key_mapping = {
    'tshirt': 'tshirt',
    't-shirt': 'tshirt',
    'felpa': 'felpa_cappuccio',
    'felpa cappuccio': 'felpa_cappuccio',
    'hoodie': 'felpa_cappuccio',
    'zip hoodie': 'zip_hoodie',
    'zip': 'zip_hoodie',
    'crewneck': 'crewneck',
    'felpa girocollo': 'crewneck',
    'sweatpants': 'sweatpants',
    'pantaloni': 'sweatpants'
}

for product in products:
    product_title = product.get("title", "").lower()
    
    # Determina il tipo di blank dal titolo del prodotto
    blank_type = "unknown"
    for key, value in blank_key_mapping.items():
        if key in product_title:
            blank_type = value
            break
    
    # Se non trovato, prova a dedurlo dalle opzioni
    if blank_type == "unknown":
        if "tshirt" in product_title or "maglietta" in product_title:
            blank_type = "tshirt"
        elif "felpa" in product_title:
            blank_type = "felpa_cappuccio"
    
    # Processa ogni variante
    for variant in product.get("variants", []):
        variant_id = str(variant["id"])
        
        # Estrai taglia e colore dalle opzioni
        size = "unknown"
        color = "unknown"
        
        # Shopify usa option1, option2, option3
        option1 = variant.get("option1", "").lower() if variant.get("option1") else ""
        option2 = variant.get("option2", "").lower() if variant.get("option2") else ""
        option3 = variant.get("option3", "").lower() if variant.get("option3") else ""
        
        # Identifica taglia (cerca tra le opzioni)
        for opt in [option1, option2, option3]:
            if opt in ['xs', 's', 'm', 'l', 'xl', 'xxl']:
                size = opt.upper()
                break
        
        # Identifica colore (prende l'opzione che non è la taglia)
        for opt in [option1, option2, option3]:
            if opt and opt not in ['xs', 's', 'm', 'l', 'xl', 'xxl', '']:
                color = opt
                break
        
        # Salva mapping
        blanks_mapping[variant_id] = {
            "blank_key": blank_type,
            "size": size,
            "color": color,
            "product_title": product.get("title", ""),
            "variant_title": variant.get("title", "")
        }
        
        total_variants += 1

print(f"✅ Mappate {total_variants} varianti")

# Statistiche
blank_types = {}
for v in blanks_mapping.values():
    bt = v['blank_key']
    blank_types[bt] = blank_types.get(bt, 0) + 1

print("\n📊 Distribuzione per tipo:")
for bt, count in sorted(blank_types.items(), key=lambda x: x[1], reverse=True):
    print(f"   {bt}: {count} varianti")

# Salva mapping
print("\n💾 Salvando mapping...")
os.makedirs('data', exist_ok=True)

with open('data/blanks_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(blanks_mapping, f, indent=2, ensure_ascii=False)

print(f"✅ Mapping salvato: data/blanks_mapping.json")
print(f"   {len(blanks_mapping)} varianti mappate")

print("\n" + "=" * 60)
print("✅ Mapping generation complete!")
print("=" * 60)

