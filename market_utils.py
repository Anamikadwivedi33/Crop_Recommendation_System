import json
import urllib.request
import urllib.error
import os

# Default fallback file
FALLBACK_FILE = "mandi_data.json"

# You can set your API Key here or in Environment Variables
# Get a free key from https://data.gov.in/
API_KEY = os.environ.get("MANDI_API_KEY", "YOUR_API_KEY_HERE")

def get_mandi_data(limit=20):
    """
    Fetches Mandi data from data.gov.in API.
    Falls back to 'mandi_data.json' if API fails.
    """
    # 1. Try fetching from Real API
    try:
        if API_KEY == "YOUR_API_KEY_HERE":
            raise ValueError("No API Key configured")

        url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit={limit}"
        
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            
        # Parse the specific format of data.gov.in
        # Usually records are under records key
        records = data.get("records", [])
        if not records:
            raise ValueError("No records found in API response")
            
        # Standardize keys to match our app
        # API returns keys like: state, district, market, commodity, min_price, max_price, modal_price
        formatted_data = []
        for r in records:
            formatted_data.append({
                "state": r.get("state", "Unknown"),
                "district": r.get("district", "Unknown"),
                "market": r.get("market", "Unknown"),
                "commodity": r.get("commodity", "Unknown"),
                "min_price": r.get("min_price", 0),
                "max_price": r.get("max_price", 0),
                "modal_price": r.get("modal_price", 0),
                "date": r.get("arrival_date", "")
            })
            
        print("✅ Fetched live data from API")
        return formatted_data

    except Exception as e:
        print(f"⚠️ API Fetch Failed: {e}")
        print("🔄 Switching to Fallback Data...")
        return load_fallback_data()

def load_fallback_data():
    """Loads data from the local JSON file."""
    try:
        with open(FALLBACK_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
