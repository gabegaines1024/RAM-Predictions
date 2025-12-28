import pandas as pd
from bs4 import BeautifulSoup as soup
import requests as rq
import re
from datetime import date


def scrape_newegg_ram():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    requests_ddr4 = rq.get("https://www.newegg.com/p/pl?N=100007611%20600561665", headers=headers)
    requests_ddr5 = rq.get("https://www.newegg.com/p/pl?N=100007611%20601410157", headers=headers)
    soup_ddr4 = soup(requests_ddr4.text, "html.parser")
    soup_ddr5 = soup(requests_ddr5.text, "html.parser")
    return soup_ddr4, soup_ddr5


# --- Extraction Helpers ---

def extract_model(title: str) -> str:
    """Extract model name (second word onward until specs start)."""
    # Typically: "Corsair Vengeance RGB Pro 16GB DDR4 3200MHz..."
    # We want "Vengeance RGB Pro" - everything after brand, before capacity/specs
    parts = title.split()
    if len(parts) < 2:
        return ""
    
    model_parts = []
    for part in parts[1:]:  # Skip brand (first word)
        # Stop when we hit capacity (e.g., "16GB") or DDR spec
        if re.match(r'^\d+GB', part, re.IGNORECASE) or re.match(r'^DDR\d', part, re.IGNORECASE):
            break
        model_parts.append(part)
    
    return " ".join(model_parts) if model_parts else parts[1]


def extract_capacity(title: str) -> int | None:
    """Extract capacity in GB from title (e.g., '16GB' -> 16)."""
    # Match patterns like "16GB", "32 GB", "2x8GB", "2 x 16GB"
    match = re.search(r'(\d+)\s*x\s*(\d+)\s*GB', title, re.IGNORECASE)
    if match:
        # Kit format: 2x8GB = 16GB total
        return int(match.group(1)) * int(match.group(2))
    
    match = re.search(r'(\d+)\s*GB', title, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def extract_frequency(title: str) -> int | None:
    """Extract frequency in MHz from title (e.g., 'DDR4 3200' -> 3200)."""
    # Common patterns: "3200MHz", "DDR4-3200", "PC4-25600" (need to convert)
    match = re.search(r'(\d{4,5})\s*MHz', title, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # DDR4-3200 or DDR5-6000 format
    match = re.search(r'DDR\d[- ](\d{4,5})', title, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # PC4-25600 format (need to divide by 8 for MHz: 25600/8 = 3200)
    match = re.search(r'PC\d[- ](\d{5})', title, re.IGNORECASE)
    if match:
        return int(match.group(1)) // 8
    
    return None


def extract_cas(item) -> int | None:
    """Extract CAS latency from product specs or title."""
    text = item.get_text()
    
    # Look for "CL16", "CAS 16", "C16", etc.
    match = re.search(r'(?:CAS|CL|C)[\s-]*(\d{1,2})', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Sometimes in timings: first number is CAS (16-18-18-36)
    match = re.search(r'\b(\d{1,2})-\d{1,2}-\d{1,2}-\d{1,2}\b', text)
    if match:
        return int(match.group(1))
    
    return None


def extract_timings(item) -> str | None:
    """Extract full timing string (e.g., '16-18-18-36')."""
    text = item.get_text()
    
    # Standard timing format: CL-tRCD-tRP-tRAS (e.g., 16-18-18-36)
    match = re.search(r'\b(\d{1,2}-\d{1,2}-\d{1,2}-\d{1,3})\b', text)
    if match:
        return match.group(1)
    
    return None


def extract_voltage(item) -> float | None:
    """Extract voltage (e.g., 1.35V -> 1.35)."""
    text = item.get_text()
    
    # Match 1.2V, 1.35V, 1.45 V, etc.
    match = re.search(r'(\d\.\d{1,2})\s*V', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return None


def extract_price(item) -> float | None:
    """Extract price in USD."""
    # Newegg price structure: <li class="price-current">$<strong>89</strong><sup>.99</sup></li>
    price_elem = item.select_one(".price-current")
    if price_elem:
        # Get all text, remove $ and commas
        price_text = price_elem.get_text().strip()
        match = re.search(r'[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', price_text)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Fallback: search for price pattern in item text
    text = item.get_text()
    match = re.search(r'\$(\d{1,3}(?:,\d{3})*\.\d{2})', text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    return None


def extract_review_count(item) -> int | None:
    """Extract number of reviews."""
    # Newegg format: "(1,247)" or "1247 Reviews"
    rating_elem = item.select_one(".item-rating-num")
    if rating_elem:
        text = rating_elem.get_text()
        match = re.search(r'\(?([\d,]+)\)?', text)
        if match:
            return int(match.group(1).replace(',', ''))
    
    # Fallback
    text = item.get_text()
    match = re.search(r'\(([\d,]+)\)\s*(?:reviews?)?', text, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(',', ''))
    
    return None


def extract_rating(item) -> float | None:
    """Extract average rating (out of 5)."""
    # Newegg uses egg icons, rating often in aria-label or title attribute
    rating_elem = item.select_one(".item-rating")
    if rating_elem:
        # Check for aria-label like "Rating: 4.8 out of 5"
        aria = rating_elem.get('aria-label', '') or rating_elem.get('title', '')
        match = re.search(r'(\d(?:\.\d)?)\s*(?:out of|\/)\s*5', aria, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Count filled eggs (each egg = 1 point, typically 5 eggs max)
        filled = len(rating_elem.select(".rating-egg-full, [class*='rating-5'], [class*='rating-4']"))
        if filled > 0:
            return float(filled)
    
    return None


# --- Main Functions ---

def parse_product(item, ddr_gen: str) -> dict | None:
    """Extract data from a single product container."""
    try:
        title_elem = item.select_one(".item-title")
        if not title_elem:
            return None
            
        title = title_elem.text.strip()
        brand = title.split()[0] if title else None

        return {
            'brand': brand,
            'model': extract_model(title),
            'ddr_generation': ddr_gen,
            'capacity_gb': extract_capacity(title),
            'frequency_mhz': extract_frequency(title),
            'cas_latency': extract_cas(item),
            'timings': extract_timings(item),
            'voltage': extract_voltage(item),
            'price_usd': extract_price(item),
            'in_stock': "OUT OF STOCK" not in item.text.upper(),
            'num_reviews': extract_review_count(item),
            'avg_rating': extract_rating(item),
            'date_scraped': date.today().isoformat()
        }
    except Exception:
        return None


def scrape_all_products() -> list[dict]:
    """Scrape all RAM products from Newegg."""
    soup_ddr4, soup_ddr5 = scrape_newegg_ram()
    
    # Find product containers
    items_ddr4 = soup_ddr4.select(".item-cell")
    items_ddr5 = soup_ddr5.select(".item-cell")
    
    # List comprehension with helper function
    products = [
        p for item in items_ddr4 
        if (p := parse_product(item, "DDR4"))
    ] + [
        p for item in items_ddr5 
        if (p := parse_product(item, "DDR5"))
    ]
    
    return products


def save_to_csv(filename: str = "ram_data.csv"):
    """Scrape products and save to CSV."""
    products = scrape_all_products()
    df = pd.DataFrame(products)
    df.to_csv(filename, index=False)
    print(f"Saved {len(products)} products to {filename}")
    return df

save_to_csv()