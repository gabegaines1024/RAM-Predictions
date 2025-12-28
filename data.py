import pandas as pd
from bs4 import BeautifulSoup as soup
import requests as rq
import re
from datetime import date


def scrape_ram():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Newegg
    requests_ddr4_newegg = rq.get("https://www.newegg.com/p/pl?N=100007611%20600561665", headers=headers)
    requests_ddr5_newegg = rq.get("https://www.newegg.com/p/pl?N=100007611%20601410157", headers=headers)
    
    # Best Buy (fixed missing https:// on DDR4)
    requests_ddr4_bestbuy = rq.get("https://www.bestbuy.com/site/searchpage.jsp?_dyncharset=UTF-8&browsedCategory=abcat0506000&id=pcat17071&iht=n&ks=960&list=y&qp=typeofmemoryram_facet%3DMemory%20Type~DDR4&sc=Global&st=categoryid%24abcat0506000&type=page&usc=All%20Categories", headers=headers)
    requests_ddr5_bestbuy = rq.get("https://www.bestbuy.com/site/searchpage.jsp?_dyncharset=UTF-8&browsedCategory=abcat0506000&id=pcat17071&iht=n&ks=960&list=y&qp=typeofmemoryram_facet%3DMemory%20Type~DDR5&sc=Global&st=categoryid%24abcat0506000&type=page&usc=All%20Categories", headers=headers)

    soup_ddr4_newegg = soup(requests_ddr4_newegg.text, "html.parser")
    soup_ddr5_newegg = soup(requests_ddr5_newegg.text, "html.parser")
    soup_ddr4_bestbuy = soup(requests_ddr4_bestbuy.text, "html.parser")
    soup_ddr5_bestbuy = soup(requests_ddr5_bestbuy.text, "html.parser")
    
    return soup_ddr4_newegg, soup_ddr5_newegg, soup_ddr4_bestbuy, soup_ddr5_bestbuy


# --- Shared Extraction Helpers ---

def extract_model(title: str) -> str:
    """Extract model name (second word onward until specs start)."""
    parts = title.split()
    if len(parts) < 2:
        return ""
    
    model_parts = []
    for part in parts[1:]:
        if re.match(r'^\d+GB', part, re.IGNORECASE) or re.match(r'^DDR\d', part, re.IGNORECASE):
            break
        model_parts.append(part)
    
    return " ".join(model_parts) if model_parts else parts[1]


def extract_capacity(title: str) -> int | None:
    """Extract capacity in GB from title (e.g., '16GB' -> 16)."""
    # Kit format: 2x8GB = 16GB total
    match = re.search(r'(\d+)\s*x\s*(\d+)\s*GB', title, re.IGNORECASE)
    if match:
        return int(match.group(1)) * int(match.group(2))
    
    match = re.search(r'(\d+)\s*GB', title, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def extract_frequency(title: str) -> int | None:
    """Extract frequency in MHz from title."""
    match = re.search(r'(\d{4,5})\s*MHz', title, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    match = re.search(r'DDR\d[- ](\d{4,5})', title, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    match = re.search(r'PC\d[- ](\d{5})', title, re.IGNORECASE)
    if match:
        return int(match.group(1)) // 8
    
    return None


def extract_cas_from_text(text: str) -> int | None:
    """Extract CAS latency from text."""
    match = re.search(r'(?:CAS|CL|C)[\s-]*(\d{1,2})', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    match = re.search(r'\b(\d{1,2})-\d{1,2}-\d{1,2}-\d{1,2}\b', text)
    if match:
        return int(match.group(1))
    
    return None


def extract_timings_from_text(text: str) -> str | None:
    """Extract full timing string (e.g., '16-18-18-36')."""
    match = re.search(r'\b(\d{1,2}-\d{1,2}-\d{1,2}-\d{1,3})\b', text)
    if match:
        return match.group(1)
    return None


def extract_voltage_from_text(text: str) -> float | None:
    """Extract voltage (e.g., 1.35V -> 1.35)."""
    match = re.search(r'(\d\.\d{1,2})\s*V', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


# --- Newegg-Specific Helpers ---

def extract_price_newegg(item) -> float | None:
    """Extract price from Newegg product."""
    price_elem = item.select_one(".price-current")
    if price_elem:
        price_text = price_elem.get_text().strip()
        match = re.search(r'[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', price_text)
        if match:
            return float(match.group(1).replace(',', ''))
    
    text = item.get_text()
    match = re.search(r'\$(\d{1,3}(?:,\d{3})*\.\d{2})', text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    return None


def extract_review_count_newegg(item) -> int | None:
    """Extract number of reviews from Newegg."""
    rating_elem = item.select_one(".item-rating-num")
    if rating_elem:
        text = rating_elem.get_text()
        match = re.search(r'\(?([\d,]+)\)?', text)
        if match:
            return int(match.group(1).replace(',', ''))
    
    text = item.get_text()
    match = re.search(r'\(([\d,]+)\)\s*(?:reviews?)?', text, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(',', ''))
    
    return None


def extract_rating_newegg(item) -> float | None:
    """Extract average rating from Newegg."""
    rating_elem = item.select_one(".item-rating")
    if rating_elem:
        aria = rating_elem.get('aria-label', '') or rating_elem.get('title', '')
        match = re.search(r'(\d(?:\.\d)?)\s*(?:out of|\/)\s*5', aria, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        filled = len(rating_elem.select(".rating-egg-full, [class*='rating-5'], [class*='rating-4']"))
        if filled > 0:
            return float(filled)
    
    return None


# --- Best Buy-Specific Helpers ---

def extract_price_bestbuy(item) -> float | None:
    """Extract price from Best Buy product."""
    # Best Buy price selectors
    price_elem = item.select_one(".priceView-customer-price span, .priceView-hero-price span")
    if price_elem:
        price_text = price_elem.get_text().strip()
        match = re.search(r'\$?([\d,]+\.?\d*)', price_text)
        if match:
            return float(match.group(1).replace(',', ''))
    
    # Fallback: look for price in data attributes
    price_attr = item.get('data-price') or item.select_one('[data-price]')
    if price_attr:
        if isinstance(price_attr, str):
            return float(price_attr)
        else:
            return float(price_attr.get('data-price', 0))
    
    # Last fallback: regex on text
    text = item.get_text()
    match = re.search(r'\$(\d{1,3}(?:,\d{3})*\.\d{2})', text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    return None


def extract_review_count_bestbuy(item) -> int | None:
    """Extract number of reviews from Best Buy."""
    # Best Buy uses .c-reviews or .c-ratings-reviews
    review_elem = item.select_one(".c-reviews, .c-ratings-reviews-count")
    if review_elem:
        text = review_elem.get_text()
        match = re.search(r'\(?([\d,]+)\)?(?:\s*reviews?)?', text, re.IGNORECASE)
        if match:
            return int(match.group(1).replace(',', ''))
    
    # Look for "Reviews" text
    text = item.get_text()
    match = re.search(r'([\d,]+)\s*(?:Reviews?|Ratings?)', text, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(',', ''))
    
    return None


def extract_rating_bestbuy(item) -> float | None:
    """Extract average rating from Best Buy."""
    # Best Buy stores rating in various ways
    rating_elem = item.select_one(".c-ratings-reviews, [class*='rating']")
    if rating_elem:
        aria = rating_elem.get('aria-label', '') or rating_elem.get('title', '')
        match = re.search(r'(\d(?:\.\d)?)\s*(?:out of|\/)\s*5', aria, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Sometimes just the number
        match = re.search(r'Rating[:\s]*(\d(?:\.\d)?)', aria, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # Check for data attribute
    rating_attr = item.select_one('[data-rating], [data-average-rating]')
    if rating_attr:
        rating = rating_attr.get('data-rating') or rating_attr.get('data-average-rating')
        if rating:
            return float(rating)
    
    return None


# --- Product Parsers ---

def parse_newegg_product(item, ddr_gen: str) -> dict | None:
    """Extract data from a Newegg product container."""
    try:
        title_elem = item.select_one(".item-title")
        if not title_elem:
            return None
            
        title = title_elem.text.strip()
        text = item.get_text()
        brand = title.split()[0] if title else None

        return {
            'source': 'newegg',
            'brand': brand,
            'model': extract_model(title),
            'ddr_generation': ddr_gen,
            'capacity_gb': extract_capacity(title),
            'frequency_mhz': extract_frequency(title),
            'cas_latency': extract_cas_from_text(text),
            'timings': extract_timings_from_text(text),
            'voltage': extract_voltage_from_text(text),
            'price_usd': extract_price_newegg(item),
            'in_stock': "OUT OF STOCK" not in text.upper(),
            'num_reviews': extract_review_count_newegg(item),
            'avg_rating': extract_rating_newegg(item),
            'date_scraped': date.today().isoformat()
        }
    except Exception:
        return None


def parse_bestbuy_product(item, ddr_gen: str) -> dict | None:
    """Extract data from a Best Buy product container."""
    try:
        # Best Buy title selectors
        title_elem = item.select_one(".sku-title a, .sku-header a, h4.sku-title")
        if not title_elem:
            # Fallback: any heading or link that looks like a title
            title_elem = item.select_one("h4 a, .product-title a")
        
        if not title_elem:
            return None
            
        title = title_elem.text.strip()
        text = item.get_text()
        brand = title.split()[0] if title else None

        # Check stock status
        in_stock = True
        if any(x in text.upper() for x in ["SOLD OUT", "OUT OF STOCK", "COMING SOON", "NOT AVAILABLE"]):
            in_stock = False

        return {
            'source': 'bestbuy',
            'brand': brand,
            'model': extract_model(title),
            'ddr_generation': ddr_gen,
            'capacity_gb': extract_capacity(title),
            'frequency_mhz': extract_frequency(title),
            'cas_latency': extract_cas_from_text(text),
            'timings': extract_timings_from_text(text),
            'voltage': extract_voltage_from_text(text),
            'price_usd': extract_price_bestbuy(item),
            'in_stock': in_stock,
            'num_reviews': extract_review_count_bestbuy(item),
            'avg_rating': extract_rating_bestbuy(item),
            'date_scraped': date.today().isoformat()
        }
    except Exception:
        return None


# --- Main Functions ---

def scrape_all_products() -> list[dict]:
    """Scrape all RAM products from Newegg and Best Buy."""
    soup_ddr4_newegg, soup_ddr5_newegg, soup_ddr4_bestbuy, soup_ddr5_bestbuy = scrape_ram()
    
    # Newegg product containers
    items_ddr4_newegg = soup_ddr4_newegg.select(".item-cell")
    items_ddr5_newegg = soup_ddr5_newegg.select(".item-cell")
    
    # Best Buy product containers (different selectors)
    items_ddr4_bestbuy = soup_ddr4_bestbuy.select(".sku-item, .list-item, [data-sku-id]")
    items_ddr5_bestbuy = soup_ddr5_bestbuy.select(".sku-item, .list-item, [data-sku-id]")
    
    products = [
        p for item in items_ddr4_newegg 
        if (p := parse_newegg_product(item, "DDR4"))
    ] + [
        p for item in items_ddr5_newegg 
        if (p := parse_newegg_product(item, "DDR5"))
    ] + [
        p for item in items_ddr4_bestbuy
        if (p := parse_bestbuy_product(item, "DDR4"))
    ] + [
        p for item in items_ddr5_bestbuy
        if (p := parse_bestbuy_product(item, "DDR5"))
    ]
    
    return products


def save_to_csv(filename: str = "ram_data.csv"):
    """Scrape products and save to CSV."""
    products = scrape_all_products()
    df = pd.DataFrame(products)
    df.to_csv(filename, index=False)
    print(f"Saved {len(products)} products to {filename}")
    return df


if __name__ == "__main__":
    save_to_csv()
