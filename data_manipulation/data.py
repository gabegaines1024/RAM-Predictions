import pandas as pd
from bs4 import BeautifulSoup as soup
import requests as rq
import cloudscraper
import re
from datetime import date, datetime, timedelta
import time
from pathlib import Path


# --- Scraping Functions ---

def get_session():
    """Create a requests session with proper headers."""
    session = rq.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    return session


def scrape_newegg():
    """Scrape Newegg using simple requests (works without JS)."""
    session = get_session()
    
    requests_ddr4 = session.get("https://www.newegg.com/p/pl?N=100007611%20600561665", timeout=15)
    requests_ddr5 = session.get("https://www.newegg.com/p/pl?N=100007611%20601410157", timeout=15)
    
    soup_ddr4 = soup(requests_ddr4.text, "html.parser")
    soup_ddr5 = soup(requests_ddr5.text, "html.parser")
    
    return soup_ddr4, soup_ddr5


def scrape_bhphoto():
    """Scrape B&H Photo using cloudscraper (bypasses Cloudflare)."""
    scraper = cloudscraper.create_scraper()
    products = []
    
    # B&H Photo RAM categories
    urls = {
        "DDR4": "https://www.bhphotovideo.com/c/search?q=ddr4%20desktop%20memory&sts=ma",
        "DDR5": "https://www.bhphotovideo.com/c/search?q=ddr5%20desktop%20memory&sts=ma",
    }
    
    for ddr_type, url in urls.items():
        try:
            time.sleep(1)  # Be polite
            response = scraper.get(url, timeout=30)
            
            if response.status_code == 200:
                page = soup(response.text, "html.parser")
                
                # B&H uses data-selenium attributes for product containers
                items = page.select('[data-selenium="miniProductPage"]')
                
                for item in items:
                    product = parse_bhphoto_product(item, ddr_type)
                    if product:
                        products.append(product)
                        
        except Exception as e:
            print(f"  B&H Photo failed for {ddr_type}: {e}")
    
    return products


def parse_bhphoto_product(item, ddr_gen: str) -> dict | None:
    """Parse a B&H Photo product container."""
    try:
        # Title - prefer h3 or description element, not empty links
        title_elem = item.select_one('h3, [data-selenium="miniProductPageDescription"], [class*="name"]')
        
        if not title_elem:
            return None
        
        title = title_elem.get_text(strip=True)
        # Clean up B&H's concatenated BH#/MFR# suffix
        title = re.sub(r'BH\s*#.*$', '', title).strip()
        
        if not title or len(title) < 5:
            return None
            
        # Skip if not RAM (B&H search might include accessories)
        if not any(x in title.upper() for x in ['DDR4', 'DDR5', 'MEMORY', 'RAM', 'GB']):
            return None
        
        brand = title.split()[0] if title else None
        
        # Price - B&H shows price in cents without decimal (e.g., $19199 = $191.99)
        price = None
        price_elem = item.select_one('[class*="price"], [class*="Price"]')
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            match = re.search(r'\$?([\d,]+)', price_text)
            if match:
                raw_price = int(match.group(1).replace(',', ''))
                # B&H stores price in cents if > 999 and no decimal in text
                if raw_price > 999 and '.' not in price_text:
                    price = raw_price / 100
                else:
                    price = float(raw_price)
        
        # Rating
        rating = None
        rating_elem = item.select_one('[data-selenium="ratingStars"], .rating, [class*="rating"]')
        if rating_elem:
            # B&H often uses aria-label for rating
            aria = rating_elem.get('aria-label', '')
            match = re.search(r'(\d(?:\.\d)?)\s*(?:out of|stars|/)\s*5?', aria, re.IGNORECASE)
            if match:
                rating = float(match.group(1))
            else:
                # Count filled stars
                stars = len(rating_elem.select('[class*="filled"], [class*="full"]'))
                if stars > 0:
                    rating = float(stars)
        
        # Reviews
        reviews = None
        review_elem = item.select_one('[data-selenium="reviewCount"], [class*="review"]')
        if review_elem:
            review_text = review_elem.get_text(strip=True)
            match = re.search(r'\(?([\d,]+)\)?', review_text)
            if match:
                reviews = int(match.group(1).replace(',', ''))
        
        # Stock status
        text = item.get_text()
        in_stock = not any(x in text.upper() for x in ["OUT OF STOCK", "SOLD OUT", "BACK-ORDER", "UNAVAILABLE"])
        
        return {
            'source': 'bhphoto',
            'brand': brand,
            'model': extract_model(title),
            'ddr_generation': ddr_gen,
            'capacity_gb': extract_capacity(title),
            'frequency_mhz': extract_frequency(title),
            'cas_latency': extract_cas_from_text(title),
            'timings': extract_timings_from_text(title),
            'voltage': extract_voltage_from_text(title),
            'price_usd': price,
            'in_stock': in_stock,
            'num_reviews': reviews,
            'avg_rating': rating,
            'date_scraped': date.today().isoformat()
        }
    except Exception:
        return None


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
    if not title:
        return None
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
    if not title:
        return None
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
    if not text:
        return None
    match = re.search(r'(?:CAS|CL|C)[\s-]*(\d{1,2})', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    match = re.search(r'\b(\d{1,2})-\d{1,2}-\d{1,2}-\d{1,2}\b', text)
    if match:
        return int(match.group(1))
    
    return None


def extract_timings_from_text(text: str) -> str | None:
    """Extract full timing string (e.g., '16-18-18-36')."""
    if not text:
        return None
    match = re.search(r'\b(\d{1,2}-\d{1,2}-\d{1,2}-\d{1,3})\b', text)
    if match:
        return match.group(1)
    return None


def extract_voltage_from_text(text: str) -> float | None:
    """Extract voltage (e.g., 1.35V -> 1.35)."""
    if not text:
        return None
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


# --- Main Functions ---

def scrape_all_products() -> list[dict]:
    """Scrape all RAM products from multiple sources."""
    products = []
    
    # Scrape Newegg (reliable, works with requests)
    print("Scraping Newegg...")
    try:
        soup_ddr4_newegg, soup_ddr5_newegg = scrape_newegg()
        
        items_ddr4_newegg = soup_ddr4_newegg.select(".item-cell")
        items_ddr5_newegg = soup_ddr5_newegg.select(".item-cell")
        
        products.extend([
            p for item in items_ddr4_newegg 
            if (p := parse_newegg_product(item, "DDR4"))
        ])
        products.extend([
            p for item in items_ddr5_newegg 
            if (p := parse_newegg_product(item, "DDR5"))
        ])
        
        newegg_count = len([p for p in products if p['source'] == 'newegg'])
        print(f"  Found {newegg_count} Newegg products")
    except Exception as e:
        print(f"  Newegg failed: {e}")
    
    # Scrape B&H Photo (works with cloudscraper)
    print("Scraping B&H Photo...")
    try:
        bhphoto_products = scrape_bhphoto()
        products.extend(bhphoto_products)
        print(f"  Found {len(bhphoto_products)} B&H Photo products")
    except Exception as e:
        print(f"  B&H Photo failed: {e}")
    
    return products


def save_to_csv(filename: str = "ram_data.csv", append: bool = False):
    """
    Scrape products and save to CSV.
    
    Args:
        filename: Output CSV filename
        append: If True, append to existing file (building price history over time)
    """
    products = scrape_all_products()
    new_df = pd.DataFrame(products)
    
    if append and Path(filename).exists():
        # Load existing data and append
        existing_df = pd.read_csv(filename)
        
        # Combine and remove exact duplicates (same product, same date, same price)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Define columns that identify a unique price point
        dedup_cols = ['source', 'brand', 'model', 'ddr_generation', 'capacity_gb', 
                      'frequency_mhz', 'price_usd', 'date_scraped']
        # Only use columns that exist
        dedup_cols = [c for c in dedup_cols if c in combined_df.columns]
        
        combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep='last')
        combined_df = combined_df.sort_values(['brand', 'model', 'date_scraped'])
        
        combined_df.to_csv(filename, index=False)
        print(f"\nAppended to {filename}: {len(new_df)} new rows, {len(combined_df)} total rows")
        return combined_df
    else:
        new_df.to_csv(filename, index=False)
        print(f"\nSaved {len(products)} products to {filename}")
        return new_df


# --- Wayback Machine Historical Scraping ---

def get_wayback_snapshots(url: str, from_date: str = None, to_date: str = None) -> list[dict]:
    """
    Get available Wayback Machine snapshots for a URL.
    
    Args:
        url: The URL to find snapshots for
        from_date: Start date (YYYYMMDD format), defaults to 1 year ago
        to_date: End date (YYYYMMDD format), defaults to today
    
    Returns:
        List of snapshot info dicts with 'timestamp' and 'url' keys
    """
    if not from_date:
        from_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y%m%d")
    
    # Wayback CDX API - returns list of snapshots
    cdx_url = f"https://web.archive.org/cdx/search/cdx?url={url}&from={from_date}&to={to_date}&output=json&fl=timestamp,original,statuscode"
    
    try:
        response = rq.get(cdx_url, timeout=30)
        if response.status_code != 200:
            print(f"  Wayback API returned {response.status_code}")
            return []
        
        data = response.json()
        if len(data) < 2:  # First row is headers
            return []
        
        snapshots = []
        seen_dates = set()
        
        for row in data[1:]:  # Skip header row
            timestamp, original, status = row[0], row[1], row[2]
            if status != "200":
                continue
            
            # Only keep one snapshot per month to avoid too many requests
            month_key = timestamp[:6]  # YYYYMM
            if month_key in seen_dates:
                continue
            seen_dates.add(month_key)
            
            snapshot_url = f"https://web.archive.org/web/{timestamp}/{original}"
            snapshot_date = datetime.strptime(timestamp[:8], "%Y%m%d").strftime("%Y-%m-%d")
            
            snapshots.append({
                'timestamp': timestamp,
                'date': snapshot_date,
                'url': snapshot_url
            })
        
        return snapshots
        
    except Exception as e:
        print(f"  Wayback API error: {e}")
        return []


def scrape_wayback_newegg(months_back: int = 12) -> list[dict]:
    """
    Scrape historical Newegg prices from Wayback Machine.
    
    Args:
        months_back: How many months of history to fetch
    
    Returns:
        List of product dicts with historical prices
    """
    products = []
    session = get_session()
    
    # Newegg RAM listing URLs
    urls = {
        "DDR4": "https://www.newegg.com/p/pl?N=100007611%20600561665",
        "DDR5": "https://www.newegg.com/p/pl?N=100007611%20601410157",
    }
    
    from_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y%m%d")
    
    for ddr_type, url in urls.items():
        print(f"  Finding Wayback snapshots for {ddr_type}...")
        snapshots = get_wayback_snapshots(url, from_date=from_date)
        print(f"    Found {len(snapshots)} monthly snapshots")
        
        for snapshot in snapshots:
            try:
                time.sleep(1)  # Be polite to archive.org
                response = session.get(snapshot['url'], timeout=30)
                
                if response.status_code != 200:
                    continue
                
                page = soup(response.text, "html.parser")
                items = page.select(".item-cell")
                
                for item in items:
                    product = parse_newegg_product(item, ddr_type)
                    if product:
                        # Override date with historical date
                        product['date_scraped'] = snapshot['date']
                        product['source'] = 'newegg_historical'
                        products.append(product)
                
                print(f"    {snapshot['date']}: {len(items)} products")
                
            except Exception as e:
                print(f"    {snapshot['date']}: failed - {e}")
                continue
    
    return products


def fetch_historical_data(months_back: int = 12, filename: str = "ram_data.csv"):
    """
    Fetch historical data from Wayback Machine and save to CSV.
    
    Args:
        months_back: How many months of history to fetch
        filename: Output CSV filename
    """
    print(f"Fetching {months_back} months of historical data from Wayback Machine...")
    print("(This may take a while - being polite to archive.org)\n")
    
    products = scrape_wayback_newegg(months_back=months_back)
    
    if not products:
        print("No historical data found")
        return None
    
    new_df = pd.DataFrame(products)
    
    # Append to existing data if file exists
    if Path(filename).exists():
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Deduplicate
        dedup_cols = ['source', 'brand', 'model', 'ddr_generation', 'capacity_gb', 
                      'frequency_mhz', 'price_usd', 'date_scraped']
        dedup_cols = [c for c in dedup_cols if c in combined_df.columns]
        combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep='last')
        combined_df = combined_df.sort_values(['brand', 'model', 'date_scraped'])
        
        combined_df.to_csv(filename, index=False)
        print(f"\nAdded {len(new_df)} historical records, {len(combined_df)} total rows")
        return combined_df
    else:
        new_df.to_csv(filename, index=False)
        print(f"\nSaved {len(products)} historical records to {filename}")
        return new_df


def get_price_history(df: pd.DataFrame, brand: str = None, model: str = None) -> pd.DataFrame:
    """
    Get price history for a specific product or brand.
    
    Args:
        df: DataFrame with price data
        brand: Filter by brand (optional)
        model: Filter by model (optional, substring match)
    
    Returns:
        DataFrame with price history sorted by date
    """
    filtered = df.copy()
    
    if brand:
        filtered = filtered[filtered['brand'].str.upper() == brand.upper()]
    
    if model:
        filtered = filtered[filtered['model'].str.contains(model, case=False, na=False)]
    
    return filtered.sort_values('date_scraped')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape RAM prices")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV (build history)")
    parser.add_argument("--historical", type=int, metavar="MONTHS", help="Fetch N months of historical data from Wayback Machine")
    parser.add_argument("--output", "-o", default="ram_data.csv", help="Output filename")
    
    args = parser.parse_args()
    
    if args.historical:
        fetch_historical_data(months_back=args.historical, filename=args.output)
    else:
        save_to_csv(filename=args.output, append=args.append)
