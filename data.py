import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_ram_data(url):
    scraper = cloudscraper.create_scraper()
    print(f"Fetching data from: {url}...")
    
    response = scraper.get(url)
    
    if response.status_code != 200:
        print(f"Blocked! Status Code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    ram_list = []

    # when viewed via scraper. Let's target the table body directly.
    table = soup.find('table', class_='list--generic')
    if not table:
        print("Could not find the data table. The page layout might have changed or you're being blocked.")
        return None

    for row in table.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        if len(cols) > 1:
            name = cols[1].text.strip() # Usually 2nd column
            price = cols[9].text.strip() # Usually 10th column (Price)
            
            ram_list.append({
                "Name": name,
                "Price": price,
                "Timestamp": pd.Timestamp.now()
            })

    df = pd.DataFrame(ram_list)
    
    # Use os.path.expanduser to handle the '~' correctly on Mac
    save_path = os.path.expanduser("~/workflow/python_proj/RAM/ram_data.csv")
    df.to_csv(save_path, index=False)
    print(f"Success! Saved {len(df)} rows to {save_path}")
    return df

target_url = "https://pcpartpicker.com/products/memory/"
df = scrape_ram_data(target_url)
print(df)
