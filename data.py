import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_ram_data(url):
    #define headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # 2. Get the HTML content
    print(f"Fetching data from: {url}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return None

    # 3. Parse with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
   
    #extract data
    ram_list = []
    
    # Example logic: Find all rows in a results table
    for item in soup.find_all('tr', class_='product-row'): # Update 'product-row'
        try:
            name = item.find('td', class_='td__name').text.strip()
            price = item.find('td', class_='td__price').text.strip()
            
            ram_list.append({
                "Name": name,
                "Price": price,
                "Timestamp": pd.Timestamp.now()
            })
        except AttributeError:
            continue # Skip rows that don't match our pattern
    df = pd.DataFrame(ram_list)
    df.to_csv("/Users/gabegaines/workflow/python_proj/RAM/ram_data.csv", index=False)
    return df

target_url = "https://pcpartpicker.com/products/memory/"
df = scrape_ram_data(target_url)
print(df.head())
