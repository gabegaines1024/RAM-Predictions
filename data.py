from seleniumbase import SB
import pandas as pd
import os

def scrape_with_sb(url):
    # Use 'uc=True' to enable Undetectable Mode
    with SB(uc=True, test=True) as sb:
        print(f"Opening {url} in UC Mode...")
        # uc_open_with_reconnect helps bypass initial Cloudflare blocks
        sb.uc_open_with_reconnect(url, reconnect_time=4)
        
        # Wait for the table to actually appear on the page
        sb.wait_for_element("table.list--generic", timeout=15)
        
        # Get the page source and parse it
        # You can use sb's built-in methods or pass to BeautifulSoup
        ram_elements = sb.find_elements("tr.product-row")
        
        ram_data = []
        for element in ram_elements:
            try:
                # Use CSS selectors to find text within the row
                name = element.find_element("td.td__name").text.strip()
                price = element.find_element("td.td__price").text.strip()
                
                ram_data.append({
                    "Name": name,
                    "Price": price,
                    "Timestamp": pd.Timestamp.now()
                })
            except:
                continue
                
        df = pd.DataFrame(ram_data)
        save_path = os.path.expanduser("~/workflow/python_proj/RAM/ram_data_selenium.csv")
        df.to_csv(save_path, index=False)
        print(f"Successfully scraped {len(df)} items!")
        return df

target_url = "https://pcpartpicker.com/products/memory/"
scrape_with_sb(target_url)
