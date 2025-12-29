"""
Transform scraped ram_data.csv and append to master memory.csv file.
"""
import pandas as pd
import os


def transform_scraped_data(scraped_file: str = "ram_data.csv") -> pd.DataFrame:
    """
    Transform scraped data to match memory.csv schema.
    
    memory.csv columns:
        name, price, speed, modules, price_per_gb, color, first_word_latency, cas_latency
    """
    df = pd.read_csv(scraped_file)
    transformed = pd.DataFrame()
    
    # name: "Brand Model Capacity GB"
    transformed['name'] = df.apply(
        lambda row: f"{row['brand']} {row['model']} {int(row['capacity_gb'])} GB" 
        if pd.notna(row['capacity_gb']) else f"{row['brand']} {row['model']}",
        axis=1
    )
    
    # price
    transformed['price'] = df['price_usd']
    
    # speed: "DDR_NUM,FREQ" (e.g., "5,6000")
    def format_speed(row):
        ddr = row['ddr_generation']
        freq = row['frequency_mhz']
        if pd.isna(ddr) or pd.isna(freq):
            return None
        ddr_num = ddr.replace('DDR', '') if isinstance(ddr, str) else ddr
        return f"{ddr_num},{int(freq)}"
    
    transformed['speed'] = df.apply(format_speed, axis=1)
    
    # modules: "NUM_STICKS,GB_PER_STICK"
    def format_modules(capacity):
        if pd.isna(capacity):
            return None
        capacity = int(capacity)
        if capacity >= 8:
            return f"2,{capacity // 2}"
        return f"1,{capacity}"
    
    transformed['modules'] = df['capacity_gb'].apply(format_modules)
    
    # price_per_gb
    transformed['price_per_gb'] = df.apply(
        lambda row: round(row['price_usd'] / row['capacity_gb'], 3) 
        if pd.notna(row['price_usd']) and pd.notna(row['capacity_gb']) and row['capacity_gb'] > 0 
        else None,
        axis=1
    )
    
    # color: not available
    transformed['color'] = None
    
    # first_word_latency: not available
    transformed['first_word_latency'] = None
    
    # cas_latency
    transformed['cas_latency'] = df['cas_latency']
    
    # Remove rows with no price
    transformed = transformed.dropna(subset=['price'])
    
    return transformed


def append_to_master(master_file: str = "memory.csv", scraped_file: str = "ram_data.csv"):
    """
    Transform scraped data and append to master memory.csv file.
    """
    # Transform scraped data
    print(f"Transforming {scraped_file}...")
    new_data = transform_scraped_data(scraped_file)
    print(f"  {len(new_data)} rows transformed")
    
    # Load master file
    print(f"Loading master file {master_file}...")
    master = pd.read_csv(master_file)
    print(f"  {len(master)} existing rows")
    
    # Append new data
    combined = pd.concat([master, new_data], ignore_index=True)
    
    # Remove duplicates (keep first occurrence)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=['name', 'price'], keep='first')
    after_dedup = len(combined)
    
    # Save back to master file
    combined.to_csv(master_file, index=False)
    
    print(f"\nResult:")
    print(f"  Added {len(new_data)} new rows")
    print(f"  Removed {before_dedup - after_dedup} duplicates")
    print(f"  Total: {len(combined)} rows in {master_file}")
    
    return combined


def cleanup_temp_files():
    """Remove unnecessary CSV files."""
    temp_files = [
        "ram_data_transformed.csv",
        "memory_combined.csv",
        "_temp_transformed.csv",
    ]
    
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted {f}")


if __name__ == "__main__":
    # Append scraped data to master
    append_to_master()
    
    # Clean up temp files
    print("\nCleaning up...")
    cleanup_temp_files()
