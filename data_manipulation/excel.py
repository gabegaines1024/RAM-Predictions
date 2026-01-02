"""
Export RAM data from CSV to Excel (.xlsx) format.
"""
import pandas as pd
from pathlib import Path


def csv_to_excel(
    input_file: str = "data/memory.csv",
    output_file: str = "data/memory.xlsx"
):
    """
    Convert a CSV file to Excel format.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output Excel file (.xlsx)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: {input_file} not found")
        return None
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  {len(df)} rows loaded")
    
    # Export to Excel
    print(f"Writing to {output_file}...")
    df.to_excel(output_file, index=False, sheet_name="RAM Data", engine="openpyxl")
    print(f"  Done! Excel file saved to {output_file}")
    
    return df


def export_all_data(output_file: str = "data/ram_all_data.xlsx"):
    """
    Export all available CSV data files to a single Excel workbook with multiple sheets.
    
    Args:
        output_file: Path to output Excel file
    """
    data_dir = Path("data")
    csv_files = {
        "memory.csv": "Memory",
        "ram_data.csv": "Scraped Data",
    }
    
    print(f"Exporting all data to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        sheets_written = 0
        
        for csv_name, sheet_name in csv_files.items():
            csv_path = data_dir / csv_name
            
            if csv_path.exists():
                print(f"  Reading {csv_name}...")
                df = pd.read_csv(csv_path)
                df.to_excel(writer, index=False, sheet_name=sheet_name)
                print(f"    {len(df)} rows -> '{sheet_name}' sheet")
                sheets_written += 1
            else:
                print(f"  Skipping {csv_name} (not found)")
    
    if sheets_written > 0:
        print(f"\nDone! {sheets_written} sheet(s) saved to {output_file}")
    else:
        print("\nNo CSV files found to export")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export RAM data to Excel")
    parser.add_argument(
        "--input", "-i",
        default="data/memory.csv",
        help="Input CSV file (default: data/memory.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output Excel file (default: same name with .xlsx extension)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Export all CSV files to a single Excel workbook with multiple sheets"
    )
    
    args = parser.parse_args()
    
    if args.all:
        output = args.output or "data/ram_all_data.xlsx"
        export_all_data(output_file=output)
    else:
        # Default output: replace .csv with .xlsx
        if args.output:
            output = args.output
        else:
            output = str(Path(args.input).with_suffix(".xlsx"))
        
        csv_to_excel(input_file=args.input, output_file=output)

