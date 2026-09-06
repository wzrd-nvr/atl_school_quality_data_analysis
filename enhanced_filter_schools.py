import pandas as pd
import numpy as np
import csv
import os

print("Starting enhanced CSV parser for school data...")

# Function to detect CSV delimiter
def detect_delimiter(file_path, num_lines=20):
    """Try to automatically detect the delimiter in a CSV file"""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        # Read a small sample of the file
        sample = ''.join(file.readline() for _ in range(num_lines))
        
        # Count potential delimiters
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}
        
        for delimiter in delimiters:
            delimiter_counts[delimiter] = sample.count(delimiter)
        
        # Return the delimiter with the highest count
        most_common_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        print(f"Detected delimiter: '{most_common_delimiter}' (found {delimiter_counts[most_common_delimiter]} times in sample)")
        
        return most_common_delimiter

# Function to try multiple approaches to read the CSV
def read_csv_robust(file_path):
    """Try multiple approaches to read a problematic CSV file"""
    # First, try to detect the delimiter
    try:
        delimiter = detect_delimiter(file_path)
        df = pd.read_csv(file_path, delimiter=delimiter, error_bad_lines=False, warn_bad_lines=True)
        print(f"Successfully read CSV with delimiter: '{delimiter}'")
        return df
    except Exception as e:
        print(f"First attempt failed: {e}")
    
    # Second, try with Python's csv module for more flexibility
    try:
        print("Trying with csv module...")
        rows = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            # Try to sniff the dialect/format
            sample = file.read(4096)
            file.seek(0)
            
            try:
                dialect = csv.Sniffer().sniff(sample)
                print(f"Detected dialect with delimiter: '{dialect.delimiter}'")
                reader = csv.reader(file, dialect)
            except:
                print("Could not detect dialect, using excel format")
                reader = csv.reader(file)
            
            headers = next(reader)
            for row in reader:
                # Skip empty rows or rows with different column counts
                if row and len(row) == len(headers):
                    rows.append(row)
        
        df = pd.DataFrame(rows, columns=headers)
        print(f"Successfully read {len(df)} rows using csv module")
        return df
    except Exception as e:
        print(f"Second attempt failed: {e}")
    
    # Third, try reading with different encodings and error handling
    for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
        try:
            print(f"Trying with encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding, engine='python')
            print(f"Successfully read CSV with encoding: {encoding}")
            return df
        except Exception as e:
            print(f"Attempt with {encoding} failed: {e}")
    
    raise Exception("Could not read the CSV file after multiple attempts")

# Load the zipcodes data first (assuming it's simpler)
try:
    zipcodes_df = pd.read_csv('data/zipcodes.csv')
    print(f"Loaded {len(zipcodes_df)} target zip codes")
    
    # Get the list of target zip codes
    # Check for 'Zipcode' column in zipcodes_df
    if 'Zipcode' in zipcodes_df.columns:
        target_zips = zipcodes_df['Zipcode'].astype(str).tolist()
    elif 'ZIP' in zipcodes_df.columns:
        target_zips = zipcodes_df['ZIP'].astype(str).tolist()
    elif 'zip' in zipcodes_df.columns:
        target_zips = zipcodes_df['zip'].astype(str).tolist()
    else:
        # Take the first column as zip code
        zip_col_name = zipcodes_df.columns[0]
        target_zips = zipcodes_df[zip_col_name].astype(str).tolist()
    
    print(f"Found {len(target_zips)} target zip codes")
    print(f"Sample of target zip codes: {target_zips[:5]}")
    
except Exception as e:
    print(f"Error reading zipcodes.csv: {e}")
    print("Using default test zipcode list")
    target_zips = ['30306', '30307', '30308', '30309', '30310']  # Default test values

# Try to load the NCES data with our robust method
try:
    print("\nAttempting to read ELSI_NCES_GA_school_data.csv...")
    nces_data = read_csv_robust('data/ELSI_NCES_GA_school_data.csv')
    print(f"Successfully loaded NCES data with {len(nces_data)} rows and {len(nces_data.columns)} columns")
    
    # Display the first few columns to help identify the zip code column
    print("\nFirst 5 columns:")
    for i, col in enumerate(nces_data.columns[:5]):
        print(f"{i}: {col}")
    
    # Display a small sample of rows to help understand the data
    print("\nSample of first 3 rows and 5 columns:")
    sample_df = nces_data.iloc[:3, :5]
    print(sample_df)
    
    # Look for likely zip code columns
    print("\nLooking for zip code columns...")
    zip_col = None
    possible_zip_cols = ['ZIP', 'Zip', 'zip', 'ZIP_CODE', 'Zip_Code', 'zip_code', 
                         'Postal', 'postal', 'Postal_Code', 'postal_code', 
                         'Location_ZIP', 'SCHZIP', 'SCH_ZIP', 'ZIP Code']
    
    for col in possible_zip_cols:
        if col in nces_data.columns:
            zip_col = col
            print(f"Found zip code column: {zip_col}")
            break
    
    # If no standard zip column found, try to identify it
    if not zip_col:
        # Look for column names containing 'zip'
        for col in nces_data.columns:
            if 'zip' in col.lower():
                zip_col = col
                print(f"Found likely zip code column: {zip_col}")
                break
        
        # If still not found, examine a sample of each column to find zip code patterns
        if not zip_col:
            print("No standard zip code column found. Examining data patterns...")
            for col in nces_data.columns:
                # Skip very large text fields 
                if nces_data[col].astype(str).str.len().max() > 20:
                    continue
                    
                sample = nces_data[col].dropna().astype(str).iloc[:10].tolist()
                # Check if values match 5-digit zip code pattern
                zip_pattern_count = sum(1 for val in sample if len(str(val)) == 5 and str(val).isdigit())
                if zip_pattern_count >= 3:  # If at least 3 values look like zip codes
                    zip_col = col
                    print(f"Found likely zip code column based on data pattern: {zip_col}")
                    print(f"Sample values: {sample[:5]}")
                    break
    
    # If still can't find zip column, list all columns and ask user to specify
    if not zip_col:
        print("\nCouldn't automatically detect zip code column. Here are all columns:")
        for i, col in enumerate(nces_data.columns):
            print(f"{i}: {col}")
        
        col_index = int(input("\nEnter the column index for zip codes: "))
        zip_col = nces_data.columns[col_index]
        print(f"Using column '{zip_col}' for zip codes")
    
    # Ensure zip codes are treated as strings
    nces_data[zip_col] = nces_data[zip_col].astype(str)
    
    # Clean up zip codes (remove any spaces, trim to first 5 digits)
    nces_data[zip_col] = nces_data[zip_col].str.replace(' ', '')
    nces_data[zip_col] = nces_data[zip_col].str.extract(r'(\d{5})').fillna(nces_data[zip_col])
    
    # Filter schools by target zip codes
    filtered_schools = nces_data[nces_data[zip_col].isin(target_zips)]
    print(f"\nFound {len(filtered_schools)} schools in the target zip codes")
    
    # Check if any target zip codes have no schools
    zips_with_schools = filtered_schools[zip_col].unique()
    missing_zips = [z for z in target_zips if z not in zips_with_schools]
    
    if missing_zips:
        print(f"\nWarning: {len(missing_zips)} target zip codes have no schools in the NCES data:")
        print(missing_zips)
    
    # Save the filtered data
    filtered_schools.to_csv('data/filtered_nces_schools.csv', index=False)
    print("\nFiltered schools data saved to 'filtered_nces_schools.csv'")
    
    # Print schools per zip code
    schools_per_zip = filtered_schools.groupby(zip_col).size()
    print("\nNumber of schools per zip code:")
    print(schools_per_zip)
    
    print("\nSuccess! Processing complete.")
    
except Exception as e:
    print(f"Error processing NCES data: {e}")
    print("\nRecommendations for manual inspection:")
    print("1. Open ELSI_NCES_GA_school_data.csv in a text editor to check its format")
    print("2. Look for unusual characters or formatting issues")
    print("3. Try opening the file in Excel or another spreadsheet program and save it as a clean CSV")