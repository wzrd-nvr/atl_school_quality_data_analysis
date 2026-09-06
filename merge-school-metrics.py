import pandas as pd
import numpy as np
import os
from fuzzywuzzy import fuzz, process

print("Starting Improved School Data Merging Process...")

# List of specific files to process
SPECIFIC_FILES = [
    "data/AP_2023-24_2025-01-15_15_03_20.csv",
    "data/ELSI_NCES_GA_school_data.csv",
    "data/EOC_2023-24__GA_TST_AGGR_2025-01-14_16_19_30.csv",
    "data/EOG_2023-24__GA_TST_AGGR_2025-01-14_16_19_30.csv",
    "data/HS_Completer_Credentials_2023-24_2025-01-14_16_45_56.csv",
    "data/School_FESR_FY24_for_Display.csv"
]

# Hard-coded column mappings for each file
COLUMN_MAPPINGS = {
    "AP_2023-24_2025-01-15_15_03_20.csv": {
        "school_name_col": "INSTN_NAME"
    },
    "ELSI_NCES_GA_school_data.csv": {
        "school_name_col": "School Name"
    },
    "EOC_2023-24__GA_TST_AGGR_2025-01-14_16_19_30.csv": {
        "school_name_col": "INSTN_NAME"
    },
    "EOG_2023-24__GA_TST_AGGR_2025-01-14_16_19_30.csv": {
        "school_name_col": "INSTN_NAME"
    },
    "HS_Completer_Credentials_2023-24_2025-01-14_16_45_56.csv": {
        "school_name_col": "INSTN_NAME"
    },
    "School_FESR_FY24_for_Display.csv": {
        "school_name_col": "schoolname"
    }
}

# Function to clean and standardize school names
def clean_school_name(name):
    """Standardize school names to improve matching."""
    if pd.isna(name):
        return ""
    
    # Convert to string and lowercase
    name = str(name).lower().strip()
    
    # Replace common abbreviations
    replacements = {
        'elem': 'elementary',
        'sch': 'school',
        'schl': 'school',
        'hs': 'high school',
        'h.s.': 'high school',
        'ms': 'middle school',
        'm.s.': 'middle school',
        'es': 'elementary school',
        'e.s.': 'elementary school',
        'acad': 'academy',
        'tech': 'technical',
        'alt': 'alternative',
        'ctr': 'center',
        'intl': 'international',
        'jr.': 'junior',
        'sr.': 'senior',
        'pk': 'pre-k',
        'prek': 'pre-k'
    }
    
    for abbr, full in replacements.items():
        # Replace the abbreviation if it's a standalone word
        name = name.replace(f" {abbr} ", f" {full} ")
        # Also check if it's at the end of the name
        if name.endswith(f" {abbr}"):
            name = name[:-len(abbr)] + full
    
    # Remove special characters and extra spaces
    name = ''.join(e for e in name if e.isalnum() or e.isspace())
    name = ' '.join(name.split())
    
    return name

# Function to find the best match for a school name in another dataset
def find_best_match(name, name_list, threshold=85):
    """
    Find the best matching school name in name_list for the given name.
    Returns the best match and the match score.
    """
    if pd.isna(name) or name == "":
        return None, 0
    
    # Get the top 3 matches
    matches = process.extract(name, name_list, scorer=fuzz.token_sort_ratio, limit=3)
    
    # Filter matches by threshold
    valid_matches = [m for m in matches if m[1] >= threshold]
    
    if valid_matches:
        # Return the best match and its score
        return valid_matches[0]
    else:
        return None, 0

# Load the filtered NCES schools data which has the schools we want to keep
try:
    filtered_file = "data/filtered_nces_schools.csv"
    print(f"Loading filtered schools data from {filtered_file}...")
    
    filtered_schools = pd.read_csv(filtered_file)
    print(f"Loaded filtered schools data with {len(filtered_schools)} schools")
    
    # Try to determine the school name column and zip code column
    school_name_col = None
    possible_name_cols = ['School Name', 'SCHNAM', 'school_name', 'NAME', 'name', 'School']
    
    for col in possible_name_cols:
        if col in filtered_schools.columns:
            school_name_col = col
            print(f"Found school name column: {school_name_col}")
            break
    
    if not school_name_col:
        # Show the first few columns to help the user identify the school name column
        print("\nAvailable columns in filtered schools data:")
        for i, col in enumerate(filtered_schools.columns[:10]):
            print(f"{i+1}. {col}")
        
        school_name_col = input("Enter the name of the school name column: ")
    
    # Find the zip code column
    zip_col = None
    possible_zip_cols = ['Zip Code', 'ZIP', 'zip', 'zipcode', 'ZIP_CODE', 'SCHZIP', 'Location ZIP [Public School] 2023-24']
    
    for col in possible_zip_cols:
        if col in filtered_schools.columns:
            zip_col = col
            print(f"Found zip code column: {zip_col}")
            break
    
    if not zip_col:
        # Look for column names containing 'zip'
        for col in filtered_schools.columns:
            if 'zip' in col.lower():
                zip_col = col
                print(f"Found likely zip code column: {zip_col}")
                break
    
    if not zip_col:
        # Show columns to help user identify zip code column
        print("\nAvailable columns in filtered schools data:")
        for i, col in enumerate(filtered_schools.columns[:10]):
            print(f"{i+1}. {col}")
        
        zip_col = input("Enter the name of the zip code column: ")
    
    # Create a clean version of school names for matching
    filtered_schools['clean_name'] = filtered_schools[school_name_col].apply(clean_school_name)
    
    # Create a unique ID for each school based on name and zip
    filtered_schools['school_id'] = filtered_schools.apply(
        lambda row: f"{row['clean_name']}_{str(row[zip_col])}", axis=1
    )
    
    # Keep only essential columns for the reference dataset
    filtered_reference = filtered_schools[[school_name_col, zip_col, 'clean_name', 'school_id']].copy()
    print(f"Reference data ready with {len(filtered_reference)} schools")
    
    # This will be our base dataframe that we'll merge additional data into
    # Start with the filtered schools data as the foundation
    merged_data = filtered_schools.copy()
    
except Exception as e:
    print(f"Error loading filtered schools data: {e}")
    exit(1)

# Process each data file and merge into the base dataframe
for file_path in SPECIFIC_FILES:
    file_name = os.path.basename(file_path)
    
    try:
        print(f"\nProcessing {file_name}...")
        
        # Skip the NCES data file since we already used it as our base
        if file_name == "ELSI_NCES_GA_school_data.csv":
            print(f"Skipping {file_name} as it's already used as the base dataset")
            continue
        
        # Load the data
        data = pd.read_csv(file_path)
        print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
        
        # Get the column mappings for this file
        file_mapping = COLUMN_MAPPINGS.get(file_name, {})
        data_name_col = file_mapping.get("school_name_col")
        
        # If we don't have a predefined mapping, try to detect the school name column
        if not data_name_col or data_name_col not in data.columns:
            possible_name_cols = ['School Name', 'SCHNAM', 'school_name', 'NAME', 'name', 'School']
            for col in possible_name_cols:
                if col in data.columns:
                    data_name_col = col
                    print(f"Found school name column in data: {data_name_col}")
                    break
        
        if not data_name_col or data_name_col not in data.columns:
            print(f"Could not find school name column in {file_name}. Available columns:")
            for i, col in enumerate(data.columns[:10]):
                print(f"{i+1}. {col}")
            
            data_name_col = input(f"Please enter the name of the school name column for {file_name}: ")
        
        # Clean the school names for matching
        data['clean_name'] = data[data_name_col].apply(clean_school_name)
        
        # Add a source file column to track where data came from
        data['source_file'] = file_name
        
        # Create a mapping dictionary to store matches
        matches = {}
        
        # Get unique list of clean school names in this data file
        data_schools = data['clean_name'].dropna().unique().tolist()
        
        # Track matches for reporting
        matches_found = 0
        total_schools = len(filtered_reference)
        
        # For each school in the filtered reference dataset, find the best match in this data file
        print(f"Matching schools from {file_name} with filtered schools...")
        
        for i, row in filtered_reference.iterrows():
            school_name = row['clean_name']
            original_name = row[school_name_col]
            zip_code = row[zip_col]
            school_id = row['school_id']
            
            # Find the best match
            best_match, score = find_best_match(school_name, data_schools)
            
            if best_match and score >= 85:
                # Found a good match - store the mapping
                matches[school_id] = best_match
                matches_found += 1
        
        print(f"Found {matches_found} matches out of {total_schools} filtered schools")
        
        if matches_found == 0:
            print(f"No matches found for {file_name}, skipping")
            continue
        
        # Add the school_id to the data file for merging
        data['school_id'] = np.nan
        
        # Use the matches dictionary to assign school_ids to matching rows
        for school_id, match_name in matches.items():
            # Find all rows with this clean_name
            matching_rows = data['clean_name'] == match_name
            # Assign the school_id to these rows
            data.loc[matching_rows, 'school_id'] = school_id
        
        # Filter to keep only rows with a school_id (matched schools)
        matched_data = data[data['school_id'].notna()].copy()
        
        # For datasets with multiple rows per school (e.g., different test subjects)
        # We need to determine how to handle them
        school_counts = matched_data['school_id'].value_counts()
        multiple_rows = school_counts[school_counts > 1].index.tolist()
        
        if multiple_rows:
            print(f"Found {len(multiple_rows)} schools with multiple rows in {file_name}")
            print("For these schools, we'll create separate columns for each unique value")
            
            # Get columns that might vary across rows for the same school
            # Exclude certain columns we know are consistent
            exclude_cols = ['clean_name', 'school_id', 'source_file', data_name_col]
            potential_varying_cols = [col for col in matched_data.columns if col not in exclude_cols]
            
            # For each school with multiple rows
            for school_id in multiple_rows:
                # Get all rows for this school
                school_rows = matched_data[matched_data['school_id'] == school_id]
                
                # Look for columns that have different values across rows
                for col in potential_varying_cols:
                    unique_values = school_rows[col].dropna().unique()
                    
                    # If there are multiple unique values, create separate columns
                    if len(unique_values) > 1:
                        # Find a distinguishing column to use as a suffix
                        # Try common differentiators like subject, grade, etc.
                        differentiator_cols = ['Subject', 'Grade', 'SUBGROUP_NAME', 'TEST_CMPNT_TYP_NM', 'ACDMC_LVL']
                        
                        differentiator = None
                        for diff_col in differentiator_cols:
                            if diff_col in school_rows.columns and school_rows[diff_col].nunique() > 1:
                                differentiator = diff_col
                                break
                        
                        if differentiator:
                            # Create new columns with the differentiator value as a suffix
                            for _, row in school_rows.iterrows():
                                diff_value = row[differentiator]
                                if pd.notna(diff_value) and pd.notna(row[col]):
                                    # Create a new column name with the differentiator
                                    new_col = f"{col}_{diff_value}"
                                    # Clean up the column name
                                    new_col = new_col.replace(" ", "_").replace(".", "").replace("/", "_")
                                    # Add this value to the first row for this school
                                    matched_data.loc[matched_data['school_id'] == school_id, new_col] = row[col]
            
            # Now collapse to one row per school by grouping on school_id
            # For any remaining duplicate values, we'll use the first one
            matched_data = matched_data.groupby('school_id').first().reset_index()
            print(f"After collapsing to one row per school: {len(matched_data)} rows")
        
        # Drop the clean_name column as it was just for matching
        if 'clean_name' in matched_data.columns:
            matched_data = matched_data.drop('clean_name', axis=1)
        
        # Prepare column names for merge - prefix columns with source file to avoid conflicts
        # Exclude certain columns from prefix (common joining columns)
        no_prefix_cols = ['school_id', school_name_col, zip_col]
        
        # Create a prefix for this data source
        file_prefix = file_name.split('_')[0].lower() + "_"
        
        # Create a dictionary to rename columns
        rename_dict = {}
        for col in matched_data.columns:
            if col not in no_prefix_cols:
                # Check if column already exists in merged_data
                if col in merged_data.columns:
                    rename_dict[col] = file_prefix + col
        
        # Rename columns to avoid conflicts
        matched_data = matched_data.rename(columns=rename_dict)
        
        # Check for any remaining column conflicts
        conflicts = [col for col in matched_data.columns if col in merged_data.columns and col != 'school_id']
        if conflicts:
            print(f"Warning: Column conflicts found: {conflicts}")
            print("These columns will be overwritten in the merged dataset")
        
        # Merge with the main dataframe
        print(f"Merging {len(matched_data)} rows from {file_name} into main dataset")
        merged_data = pd.merge(
            merged_data, 
            matched_data,
            on='school_id',
            how='left',
            suffixes=('', f'_{file_prefix}')
        )
        print(f"Merged dataset now has {len(merged_data)} rows and {len(merged_data.columns)} columns")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print(f"Continuing with next file...")

# Drop the temporary columns used for merging
if 'clean_name' in merged_data.columns:
    merged_data = merged_data.drop('clean_name', axis=1)

# Save the properly merged dataset
try:
    output_filename = "data/properly_merged_school_data.csv"
    merged_data.to_csv(output_filename, index=False)
    print(f"\nSaved properly merged dataset with {len(merged_data)} rows and {len(merged_data.columns)} columns to {output_filename}")
except Exception as e:
    print(f"Error saving merged dataset: {e}")

print("\nProcessing complete!")