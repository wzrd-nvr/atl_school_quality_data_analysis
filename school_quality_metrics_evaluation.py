import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re

print("Starting School Quality Metrics Evaluation...")

# Configuration - metric categories and their weights
METRIC_WEIGHTS = {
    'academic': 0.35,     # Test scores, proficiency rates
    'advanced': 0.15,     # AP scores, advanced courses
    'resources': 0.15,    # Funding, student-teacher ratio
    'equity': 0.15,       # Achievement gaps, free/reduced lunch performance
    'environment': 0.10,  # School climate, student engagement
    'outcomes': 0.10      # Graduation rates, college readiness
}

# Define patterns to identify columns related to each metric category
METRIC_PATTERNS = {
    'academic': [
        'PROFICIENT', 'DISTINGUISHED', 'test', 'score', 'assessment', 'achievement',
        'reading', 'math', 'science', 'EOG', 'EOC', 'LABEL_LVL'
    ],
    'advanced': [
        'AP', 'advanced placement', 'honors', 'gifted', 'accelerated',
        'dual enrollment', 'international baccalaureate', 'NUMBER_TESTS_3_OR_HIGHER'
    ],
    'resources': [
        'PPE', 'expenditure', 'spend', 'ratio', 'Pupil/Teacher', 'funding',
        'Federal_Amt', 'State_Local_Amt', 'allocation'
    ],
    'equity': [
        'gap', 'subgroup', 'Free Lunch', 'Reduced-price Lunch', 'Direct Certification',
        'economically disadvantaged', 'equity'
    ],
    'environment': [
        'climate', 'attendance', 'absent', 'discipline', 'suspension',
        'safety', 'engagement', 'extracurricular'
    ],
    'outcomes': [
        'graduation', 'dropout', 'college', 'career', 'readiness', 'completer',
        'credential', 'FESR', 'COMPLETER_TYPE'
    ]
}

# Define which metrics are positive (higher is better) vs negative (lower is better)
NEGATIVE_METRICS = [
    'absent', 'dropout', 'suspension', 'discipline', 'gap'
]

def load_data(file_path):
    """Load the school data from CSV."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def identify_key_columns(df):
    """Identify the school name, zip code, and key metric columns."""
    # Find school name column
    school_name_col = None
    possible_name_cols = ['School Name', 'SCHNAM', 'school_name', 'NAME', 'name', 'School', 
                          'INSTN_NAME', 'School Name [Public School] 2023-24']
    
    for col in possible_name_cols:
        if col in df.columns:
            school_name_col = col
            print(f"Found school name column: {school_name_col}")
            break
    
    if not school_name_col:
        print("Could not identify school name column. Using first column as default.")
        school_name_col = df.columns[0]
    
    # Find zip code column
    zip_col = None
    possible_zip_cols = ['Zip Code', 'ZIP', 'zip', 'zipcode', 'ZIP_CODE', 'SCHZIP',
                         'Location ZIP [Public School] 2023-24', 'zip_col']
    
    for col in possible_zip_cols:
        if col in df.columns:
            zip_col = col
            print(f"Found zip code column: {zip_col}")
            break
    
    if not zip_col:
        # Look for column names containing 'zip'
        for col in df.columns:
            if 'zip' in col.lower():
                zip_col = col
                print(f"Found likely zip code column: {zip_col}")
                break
    
    if not zip_col:
        print("Could not identify zip code column.")
        zip_col = None
    
    # Identify metric columns by category
    metric_columns = {category: [] for category in METRIC_WEIGHTS.keys()}
    
    for col in df.columns:
        if col == school_name_col or col == zip_col:
            continue
            
        # Check if column contains numeric data (potential metric)
        if df[col].dtype in ['int64', 'float64'] or pd.to_numeric(df[col], errors='coerce').notna().any():
            # Determine which category this metric belongs to
            for category, patterns in METRIC_PATTERNS.items():
                if any(pattern.lower() in col.lower() for pattern in patterns):
                    metric_columns[category].append(col)
                    break
    
    # Print summary of identified metric columns
    for category, columns in metric_columns.items():
        if columns:
            print(f"Found {len(columns)} columns for {category} metrics")
    
    return school_name_col, zip_col, metric_columns

def is_positive_metric(col_name):
    """Determine if a metric is positive (higher is better) or negative (lower is better)."""
    # Default assumption: higher values are better
    return not any(neg_pattern.lower() in col_name.lower() for neg_pattern in NEGATIVE_METRICS)

def calculate_category_scores(df, metric_columns):
    """Calculate normalized scores for each metric category."""
    category_scores = {}
    df_scores = df.copy()
    
    # Scaler for normalizing metrics to 0-1 range
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    for category, columns in metric_columns.items():
        if not columns:
            print(f"No columns found for {category} category. Skipping.")
            continue
        
        category_df = pd.DataFrame(index=df.index)
        
        # Process each metric column
        for col in columns:
            # Convert to numeric, handling non-numeric values
            series = pd.to_numeric(df[col], errors='coerce')
            
            # Skip if too many missing values
            if series.isna().sum() > 0.5 * len(series):
                print(f"Skipping {col} - too many missing values")
                continue
                
            # Fill remaining missing values with median
            series = series.fillna(series.median())
            
            # Normalize the values to 0-1 scale
            try:
                normalized = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
                
                # Invert if this is a negative metric (lower is better)
                if not is_positive_metric(col):
                    normalized = 1 - normalized
                    
                category_df[col] = normalized
            except Exception as e:
                print(f"Error normalizing {col}: {e}")
                continue
        
        # Calculate the mean score for this category (if we have metrics)
        if not category_df.empty:
            category_scores[category] = category_df.mean(axis=1)
            df_scores[f'{category}_score'] = category_df.mean(axis=1)
            
            # Store which metrics contributed to this category
            metric_count = len(category_df.columns)
            df_scores[f'{category}_metrics_used'] = metric_count
            df_scores[f'{category}_metrics'] = ', '.join(category_df.columns[:5]) + \
                                              (f' and {metric_count-5} more' if metric_count > 5 else '')
        else:
            # No valid metrics for this category
            category_scores[category] = pd.Series(np.nan, index=df.index)
            df_scores[f'{category}_score'] = np.nan
            df_scores[f'{category}_metrics_used'] = 0
            df_scores[f'{category}_metrics'] = 'None'
    
    return category_scores, df_scores

def calculate_overall_scores(df_scores, category_scores):
    """Calculate weighted overall quality scores."""
    # Calculate weighted sum of category scores
    overall_scores = pd.Series(0.0, index=df_scores.index)
    weights_applied = pd.Series(0.0, index=df_scores.index)
    
    for category, weight in METRIC_WEIGHTS.items():
        if category in category_scores:
            # Only apply weight if we have data for this category
            mask = ~category_scores[category].isna()
            overall_scores.loc[mask] += category_scores[category].loc[mask] * weight
            weights_applied.loc[mask] += weight
    
    # Adjust for missing categories
    mask = weights_applied > 0
    overall_scores.loc[mask] = overall_scores.loc[mask] / weights_applied.loc[mask]
    
    # Convert to 0-100 scale
    overall_scores = overall_scores * 100
    
    # Assign letter grades based on the distribution of scores
    # Find the min and max scores
    min_score = overall_scores.min()
    max_score = overall_scores.max()
    score_range = max_score - min_score
    
    # Create grade boundaries based on the distribution
    grade_boundaries = {
        'A': min_score + (score_range * 0.8),  # Top 20%
        'B': min_score + (score_range * 0.6),  # 60-80%
        'C': min_score + (score_range * 0.4),  # 40-60%
        'D': min_score + (score_range * 0.2),  # 20-40%
        'F': min_score                         # Bottom 20%
    }
    
    # Assign letter grades based on these boundaries
    letter_grades = pd.Series(index=overall_scores.index)
    letter_grades.loc[overall_scores >= grade_boundaries['A']] = 'A'
    letter_grades.loc[(overall_scores >= grade_boundaries['B']) & (overall_scores < grade_boundaries['A'])] = 'B'
    letter_grades.loc[(overall_scores >= grade_boundaries['C']) & (overall_scores < grade_boundaries['B'])] = 'C'
    letter_grades.loc[(overall_scores >= grade_boundaries['D']) & (overall_scores < grade_boundaries['C'])] = 'D'
    letter_grades.loc[overall_scores < grade_boundaries['D']] = 'F'
    
    print(f"\nGrade Distribution Boundaries:")
    print(f"A: >= {grade_boundaries['A']:.1f}")
    print(f"B: {grade_boundaries['B']:.1f} - {grade_boundaries['A']:.1f}")
    print(f"C: {grade_boundaries['C']:.1f} - {grade_boundaries['B']:.1f}")
    print(f"D: {grade_boundaries['D']:.1f} - {grade_boundaries['C']:.1f}")
    print(f"F: < {grade_boundaries['D']:.1f}")
    
    return overall_scores, letter_grades

def generate_quality_report(df, school_name_col, zip_col, overall_scores, letter_grades, df_scores):
    """Generate the final quality report with schools, ratings, and metrics used."""
    # Create the base DataFrame for the report
    report = pd.DataFrame({
        'School Name': df[school_name_col] if school_name_col else "Unknown",
        'Quality Score': overall_scores.round(1),
        'Letter Grade': letter_grades
    })
    
    # Add ZIP code if found
    if zip_col:
        report['ZIP Code'] = df[zip_col]
    
    # Add category scores (excluding Advanced Score as requested)
    for category in METRIC_WEIGHTS.keys():
        if category == 'advanced':  # Skip the Advanced Score as requested
            continue
            
        score_col = f'{category}_score'
        if score_col in df_scores.columns:
            report[f'{category.capitalize()} Score'] = df_scores[score_col].round(2) * 100
    
    # Calculate strengths and weaknesses
    category_columns = [f'{category.capitalize()} Score' for category in METRIC_WEIGHTS.keys() 
                        if f'{category}_score' in df_scores.columns and category != 'advanced']
    
    if category_columns:
        # Find top 2 categories for each school
        report['Strengths'] = report[category_columns].apply(
            lambda x: ', '.join([col.split()[0] for col in category_columns 
                                if pd.notna(x[col]) and x[col] >= x[category_columns].median()]), 
            axis=1
        )
        
        # Find bottom 2 categories for each school
        report['Areas for Improvement'] = report[category_columns].apply(
            lambda x: ', '.join([col.split()[0] for col in category_columns 
                                if pd.notna(x[col]) and x[col] < x[category_columns].median()]), 
            axis=1
        )
    
    return report

def main():
    # Load the merged school data
    data_file = "data/properly_merged_school_data.csv"
    df = load_data(data_file)
    
    # Identify key columns
    school_name_col, zip_col, metric_columns = identify_key_columns(df)
    
    # Calculate category scores
    category_scores, df_scores = calculate_category_scores(df, metric_columns)
    
    # Calculate overall quality scores
    overall_scores, letter_grades = calculate_overall_scores(df_scores, category_scores)
    
    # Generate the final report
    quality_report = generate_quality_report(df, school_name_col, zip_col, 
                                            overall_scores, letter_grades, df_scores)
    
    # Save the report
    output_file = "data/school_quality_ratings.csv"
    quality_report.to_csv(output_file, index=False)
    print(f"Saved quality ratings for {len(quality_report)} schools to {output_file}")
    
    # Print summary statistics
    grade_counts = quality_report['Letter Grade'].value_counts()
    print("\nGrade Distribution:")
    for grade in ['A', 'B', 'C', 'D', 'F']:
        if grade in grade_counts:
            print(f"{grade}: {grade_counts[grade]} schools ({grade_counts[grade]/len(quality_report)*100:.1f}%)")
    
    # Display top and bottom schools
    print("\nTop 5 Schools:")
    top_schools = quality_report.nlargest(5, 'Quality Score')
    for idx, row in top_schools.iterrows():
        print(f"{row['School Name']}: {row['Quality Score']:.1f} ({row['Letter Grade']})")
    
    print("\nBottom 5 Schools:")
    bottom_schools = quality_report.nsmallest(5, 'Quality Score')
    for idx, row in bottom_schools.iterrows():
        print(f"{row['School Name']}: {row['Quality Score']:.1f} ({row['Letter Grade']})")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()