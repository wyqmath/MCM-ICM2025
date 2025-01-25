import pandas as pd
import numpy as np
from math import log, sqrt, exp
from datetime import datetime

# Constants
DECAY_CONSTANT = 0.05  # Optimal decay coefficient (λ)
MIN_GDP = 1e-6  # Minimum GDP to avoid log(0)
MIN_POPULATION = 1  # Minimum population to avoid division by zero
EPSILON = 1e-10  # Small value for numerical stability

def validate_and_clean_data(df):
    """Validate and clean input data"""
    required_columns = {'Total', 'imfGDP', 'population', 'Year'}
    
    # Check for required columns
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
    
    # Clean data
    df = df.copy()
    
    # Fill missing GDP with median of country's historical GDP, fallback to global median
    if df['imfGDP'].isnull().any():
        print("Filling missing GDP values:")
        global_gdp_median = df['imfGDP'].median()
        df['imfGDP'] = df.groupby('NOC')['imfGDP'].transform(
            lambda x: x.fillna(x.median() if not x.isnull().all() else global_gdp_median))
        print(f" - Filled {df['imfGDP'].isnull().sum()} remaining missing GDP values with global median")
    
    # Fill missing population with median of country's historical population, fallback to global median
    if df['population'].isnull().any():
        print("Filling missing population values:")
        global_pop_median = df['population'].median()
        df['population'] = df.groupby('NOC')['population'].transform(
            lambda x: x.fillna(x.median() if not x.isnull().all() else global_pop_median))
        print(f" - Filled {df['population'].isnull().sum()} remaining missing population values with global median")
    
    # Handle remaining missing values
    if df[list(required_columns)].isnull().any().any():
        print("Warning: Dropping rows with remaining missing values")
        df = df.dropna(subset=list(required_columns))
    
    # Check for negative values
    if (df[['Total', 'imfGDP', 'population']] < 0).any().any():
        raise ValueError("Negative values found in data")
    
    return df

def calculate_weight_vectorized(df, current_year, lambda_val):
    """
    Calculate dynamic national power weights using vectorized operations
    
    Parameters:
    df (pd.DataFrame): Input data containing medals, GDP, population
    current_year (int): Current year for time decay calculation
    lambda_val (float): Decay constant
    
    Returns:
    pd.Series: Calculated weights
    """
    # Calculate time differences
    year_diffs = current_year - df['Year']
    
    # Handle edge cases for GDP and population
    gdp_values = np.maximum(df['imfGDP'], MIN_GDP)
    population_values = np.maximum(df['population'], MIN_POPULATION)
    
    # Calculate components
    medal_term = np.log1p(df['Total'])  # log(medals + 1)
    gdp_term = np.log(gdp_values)
    population_term = np.sqrt(population_values)
    decay_term = np.exp(-lambda_val * year_diffs)
    
    # Calculate final weights with numerical stability
    denominator = gdp_term * population_term + EPSILON
    weights = (medal_term / denominator) * decay_term
    
    return weights

def main():
    # Load and clean data
    medals = pd.read_csv('summerOly_medal_counts.csv')
    gdp_pop = pd.read_csv('world-gdp-data.csv')
    
    # Clean and merge data
    medals['NOC'] = medals['NOC'].str.strip()
    gdp_pop['country'] = gdp_pop['country'].str.strip()
    merged = pd.merge(medals, gdp_pop, left_on='NOC', right_on='country', how='left')
    
    # Validate and clean input data
    merged = validate_and_clean_data(merged)
    
    # Calculate current year
    current_year = datetime.now().year
    
    # Calculate weights using vectorized approach
    weights = calculate_weight_vectorized(merged, current_year, DECAY_CONSTANT)
    
    # Store results
    result = merged[['NOC', 'Year']].copy()
    result['Weight'] = weights
    result.to_csv('dynamic_weight_matrix_2028proj.csv', index=False)

# 检查生成的文件中是否有NaN值
def check_nan_in_file(file_path):
    df = pd.read_csv(file_path)
    if df.isnull().values.any():
        print(f"警告：文件 {file_path} 中包含NaN值。")
    else:
        print(f"文件 {file_path} 中不包含NaN值。")

if __name__ == '__main__':
    main()
    check_nan_in_file('dynamic_weight_matrix_2028proj.csv')
    print("输出完毕！")