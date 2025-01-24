import pandas as pd
import numpy as np
from math import log, sqrt, exp
from datetime import datetime

# Load data
medals = pd.read_csv('summerOly_medal_counts.csv')
gdp_pop = pd.read_csv('world-gdp-data.csv')

# Clean and merge data
medals['NOC'] = medals['NOC'].str.strip()
gdp_pop['country'] = gdp_pop['country'].str.strip()

# Merge on country name
merged = pd.merge(medals, gdp_pop, left_on='NOC', right_on='country', how='left')

# Calculate current year
current_year = datetime.now().year

# Define decay constant
lambda_val = 0.05

def calculate_weight(row):
    try:
        # Get medal count (using total medals)
        medals_tc = row['Total']
        
        # Get GDP and population
        gdp_tc = row['imfGDP']
        population_tc = row['population']
        
        # Get year difference
        year_diff = current_year - row['Year']
        
        # Calculate weight
        numerator = log(medals_tc + 1)
        denominator = log(gdp_tc) * sqrt(population_tc)
        decay_factor = exp(-lambda_val * year_diff)
        
        return (numerator / denominator) * decay_factor
    except:
        return np.nan

# Apply calculation
merged['Weight'] = merged.apply(calculate_weight, axis=1)

# Select relevant columns and save
result = merged[['NOC', 'Year', 'Weight']]
result.to_csv('dynamic_weight_matrix_2028proj.csv', index=False)