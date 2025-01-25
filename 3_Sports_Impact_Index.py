import pandas as pd
import numpy as np
import networkx as nx

# Load data
medals = pd.read_csv('summerOly_medal_counts.csv')
hosts = pd.read_csv('summerOly_hosts.csv') 
country_map = pd.read_csv('country_mapping.csv')

# Check initial NaN values
print("初始 medals NaN 值数量:", medals.isna().sum().sum())
print("初始 hosts NaN 值数量:", hosts.isna().sum().sum())
print("初始 country_map NaN 值数量:", country_map.isna().sum().sum())

# Check for rows with NaN values in country_map
nan_rows = country_map[country_map.isna().any(axis=1)]

# Preprocess data
# Standardize country names using mapping
medals = medals.merge(country_map, left_on='NOC', right_on='RawName', how='left')
print("合并后 medals NaN 值数量:", medals.isna().sum().sum())

medals['Country'] = medals['IOC_Code'].fillna(medals['NOC'])
print("填充后 medals NaN 值数量:", medals.isna().sum().sum())

# Extract and standardize host countries
def extract_host_country(host_str):
    # Skip cancelled years
    if 'Cancelled' in host_str:
        return None
    
    # Clean up non-breaking spaces and special characters
    host_str = host_str.replace('\xa0', ' ').strip()
    
    # Handle postponed year
    if 'postponed' in host_str.lower():
        # Extract the main country name before parentheses
        match = re.search(r'^([A-Za-z\s]+)\s*\(', host_str)
        if match:
            return match.group(1).strip()
    
    # Handle common patterns
    patterns = [
        r',\s*([A-Za-z\s]+)$',  # "City, Country"
        r'\(([A-Za-z\s]+)\)$',  # "Country (City)"
        r'\[([A-Za-z\s]+)\]$',  # "Country [City]"
        r'([A-Za-z\s]+)\s*\(',  # "Country (additional info)"
        r'([A-Za-z\s]+)$'       # Just country name
    ]
    
    for pattern in patterns:
        match = re.search(pattern, host_str)
        if match:
            country = match.group(1).strip()
            # Handle special cases
            if 'West Germany' in country:
                return 'Germany'
            return country
    return None

# Filter out cancelled and future Olympics
current_year = 2025  # Update this as needed
hosts = hosts[~hosts['Host'].str.contains('Cancelled')]
hosts = hosts[hosts['Year'] <= current_year]

from fuzzywuzzy import process

# Add comprehensive manual mappings for special cases
manual_mappings = {
    'usa': 'United States',
    'uk': 'United Kingdom',
    'ussr': 'Soviet Union',
    'russia': 'Russian Federation',
    'west germany': 'Germany',
    'great britain': 'United Kingdom',
    'united states': 'United States',
    'soviet union': 'Russia',
    'russian federation': 'Russia',
    'czech republic': 'Czechia',
    'czechoslovakia': 'Czechia',
    'yugoslavia': 'Serbia',
    'federal republic of yugoslavia': 'Serbia',
    'serbia and montenegro': 'Serbia',
    'unified team': 'Russia',
    'east germany': 'Germany',
    'west germany': 'Germany',
    'united team of germany': 'Germany',
    'republic of china': 'China',
    'taiwan': 'Chinese Taipei',
    'formosa': 'Chinese Taipei',
    'hong kong': 'Hong Kong, China',
    'macau': 'China',
    'north korea': 'Korea, Democratic People\'s Republic of',
    'south korea': 'Korea, Republic of',
    'korea, republic of': 'Korea, Republic of',
    'korea republic of': 'Korea, Republic of',
    'ivory coast': 'Côte d\'Ivoire',
    'cote d\'ivoire': 'Côte d\'Ivoire',
    'burma': 'Myanmar',
    'russia': 'Russian Federation',
    'russian federation': 'Russia',
    'united states': 'United States',
    'usa': 'United States',
    'united states of america': 'United States',
    'united kingdom': 'United Kingdom',
    'uk': 'United Kingdom',
    'great britain': 'United Kingdom',
    'germany': 'Germany',
    'france': 'France',
    'italy': 'Italy',
    'japan': 'Japan',
    'china': 'China',
    'australia': 'Australia',
    'canada': 'Canada',
    'brazil': 'Brazil',
    'spain': 'Spain',
    'netherlands': 'Netherlands',
    'sweden': 'Sweden',
    'greece': 'Greece',
    'finland': 'Finland',
    'belgium': 'Belgium',
    'mexico': 'Mexico',
    'cape verde': 'Cabo Verde',
    'east timor': 'Timor-Leste',
    'swaziland': 'Eswatini',
    'macedonia': 'North Macedonia'
}

# Preprocess host country names
import re
hosts['HostCountry'] = hosts['Host'].apply(extract_host_country)

# Standardize names before mapping
country_map['RawName'] = country_map['RawName'].str.strip().str.lower()
# Apply manual mappings to country_map to ensure consistency
country_map['RawName'] = country_map['RawName'].apply(
    lambda x: manual_mappings.get(x, x)
)
hosts['HostCountry'] = hosts['HostCountry'].str.strip().str.lower()

# Apply manual mappings
hosts['HostCountry'] = hosts['HostCountry'].apply(
    lambda x: manual_mappings.get(x, x)
)

# Fuzzy matching for remaining unmatched countries
def fuzzy_match_country(country_name):
    if pd.isna(country_name):
        return None
        
    # Get list of all possible country names
    possible_names = country_map['RawName'].unique()
    
    # Find best match using fuzzywuzzy
    match, score = process.extractOne(country_name, possible_names)
    
    # Print debug information for unmatched countries
    if score <= 85:
        print(f"Unmatched country: {country_name} (best match: {match} with score {score})")
        return None
    
    return match

# Apply fuzzy matching to remaining unmatched countries
hosts['HostCountry'] = hosts['HostCountry'].apply(
    lambda x: fuzzy_match_country(x) if pd.isna(x) else x
)

# Print detailed mapping information for all hosts
#print("\nDetailed host country mapping information:")
for _, row in hosts.iterrows():
    extracted = extract_host_country(row['Host'])
    mapped = manual_mappings.get(extracted.lower(), None) if extracted else None
    fuzzy = fuzzy_match_country(extracted) if extracted else None

    '''
    print(f"\nYear: {row['Year']}")
    print(f"Original Host: {row['Host']}")
    print(f"Extracted country: {extracted}")
    print(f"Manual mapping: {mapped}")
    print(f"Fuzzy match: {fuzzy}")
    print(f"Final HostCountry: {row['HostCountry']}")
    print("-" * 40)
    '''
# Now calculate and print NaN counts
#print("\nFinal NaN counts:")

# Print unmapped host countries for debugging
unmapped_hosts = hosts[hosts['HostCountry'].isna()]
if not unmapped_hosts.empty:
    print("\nUnmapped host countries:")
    print(unmapped_hosts[['Year', 'Host']])

# Merge with mapping
hosts = hosts.merge(country_map, left_on='HostCountry', right_on='RawName', how='left')
print("合并后 hosts NaN 值数量:", hosts.isna().sum().sum())

# Fill missing mappings
hosts['HostCode'] = hosts['IOC_Code'].fillna(hosts['HostCountry'])
print("填充后 hosts NaN 值数量:", hosts.isna().sum().sum())

# Calculate project-country association scores
scores = []
for year, year_medals in medals.groupby('Year'):
    # Get host country for this year
    host = hosts.loc[hosts['Year'] == year, 'HostCode'].values[0]
    
    # Calculate scores for each country
    for _, row in year_medals.iterrows():
        country = row['Country']
        medal_score = row['Total'] / row['Rank']
        host_score = 1 if country == host else 0.1  # Give host country 10x weight
        score = medal_score * host_score
        scores.append((year, country, score))

# Create influence network
scores_df = pd.DataFrame(scores, columns=['Year', 'Country', 'Score'])
scores_df = scores_df.groupby('Country')['Score'].sum().reset_index()

# Create graph and calculate PageRank
G = nx.DiGraph()
for _, row in scores_df.iterrows():
    G.add_edge('Sports', row['Country'], weight=row['Score'])
    
pagerank = nx.pagerank(G, weight='weight')

# Save results
results = pd.DataFrame.from_dict(pagerank, orient='index', columns=['Score'])
results = results[results.index != 'Sports']  # Remove sports node
results.to_csv('sport_pagerank_scores.csv')