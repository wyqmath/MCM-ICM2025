import pandas as pd
import numpy as np
import networkx as nx

# Load data
medals = pd.read_csv('summerOly_medal_counts.csv')
hosts = pd.read_csv('summerOly_hosts.csv') 
country_map = pd.read_csv('country_mapping.csv')

# Preprocess data
# Standardize country names using mapping
medals = medals.merge(country_map, left_on='NOC', right_on='RawName', how='left')
medals['Country'] = medals['IOC_Code'].fillna(medals['NOC'])

# Extract host countries
hosts['HostCountry'] = hosts['Host'].str.extract(r'([A-Za-z]+)')[0]
hosts = hosts.merge(country_map, left_on='HostCountry', right_on='RawName', how='left')
hosts['HostCode'] = hosts['IOC_Code'].fillna(hosts['HostCountry'])

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