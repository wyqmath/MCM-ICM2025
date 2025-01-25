"""
奥林匹克国家实体标准化系统:支持动态政权更迭映射
"""
import re
import pandas as pd
from Levenshtein import ratio as lev_ratio
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import tqdm

# 政权更迭映射表（核心知识库）
HISTORICAL_MAPPING = {
    # Modern countries
    'United States': ('USA', None),
    'Great Britain': ('GBR', None),
    'Trinidad and Tobago': ('TTO', None),
    'Puerto Rico': ('PUR', None),
    'Dominican Republic': ('DOM', None),
    'Costa Rica': ('CRC', None),
    'Virgin Islands': ('ISV', None),
    'Saudi Arabia': ('KSA', None),
    'Czech Republic': ('CZE', None),
    'Mixed team': ('MIX', None),
    'United Kingdom': ('GBR', None),
    'South Africa': ('RSA', None),
    'New Zealand': ('NZL', None),
    'Soviet Union': ('RUS', 1991),
    'East Germany': ('GER', 1990),
    'West Germany': ('GER', 1990),
    'Czechoslovakia': ('CZE', 1993),
    'Yugoslavia': ('SRB', 2003),
    'Bohemia': ('CZE', 1918),
    
    # Historical entities
    'Russian Empire': ('RUS', 1917),
    'United Team of Germany': ('GER', 1968),
    'British West Indies': ('JAM', 1962),  # Most athletes from Jamaica
    
    # Additional mappings for missing IOC codes
    'Ivory Coast': ('CIV', None),
    'Côte d\'Ivoire': ('CIV', None),
    'Chinese Taipei': ('TPE', None),
    'Netherlands Antilles': ('AHO', None),
    'Unified Team': ('EUN', 1992),
    'Independent Olympic Athletes': ('IOA', None),
    'Congo (Brazzaville)': ('CGO', None),
    'Congo (Kinshasa)': ('COD', None),
    'Federated States of Micronesia': ('FSM', None),
    'Marshall Islands': ('MHL', None),
    'Saint Vincent and the Grenadines': ('VIN', None),
    'Cape Verde': ('CPV', None),
    'Cabo Verde': ('CPV', None),
    'Central African Republic': ('CAF', None),
    'South Sudan': ('SSD', None),
    'North Macedonia': ('MKD', None),
    'Republic of Moldova': ('MDA', None),
    'Brunei Darussalam': ('BRN', None),
    'Serbia and Montenegro': ('SCG', 2006),
    'Saint Lucia': ('LCA', None),
    'San Marino': ('SMR', None),
    'Central African Republic': ('CAF', None),
    'São Tomé and Príncipe': ('STP', None),
    'Sao Tome & Principe': ('STP', None),
    'Sri Lanka': ('LKA', None),
    'El Salvador': ('ESA', None),
    'Saint Kitts and Nevis': ('SKN', None),
    'Bosnia & Herzegovina': ('BIH', None),
    'Bosnia and Herzegovina': ('BIH', None),
    'Lao PDR': ('LAO', None),
    'East Timor': ('TLS', None),
    'Côte d\'Ivoire': ('CIV', None),
    'Cote d\'Ivoire': ('CIV', None),
    'Cayman Islands': ('CAY', None),
    'Antigua and Barbuda': ('ANT', None),
    'São Tomé and Príncipe': ('STP', None),
    'Sao Tome & Principe': ('STP', None),
    'Independent Olympic Participants': ('IOP', None),
    'Refugee Olympic Team': ('ROT', None),
    'Individual Olympic Athletes': ('IOA', None),
    'Refugee Olympic Athletes': ('ROA', None),
    'São Tomé and Príncipe': ('STP', None),
    'Sao Tome & Principe': ('STP', None),
    'Timor Leste': ('TLS', None),
    'East Timor': ('TLS', None),
    'Centr Afric Re': ('CAF', None),
    'Central African Republic': ('CAF', None),
    'St Kitts and Nevis': ('SKN', None),
    'Saint Kitts and Nevis': ('SKN', None),
    'São Tomé and Príncipe': ('STP', None),
    'Sao Tome & Principe': ('STP', None),
    'St. Lucia': ('LCA', None),
    'Saint Vincent and the Grenadines': ('VIN', None),
    'United Arab Emirates': ('UAE', None),
    'UA Emirates': ('UAE', None),
    'Cook Islands': ('COK', None),
    'Solomon Islands': ('SOL', None),
    'Tuvalu': ('TUV', None),
    'Kiribati': ('KIR', None),
    'Palau': ('PLW', None),
    'Nauru': ('NRU', None),
    'Samoa': ('SAM', None),
    'Tonga': ('TGA', None),
    'Vanuatu': ('VAN', None),
    'Comoros': ('COM', None),
    'Seychelles': ('SEY', None),
    'Maldives': ('MDV', None),
    'Bhutan': ('BHU', None),
    'Laos': ('LAO', None),
    'Myanmar': ('MYA', None),
    'Cambodia': ('CAM', None),
    'Afghanistan': ('AFG', None),
    'Yemen': ('YEM', None),
    'Oman': ('OMA', None),
    'Qatar': ('QAT', None),
    'Bahrain': ('BRN', None),
    'Kuwait': ('KUW', None),
    'Jordan': ('JOR', None),
    'Lebanon': ('LBN', None),
    'Syria': ('SYR', None),
    'Iraq': ('IRQ', None),
    'Iran': ('IRI', None),
    'IR Iran': ('IRI', None),
    'Türkiye': ('TUR', None),
    'Azerbaijan': ('AZE', None),
    'Georgia': ('GEO', None),
    'Armenia': ('ARM', None),
    'Kazakhstan': ('KAZ', None),
    'Uzbekistan': ('UZB', None),
    'Turkmenistan': ('TKM', None),
    'Tajikistan': ('TJK', None),
    'Kyrgyzstan': ('KGZ', None),
    'Mongolia': ('MGL', None),
    'North Korea': ('PRK', None),
    'DPR Korea': ('PRK', None),
    'South Korea': ('KOR', None),
    'Korea': ('KOR', None),
    'Hong Kong': ('HKG', None),
    'Hong Kong, China': ('HKG', None),
    'Macau': ('MAC', None),
    'Taiwan': ('TPE', None),
    'Formosa': ('TPE', None),
    'Palestine': ('PLE', None),
    'Western Sahara': ('ESH', None),
    'Somalia': ('SOM', None),
    'Djibouti': ('DJI', None),
    'Eritrea': ('ERI', None),
    'Sudan': ('SUD', None),
    'South Sudan': ('SSD', None),
    'Chad': ('CHA', None),
    'Niger': ('NIG', None),
    'Mali': ('MLI', None),
    'Burkina Faso': ('BUR', None),
    'Benin': ('BEN', None),
    'Togo': ('TOG', None),
    'Ghana': ('GHA', None),
    'Côte d\'Ivoire': ('CIV', None),
    'Liberia': ('LBR', None),
    'Sierra Leone': ('SLE', None),
    'Guinea': ('GUI', None),
    'Guinea-Bissau': ('GBS', None),
    'Gambia': ('GAM', None),
    'The Gambia': ('GAM', None),
    'Senegal': ('SEN', None),
    'Mauritania': ('MTN', None),
    'Cape Verde': ('CPV', None),
    'Cabo Verde': ('CPV', None),
    'São Tomé and Príncipe': ('STP', None),
    'Sao Tome & Principe': ('STP', None),
    'Equatorial Guinea': ('GEQ', None),
    'Gabon': ('GAB', None),
    'Republic of the Congo': ('CGO', None),
    'Democratic Republic of the Congo': ('COD', None),
    'DR Congo': ('COD', None),
    'Central African Republic': ('CAF', None),
    'Cameroon': ('CMR', None),
    'Nigeria': ('NGR', None),
    'Niger': ('NIG', None),
    'Chad': ('CHA', None),
    'Sudan': ('SUD', None),
    'South Sudan': ('SSD', None),
    'Eritrea': ('ERI', None),
    'Ethiopia': ('ETH', None),
    'Somalia': ('SOM', None),
    'Djibouti': ('DJI', None),
    'Kenya': ('KEN', None),
    'Uganda': ('UGA', None),
    'Tanzania': ('TAN', None),
    'Rwanda': ('RWA', None),
    'Burundi': ('BDI', None),
    'Malawi': ('MAW', None),
    'Zambia': ('ZAM', None),
    'Zimbabwe': ('ZIM', None),
    'Mozambique': ('MOZ', None),
    'Madagascar': ('MAD', None),
    'Comoros': ('COM', None),
    'Seychelles': ('SEY', None),
    'Mauritius': ('MRI', None),
    'Swaziland': ('SWZ', None),
    'Eswatini': ('SWZ', None),
    'Lesotho': ('LES', None),
    'Botswana': ('BOT', None),
    'Namibia': ('NAM', None),
    'South Africa': ('RSA', None),
    'Angola': ('ANG', None),
    
    # Common data issues
    'nan': (None, None),
    'Unknown': (None, None)
}

SPECIAL_TEAM_MAPPING = {
    # Boat clubs and sports teams
    r'.*Boat Club': 'UNK-TEAM',
    r'.*Rowing Club': 'UNK-TEAM',
    r'.*Athletic Club': 'UNK-TEAM',
    r'.*Turnverein': 'UNK-TEAM',
    
    # Handle special characters
    r'\s+': ' ',  # Normalize whitespace
    r'[^\x00-\x7F]+': '',  # Remove non-ASCII
    r'\u00A0': ' '  # Replace non-breaking spaces
}

def clean_entity_name(name):
    """Clean and normalize entity names"""
    if pd.isna(name):
        return None
        
    # Convert to string and clean
    name = str(name)
    
    # Apply special character cleaning
    for pattern, replacement in SPECIAL_TEAM_MAPPING.items():
        name = re.sub(pattern, replacement, name)
    
    # Strip and normalize
    name = name.strip()
    if not name:
        return None
        
    return name

def historical_adjustment(raw_name, year):
    """历史政权时间轴对齐"""
    # Clean and normalize the name first
    clean_name = clean_entity_name(raw_name)
    if not clean_name:
        return None
    
    # Check for special cases first
    for pattern, (code, split_year) in HISTORICAL_MAPPING.items():
        # Exact match for special cases
        if pattern.lower() == clean_name.lower():
            return code
            
        # Historical entity matching
        if pattern in clean_name:
            if split_year is None or year > split_year:
                return code
                
    # Check for team/club patterns
    for pattern, code in SPECIAL_TEAM_MAPPING.items():
        if re.search(pattern, clean_name, re.IGNORECASE):
            return code
            
    return None

def fuzzy_match(name, ref_dict, threshold=0.85):
    """Enhanced fuzzy matching with multiple metrics"""
    if not name or pd.isna(name):
        return None
        
    # Normalize input
    name = name.lower().strip()
    if not name:
        return None
        
    # Track best matches
    matches = []
    
    # Calculate multiple similarity metrics
    for ref_name, code in ref_dict.items():
        ref_lower = ref_name.lower()
        
        # Basic Levenshtein ratio
        lev_score = lev_ratio(name, ref_lower)
        
        # Additional metrics
        jaro_score = fuzz.ratio(name, ref_lower)
        partial_ratio = fuzz.partial_ratio(name, ref_lower)
        
        # Combined score (weighted average)
        combined_score = (lev_score * 0.5) + (jaro_score * 0.3) + (partial_ratio * 0.2)
        
        if combined_score >= threshold:
            matches.append({
                'name': ref_name,
                'code': code,
                'score': combined_score,
                'lev_score': lev_score,
                'jaro_score': jaro_score,
                'partial_ratio': partial_ratio
            })
    
    # Return best match if any
    if matches:
        # Sort by combined score descending
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[0]['code']
        
    return None

def calculate_metrics(df):
    """Calculate and log validation metrics"""
    total = len(df)
    matched = df[df['IOC_Code'] != 'UNK'].shape[0]
    accuracy = matched / total
    
    # Calculate error types
    historical_errors = df[df['Country'].isin(HISTORICAL_MAPPING.keys()) & (df['IOC_Code'] == 'UNK')]
    team_errors = df[df['Country'].str.contains('|'.join(SPECIAL_TEAM_MAPPING.keys())) & (df['IOC_Code'] == 'UNK')]
    
    print(f"\nValidation Metrics:")
    print(f"- Total Entities: {total}")
    print(f"- Matched: {matched} ({accuracy:.2%})")
    print(f"- Historical Errors: {len(historical_errors)}")
    print(f"- Team/Club Errors: {len(team_errors)}")
    
    return {
        'total': total,
        'matched': matched,
        'accuracy': accuracy,
        'historical_errors': len(historical_errors),
        'team_errors': len(team_errors)
    }

def process_entities():
    """主处理流程"""
    # 加载GDP数据获取IOC代码
    gdp_data = pd.read_csv('world-gdp-data-with-ioc.csv')
    ioc_mapping = gdp_data[['country', 'IOC_Code']].drop_duplicates()
    ioc_mapping = ioc_mapping.dropna(subset=['IOC_Code'])
    ioc_ref = ioc_mapping.set_index('country')['IOC_Code'].to_dict()

    # 加载多源数据
    medal_counts = pd.read_csv('summerOly_medal_counts.csv')
    hosts = pd.read_csv('summerOly_hosts.csv')
    athletes = pd.read_csv('summerOly_athletes.csv')

    # 实体抽取
    entities = pd.concat([
        medal_counts['NOC'].astype(str),
        hosts['Host'].str.extract(r',\s*(.+)$')[0].astype(str),  # 提取主办国
        athletes['Team'].str.replace(r'-\d+$', '', regex=True).astype(str)  # 处理队伍编号
    ]).drop_duplicates().reset_index(drop=True)

    # 消歧处理
    mapping = []
    for raw_name in tqdm(entities, desc='Processing'):
        # 时空上下文处理
        historical_code = historical_adjustment(raw_name, 2024)
        if historical_code:
            mapping.append((raw_name, historical_code))
            continue

        # 直接匹配IOC代码
        if raw_name in ioc_ref:
            mapping.append((raw_name, ioc_ref[raw_name]))
            continue
            
        # Fallback to manual mapping
        clean_name = clean_entity_name(raw_name)
        if clean_name in HISTORICAL_MAPPING:
            code = HISTORICAL_MAPPING[clean_name][0]
            mapping.append((raw_name, code))
            continue
            
        # Final fallback to UNK
        mapping.append((raw_name, 'UNK'))

    # 生成结果
    df = pd.DataFrame(mapping, columns=['Country', 'IOC_Code'])
    
    # 处理剩余的空值
    df['IOC_Code'] = df['IOC_Code'].replace({
        None: 'UNK',
        '': 'UNK',
        'nan': 'UNK'
    })

    # 验证模块
    sample = df.sample(50, random_state=42)
    accuracy = sample[sample['IOC_Code'] != 'UNK'].shape[0] / 50
    print(f"Validation Accuracy: {accuracy:.2%}")
    
    # 输出未匹配的实体
    unmatched = df[df['IOC_Code'] == 'UNK']
    if not unmatched.empty:
        print("\nUnmatched Entities:")
        print(unmatched['Country'].unique())
    
    return df

# 执行处理流程
if __name__ == "__main__":
    process_entities()