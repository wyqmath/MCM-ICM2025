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
    
    # Special cases
    'Mixed team': ('MIX', None),
    'Independent Olympic Participants': ('IOP', None),
    'Refugee Olympic Team': ('EOR', None),
    'Individual Olympic Athletes': ('IOA', None),
    
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

def build_ioc_reference():
    """构建IOC标准编码库"""
    # 从各数据集中提取标准NOC编码
    athletes = pd.read_csv('summerOly_athletes.csv')
    noc_ref = athletes[['Team', 'NOC']].drop_duplicates()
    return noc_ref.set_index('Team')['NOC'].to_dict()

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
    historical_errors = df[df['RawName'].isin(HISTORICAL_MAPPING.keys()) & (df['IOC_Code'] == 'UNK')]
    team_errors = df[df['RawName'].str.contains('|'.join(SPECIAL_TEAM_MAPPING.keys())) & (df['IOC_Code'] == 'UNK')]
    
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
    # 初始化参考库
    ioc_ref = build_ioc_reference()

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

        # 模糊匹配
        code = fuzzy_match(raw_name, ioc_ref)
        mapping.append((raw_name, code if code else 'UNK'))

    # 生成映射表
    df = pd.DataFrame(mapping, columns=['RawName', 'IOC_Code'])
    df.to_csv('country_mapping.csv', index=False)

    # 验证模块
    sample = df.sample(50, random_state=42)
    accuracy = sample[sample['IOC_Code'] != 'UNK'].shape[0] / 50
    print(f"Validation Accuracy: {accuracy:.2%}")

    return df

# 执行处理流程
if __name__ == "__main__":
    process_entities()

# Created/Modified files during execution:
print("country_mapping.csv")