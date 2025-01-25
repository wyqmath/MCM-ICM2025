import csv
from collections import defaultdict

# Load country mappings
country_map = defaultdict(str)
with open('country_mapping.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Create normalized key by stripping whitespace and making lowercase
        clean_name = row['RawName'].strip().casefold()
        country_map[clean_name] = row['IOC_Code']

# Function to get IOC code
def get_ioc_code(country):
    clean_country = country.strip().casefold()
    
    # Try direct match
    code = country_map.get(clean_country, '')
    
    # Fallback: Try without parentheticals
    if not code and '(' in country:
        base_name = country.split('(')[0].strip()
        code = country_map.get(base_name.casefold(), '')
    
    # Fallback: Try common alternative names
    if not code:
        alt_names = {
            'United Kingdom': 'GBR',
            'DR Congo': 'COD',
            'Republic of the Congo': 'COG',
            'Ivory Coast': 'CIV'
        }
        code = alt_names.get(country, 'N/A')
    
    return code

# Process GDP data
output_rows = []
with open('world-gdp-data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ['IOC_Code']
    
    for row in reader:
        country = row['country'].strip()
        row['IOC_Code'] = get_ioc_code(country)
        output_rows.append(row)

# Write GDP output
with open('world-gdp-data-with-ioc.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print("Successfully added IOC codes to world-gdp-data-with-ioc.csv")

# 处理夏季奥运会奖牌数据
with open('summerOly_medal_counts.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ['IOC_Code']
    
    output_rows = []
    for row in reader:
        country = row['NOC'].strip()  # 假设NOC列包含国家名称
        row['IOC_Code'] = get_ioc_code(country)
        output_rows.append(row)

# 写入奖牌数据输出
with open('summerOly_medal_counts_with_ioc.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print("成功将IOC代码添加到summerOly_medal_counts_with_ioc.csv")

# 处理夏季奥运会主办城市数据
with open('summerOly_hosts.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ['IOC_Code']
    
    output_rows = []
    for row in reader:
        host = row['Host'].strip()
        
        # 检查是否被取消
        if 'Cancelled' in host:
            row['IOC_Code'] = 'N/A'
        else:
            # 提取国家部分（逗号后）
            if ',' in host:
                country = host.split(',')[-1].strip()
                row['IOC_Code'] = get_ioc_code(country)
            else:
                # 如果没有逗号，尝试整个字段匹配
                row['IOC_Code'] = get_ioc_code(host)
        
        output_rows.append(row)

# 写入主办城市数据输出
with open('summerOly_hosts_with_ioc.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print("成功将IOC代码添加到summerOly_hosts_with_ioc.csv")