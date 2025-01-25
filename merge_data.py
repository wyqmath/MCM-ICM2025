import pandas as pd
import re

def extract_country(host_entry):
    """
    从 Host 列中提取国家名称。
    """
    if isinstance(host_entry, str):
        # 去除所有括号内的内容
        host_entry = re.sub(r'\(.*?\)', '', host_entry)
        # 去除前导和尾随空格（包括非断行空格）
        host_entry = host_entry.strip()
        # 分割逗号，提取最后一部分作为国家名称
        parts = host_entry.split(',')
        if len(parts) >= 2:
            country = parts[-1].strip()
            return country
        else:
            return host_entry.strip()
    return host_entry

def merge_data():
    """
    合并所需的数据文件，生成一个格式化的 CSV 文件供 5_model_validation.py 使用。
    添加调试信息以帮助查找 'IOC_Code' 缺失的问题。
    """
    try:
        # 1. 读取国家映射文件，包含国家名称到 IOC_Code 的映射
        print("读取 'country_mapping.csv' 文件...")
        country_mapping = pd.read_csv('country_mapping.csv')
        print(f"country_mapping 数据行数: {country_mapping.shape[0]}")
        print(f"country_mapping 列: {list(country_mapping.columns)}")
        
        # 确保必要的列存在
        if not {'RawName', 'IOC_Code'}.issubset(country_mapping.columns):
            raise KeyError("country_mapping.csv 文件必须包含 'RawName' 和 'IOC_Code' 列。")
        
        country_mapping = country_mapping[['RawName', 'IOC_Code']].drop_duplicates()
        print("country_mapping 处理完成，唯一的国家映射数量:", country_mapping.shape[0])
        
        # 2. 读取奖牌数据
        print("\n读取 'summerOly_medal_counts.csv' 文件...")
        medals = pd.read_csv('summerOly_medal_counts.csv')
        print(f"medals 数据行数: {medals.shape[0]}")
        print(f"medals 列: {list(medals.columns)}")
        
        # 确保必要的列存在
        if 'NOC' not in medals.columns:
            raise KeyError("summerOly_medal_counts.csv 文件必须包含 'NOC' 列。")
        
        medals = medals.rename(columns={'NOC': 'RawName'})  # 将 NOC 重命名为 RawName 以便合并
        print("重命名 'NOC' 为 'RawName' 完成。")
        
        medals = pd.merge(medals, country_mapping, on='RawName', how='left')
        print("合并 medals 与 country_mapping 完成。")
        print(f"合并后 medals 列: {list(medals.columns)}")
        
        # 检查是否所有国家都成功映射
        missing_ioc_medals = medals[medals['IOC_Code'].isnull()]['RawName'].unique()
        if len(missing_ioc_medals) > 0:
            print(f"警告: 以下国家的 IOC_Code 未找到映射 ({len(missing_ioc_medals)} 个): {missing_ioc_medals}")
        else:
            print("所有国家的 IOC_Code 均已成功映射。")
        
        # 3. 读取 GDP 数据
        print("\n读取 'world-gdp-data.csv' 文件...")
        gdp = pd.read_csv('world-gdp-data.csv')
        print(f"gdp 数据行数: {gdp.shape[0]}")
        print(f"gdp 列: {list(gdp.columns)}")
        
        # 确保必要的列存在
        if not {'country', 'imfGDP'}.issubset(gdp.columns):
            raise KeyError("world-gdp-data.csv 文件必须包含 'country' 和 'imfGDP' 列。")
        
        # 清理 'imfGDP' 列，移除非数值字符并转换为数值类型
        print("清理 'imfGDP' 列的数据格式...")
        gdp['imfGDP'] = gdp['imfGDP'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        gdp['imfGDP'] = pd.to_numeric(gdp['imfGDP'], errors='coerce')
        
        # 检查 'imfGDP' 清理后的统计信息
        print("\n清理后的 'imfGDP' 统计信息：")
        print(gdp['imfGDP'].describe())
        
        # 合并 GDP 与 country_mapping
        gdp = pd.merge(gdp, country_mapping, left_on='country', right_on='RawName', how='left')
        print("合并 gdp 与 country_mapping 完成。")
        print(f"合并后 gdp 列: {list(gdp.columns)}")
        
        # 检查是否所有国家都成功映射
        missing_ioc_gdp = gdp[gdp['IOC_Code'].isnull()]['country'].unique()
        if len(missing_ioc_gdp) > 0:
            print(f"警告: 以下国家的 IOC_Code 未找到映射 ({len(missing_ioc_gdp)} 个): {missing_ioc_gdp}")
        else:
            print("所有国家的 IOC_Code 均已成功映射。")
        
        # 选择需要的列
        gdp = gdp[['IOC_Code', 'imfGDP']]
        print("GDP 数据整理完成。")
        
        # 4. 读取主办国数据
        print("\n读取 'summerOly_hosts.csv' 文件...")
        hosts = pd.read_csv('summerOly_hosts.csv')
        print(f"hosts 数据行数: {hosts.shape[0]}")
        print(f"hosts 列: {list(hosts.columns)}")
        
        hosts['Host'] = hosts['Host'].astype(str).str.strip()
        # 过滤掉包含 "Cancelled" 的条目
        initial_hosts_count = hosts.shape[0]
        hosts = hosts[~hosts['Host'].str.contains('Cancelled', case=False, na=False)]
        filtered_hosts_count = hosts.shape[0]
        print(f"过滤掉取消的主办国条目：{initial_hosts_count - filtered_hosts_count} 个条目被移除。")
        
        # 提取 Host_Country
        hosts['Host_Country'] = hosts['Host'].apply(extract_country)
        print("提取 'Host_Country' 完成。")
        
        print("读取后的 hosts 数据预览:")
        print(hosts.head())
        
        # 合并 Host_Country 与 country_mapping
        hosts = pd.merge(hosts, country_mapping, left_on='Host_Country', right_on='RawName', how='left')
        print("合并 hosts 与 country_mapping 完成。")
        print(f"合并后 hosts 列: {list(hosts.columns)}")
        
        # 检查是否所有主办国都成功映射
        missing_ioc_hosts = hosts[hosts['IOC_Code'].isnull()]['Host_Country'].unique()
        if len(missing_ioc_hosts) > 0:
            print(f"警告: 以下主办国的 IOC_Code 未找到映射 ({len(missing_ioc_hosts)} 个): {missing_ioc_hosts}")
        else:
            print("所有主办国的 IOC_Code 均已成功映射。")
        
        # 转换 Host 列为二元标志
        hosts['Host_Flag'] = 1
        hosts = hosts[['Year', 'IOC_Code', 'Host_Flag']]
        print("Hosts 数据整理完成。")
        
        # 5. 计算 Medals 列（如果不存在）
        if 'Medals' not in medals.columns:
            medals['Medals'] = medals['Gold'] + medals['Silver'] + medals['Bronze']
            print("[merge_data] 添加 'Medals' 列作为 Gold、Silver 和 Bronze 的总和。")
        else:
            print("[merge_data] 'Medals' 列已存在，无需添加。")
        
        # 6. 合并奖牌数据与 GDP 数据
        print("\n合并 medals 与 gdp 数据...")
        data = pd.merge(medals, gdp, on='IOC_Code', how='left')
        print(f"合并后 data 列数: {list(data.columns)}")
        
        # 7. 合并主办国数据
        print("合并 data 与 hosts 数据...")
        data = pd.merge(data, hosts, on=['Year', 'IOC_Code'], how='left')
        print(f"合并后 data 列数: {list(data.columns)}")
        
        # 填充 Host_Flag 的缺失值为 0（不是主办国）
        missing_hosts = data['Host_Flag'].isnull().sum()
        if missing_hosts > 0:
            print(f"填充 'Host_Flag' 列中 {missing_hosts} 个缺失值为 0。")
            # 使用更安全的填充方法来避免 FutureWarning
            data['Host_Flag'] = data['Host_Flag'].fillna(0).astype(int)
        else:
            print("所有 'Host_Flag' 列的值均已填充。")
        
        # 8. 数据质量检查
        print("\n进行数据质量检查...")
        # 删除 'imfGDP' 列中缺失值的行
        missing_imfGDP = data['imfGDP'].isnull().sum()
        if missing_imfGDP > 0:
            print(f"删除 'imfGDP' 列中 {missing_imfGDP} 个缺失值对应的行。")
            data = data.dropna(subset=['imfGDP'])
        else:
            print("所有 'imfGDP' 列的值均已填充。")
        
        # 重新检查 'IOC_Code' 列中是否有缺失值
        missing_ioc_final = data['IOC_Code'].isnull().sum()
        if missing_ioc_final > 0:
            print(f"错误: 'IOC_Code' 列中仍有 {missing_ioc_final} 个缺失值。")
            raise ValueError("合并数据后存在缺失的 'IOC_Code'。请检查原始数据和映射文件。")
        else:
            print("所有 'IOC_Code' 列的值均已填充。")
        
        # 9. 保存合并后的数据到新的 CSV 文件
        output_file = 'combined_data.csv'
        data.to_csv(output_file, index=False)
        print(f"\n[merge_data] 数据合并完成，保存为 '{output_file}'。")
    
    except Exception as e:
        print(f"错误: {e}")
        print("请检查您的输入数据和参数。")

if __name__ == "__main__":
    merge_data() 