def merge_gdp_columns():
    import pandas as pd

    # 读取 world-gdp-data-with-ioc.csv
    world_gdp = pd.read_csv('world-gdp-data-with-ioc.csv')

    # 读取 combined_data.csv，使用逗号作为分隔符，并且包含表头
    combined = pd.read_csv('combined_data.csv', sep=',', header=0)

    # 重命名 'RawName' 列为 'country' 以便于合并
    combined.rename(columns={'RawName': 'country'}, inplace=True)

    # 合并两个数据集，基于 'country' 列
    merged = combined.merge(world_gdp[['country', 'unGDP', 'gdpPerCapita']], on='country', how='left')

    # 保存更新后的数据到新的 CSV 文件
    merged.to_csv('combined_data_updated.csv', index=False)

if __name__ == "__main__":
    merge_gdp_columns() 