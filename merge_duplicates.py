import pandas as pd

def merge_duplicate_countries(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 确保数值列为数值类型，填充缺失值为0
    numeric_cols = ['Gold', 'Silver', 'Bronze', 'Total', 'Medals', 'imfGDP', 'unGDP', 'gdpPerCapita']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 按照年份、国家和IOC_Code进行分组，并汇总数值列
    grouped_df = df.groupby(['Year', 'country', 'IOC_Code', 'Host_Flag']).agg({
        'Rank': 'min',        # 假设取最小排名
        'Gold': 'sum',
        'Silver': 'sum',
        'Bronze': 'sum',
        'Total': 'sum',
        'Medals': 'sum',
        'imfGDP': 'sum',
        'unGDP': 'sum',
        'gdpPerCapita': 'mean'  # 平均人均GDP
    }).reset_index()

    # 保存到新的CSV文件
    grouped_df.to_csv(output_file, index=False)
    print(f"合并后的数据已保存到 {output_file}")

if __name__ == "__main__":
    input_csv = 'combined_data_updated.csv'
    output_csv = 'combined_data_merged.csv'
    merge_duplicate_countries(input_csv, output_csv) 