import pandas as pd
import numpy as np

class OlympicDataLoader:
    def load_weight_matrix(self, filepath):
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # Convert only numeric columns
        numeric_cols = ['Year', 'Weight']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Check for NaN values in numeric columns
        if df[numeric_cols].isna().any().any():
            print("警告：权重矩阵中存在NaN值，将被替换为0。")
            # Show rows with NaN values
            nan_rows = df[df[numeric_cols].isna().any(axis=1)]
            print("包含NaN值的行：")
            print(nan_rows)
            
            # Replace NaN with 0 only in numeric columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
        return df

# Test the loader
if __name__ == "__main__":
    loader = OlympicDataLoader()
    weights = loader.load_weight_matrix('dynamic_weight_matrix_2028proj.csv')
    print("Loaded weight matrix:")
    print(weights.head())
    print("\nData types:")
    print(weights.dtypes)