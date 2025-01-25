import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from PyPortfolioOpt import EfficientFrontier, risk_models, expected_returns

class OlympicDataLoader:
    def load_combined_data(self, filepath):
        """
        加载合并后的数据文件。

        参数：
            filepath (str): CSV文件的路径。

        返回：
            df (pd.DataFrame): 加载并清洗后的DataFrame。
        """
        # 加载CSV文件
        df = pd.read_csv(filepath)
        
        # 重命名列以匹配代码中的要求（假设 'Medals' 对应 'Medal_count'）
        df = df.rename(columns={
            'country': 'Country',
            'Medals': 'Medal_count'
            # 如果有其他需要重命名的列，请在这里添加
        })
        
        # 检查并添加缺失的列
        required_columns = ['Sport', 'Treat', 'Post']
        for col in required_columns:
            if col not in df.columns:
                print(f"警告: 数据中缺少列'{col}'，将初始化为0。")
                df[col] = 0  # 根据实际情况调整初始化值
        
        # 检查并转换必要的列为数值类型
        numeric_cols = ['Year', 'Treat', 'Post', 'Medal_count', 'imfGDP', 'unGDP', 'gdpPerCapita']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"警告: 数据中缺少数值列'{col}'，将初始化为0。")
                df[col] = 0
        
        # 检查NaN值并处理
        if df[numeric_cols].isna().any().any():
            print("警告：数据中存在NaN值，将被替换为0。")
            # 显示包含NaN值的行
            nan_rows = df[df[numeric_cols].isna().any(axis=1)]
            print("包含NaN值的行：")
            print(nan_rows)
            
            # 替换NaN为0
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df

class TripleDifferenceModel:
    def __init__(self, data):
        """
        初始化与合并数据集。

        参数：
            data (pd.DataFrame): 包含以下列的DataFrame
                - 'Country': 国家
                - 'Sport': 运动项目
                - 'Year': 年份
                - 'Treat': 是否受到"伟大教练"干预（0/1）
                - 'Post': 干预后时期（0/1）
                - 'Medal_count': 奖牌数
        """
        self.data = data
        
    def fit_model(self):
        """
        拟合三重差分模型：
        Medal_count = β0 + β1Treat + β2Post + β3C(Sport) + β4DID + β5DDD + ε

        返回：
            model (RegressionResultsWrapper): 拟合后的回归模型。
        """
        # 创建交互项
        self.data['DID'] = self.data['Treat'] * self.data['Post']
        if 'Sport' in self.data.columns:
            self.data['DDD'] = self.data['DID'] * self.data['Sport'].astype('category').cat.codes
        else:
            self.data['DDD'] = 0  # 如果无Sport，设为0
        
        # 拟合OLS模型
        formula = 'Medal_count ~ Treat + Post + C(Sport) + DID + DDD' if 'Sport' in self.data.columns else 'Medal_count ~ Treat + Post + DID + DDD'
        self.model = ols(formula, data=self.data).fit()
        return self.model
    
    def effect_decomposition(self):
        """
        分解效应为个体教练、教练团队和传承效应。

        返回：
            effects (dict): 包含各效应及其置信区间的字典。
        """
        params = self.model.params
        ci = self.model.conf_int()
        
        effects = {
            'Individual Coach': {
                'effect': params.get('DID', np.nan),
                'ci': ci.loc['DID'].tolist() if 'DID' in ci.index else [np.nan, np.nan]
            },
            'Coaching Team': {
                'effect': params.get('DDD', np.nan) if 'DDD' in params else np.nan,
                'ci': ci.loc['DDD'].tolist() if 'DDD' in ci.index else [np.nan, np.nan]
            },
            'Legacy Effect': {
                'effect': params.get('Treat', np.nan),
                'ci': ci.loc['Treat'].tolist() if 'Treat' in ci.index else [np.nan, np.nan]
            }
        }
        return effects
    
    def optimize_resource_allocation(self, investment_budget):
        """
        使用马科维茨均值-方差优化教练资源分配。

        参数：
            investment_budget (float): 可投资预算。

        返回：
            allocation (dict): 各运动项目的投资分配。
        """
        # 计算预期收益和协方差矩阵
        sport_returns = self.data.groupby('Sport')['Medal_count'].mean()
        cov_matrix = self.data.pivot_table(
            index='Year', 
            columns='Sport', 
            values='Medal_count'
        ).cov()
        
        # 计算预期收益和协方差
        mu = expected_returns.ema_historical_return(self.data.pivot_table(
            index='Year',
            columns='Sport',
            values='Medal_count'
        ), returns_data=False)
        S = risk_models.sample_cov(self.data.pivot_table(
            index='Year',
            columns='Sport',
            values='Medal_count'
        ))
        
        # 优化组合
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        # 缩放到投资预算
        allocation = {sport: weight * investment_budget for sport, weight in cleaned_weights.items()}
        return allocation

def analyze_coach_effects(data, investment_budget=1000000):
    """
    进行"伟大教练效应分析（阶段IV）"的完整示例流程：
    1. 设定并拟合三重差分模型（TripleDifferenceModel）
    2. 输出实证结果（包括回归摘要和效应分解）
    3. 使用马科维茨均值-方差模型给出战略建议（最优投资组合）
    
    参数：
    data (pd.DataFrame): 包含以下列的 DataFrame
        - 'Country': 国家
        - 'Sport': 运动项目
        - 'Year': 年份
        - 'Treat': 是否受到"伟大教练"干预（0/1）
        - 'Post': 干预后时期（0/1）
        - 'Medal_count': 奖牌数
    investment_budget (float): 为教练团队分配的可投资预算
    """
    # 初始化三重差分模型
    triple_diff_model = TripleDifferenceModel(data)
    
    # 拟合模型
    model_result = triple_diff_model.fit_model()
    
    # 输出回归摘要
    print("三重差分模型（DDD）回归摘要：")
    print(model_result.summary())
    
    # 进行效应分解
    effects = triple_diff_model.effect_decomposition()
    print("\n三重差分效应分解：")
    print("--------------------------------------------------")
    print(f"{'影响源':<15} | {'效应值':<10} | {'置信区间'}")
    print("--------------------------------------------------")
    for source, result in effects.items():
        effect_val = round(result['effect'], 3) if not pd.isna(result['effect']) else 'N/A'
        ci_lower = round(result['ci'][0], 3) if not pd.isna(result['ci'][0]) else 'N/A'
        ci_upper = round(result['ci'][1], 3) if not pd.isna(result['ci'][1]) else 'N/A'
        print(f"{source:<15} | {effect_val:<10} | [{ci_lower},{ci_upper}]")
    print("--------------------------------------------------")

    # 给出战略建议：使用马科维茨均值-方差模型进行资源分配
    if 'Sport' in data.columns:
        allocation_plan = triple_diff_model.optimize_resource_allocation(investment_budget)
        
        print("\n教练团队最优投资组合建议：")
        print("--------------------------------------------------")
        print(f"{'运动项目':<20} | {'分配预算(¥)'}")
        print("--------------------------------------------------")
        for sport, allocated in allocation_plan.items():
            print(f"{sport:<20} | {round(allocated,2):<10}")
        print("--------------------------------------------------")
        print("注：以上为基于马科维茨均值-方差模型所得到的示例性资源分配策略。")
    else:
        print("\n无法进行资源分配优化，因为数据中缺少 'Sport' 列。")

def generate_sample_data():
    """
    生成一份用于测试三重差分模型（DDD）的示例数据

    返回：
        df (pd.DataFrame): 包含'Country', 'Sport', 'Year', 'Treat', 'Post', 'Medal_count'列的样例DataFrame
    """
    np.random.seed(42)
    
    # 假设有三个国家、三个运动项目，以及几个不同的年份
    countries = ['IND', 'BRA', 'KEN']
    sports = ['Shooting', 'Gymnastics', 'Swimming']
    years = [2012, 2016, 2020, 2024]  # 示例: 4个循环周期
    
    data_list = []
    
    for c in countries:
        for s in sports:
            for y in years:
                # Treat 用于指示是否受到"伟大教练"干预
                # 我们简单地假设每个国家在不同时间点开始干预
                treat = 1 if (c == 'IND' and y >= 2016) or \
                           (c == 'BRA' and y >= 2020) or \
                           (c == 'KEN' and y >= 2024) else 0
                           
                # Post 用于指示是否处于干预后时期（模型中往往与Treat有重叠，但也可根据实际需求自行设定）
                # 这里简单处理，使 Post = 1 表示 2020 年以及之后的时间
                post = 1 if y >= 2020 else 0
                
                # Medal_count 奖牌数模拟
                # 为了体现干预可能带来的增加，若 treat=1，则多加一点随机增益
                base_medals = np.random.poisson(lam=2)  # 基础奖牌分布
                boost = np.random.randint(1, 3) if treat == 1 else 0
                total_medals = base_medals + boost
                
                data_list.append({
                    'Country': c,
                    'Sport': s,
                    'Year': y,
                    'Treat': treat,
                    'Post': post,
                    'Medal_count': total_medals
                })
    
    df = pd.DataFrame(data_list)
    return df

def main():
    # 初始化数据加载器
    data_loader = OlympicDataLoader()
    
    # 指定CSV文件路径
    filepath = 'combined_data_merged.csv'
    
    try:
        # 加载合并后的数据
        df = data_loader.load_combined_data(filepath)
        
        # 显示加载的数据前几行以确认
        print("加载的数据预览：")
        print(df.head())
        
        # 确保数据中包含必要的列
        required_columns = ['Country', 'Sport', 'Year', 'Treat', 'Post', 'Medal_count']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 初始化并分析三重差分模型
        analyze_coach_effects(data=df, investment_budget=1000000)
        
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{filepath}'。请确保文件路径正确。")
    except ValueError as ve:
        print(f"数据错误: {ve}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    main()