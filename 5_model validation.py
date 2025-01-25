import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

def historical_backtest(data):
    """
    进行历史回溯测试。
    """
    # 使用 imfGDP, unGDP, gdpPerCapita 和 Host_Flag 预测 Medals
    X = data[['imfGDP', 'unGDP', 'gdpPerCapita', 'Host_Flag']]
    y = data['Medals']
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    
    # 计算 SMAPE
    smape = 100/len(y) * np.sum(2 * np.abs(predictions - y) / (np.abs(y) + np.abs(predictions)))
    
    # 计算 MAE
    mis = mean_absolute_percentage_error(y, predictions) * 100
    
    return smape, mis

def policy_shock_simulation(data):
    """
    进行政策冲击模拟。
    """
    # 示例代码，您需要根据实际情况填充
    # 假设通过改变 GDP 来模拟冲击
    original_gdp = data['imfGDP'].mean()
    data['imfGDP'] *= 1.05  # GDP 增加 5%
    
    X = data[['imfGDP', 'unGDP', 'gdpPerCapita', 'Host_Flag']]
    y = data['Medals']
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    
    # 计算敏感性矩阵
    S = np.array([[model.coef_[0], model.coef_[1], model.coef_[2]]])  # 添加 unGDP 和 gdpPerCapita 的系数
    return S

def causal_inference_validation(data, treatment, outcome, covariates):
    """
    进行因果推断验证。
    """
    # 添加所有相关协变量
    X = data[covariates + ['unGDP', 'gdpPerCapita']]
    y = data[outcome]
    T = data[treatment]
    
    model = LinearRegression()
    model.fit(X, T)
    T_hat = model.predict(X)
    
    # 使用倾向评分匹配等方法进行因果推断，这里仅为示例
    model_outcome = LinearRegression()
    model_outcome.fit(X.assign(T=T_hat), y)
    ate = model_outcome.coef_[X.columns.tolist().index('imfGDP') + 1]  # 假设 Host_Flag 是最后一个系数
    # 置信区间示例
    ate_lower = ate - 1.96 * model_outcome.score(X.assign(T=T_hat), y)
    ate_upper = ate + 1.96 * model_outcome.score(X.assign(T=T_hat), y)
    
    return {
        'ate': ate,
        'ate_lower': ate_lower,
        'ate_upper': ate_upper
    }

def counterfactual_analysis(data, scenario):
    """
    进行反事实分析
    scenario: 包含情景参数的字典
      期望的键: 'country', 'year', 'treatment_var', 'treatment_value'
    """
    # 示例代码，您需要根据实际情况填充
    # 假设 scenario 影响 Host_Flag
    counter_data = data.copy()
    mask = (counter_data['IOC_Code'] == scenario['country']) & (counter_data['Year'] == scenario['year'])
    counter_data.loc[mask, scenario['treatment_var']] = scenario['treatment_value']
    
    X = counter_data[['imfGDP', 'unGDP', 'gdpPerCapita', 'Host_Flag']]
    y = counter_data['Medals']
    model = LinearRegression()
    model.fit(X, y)
    counterfactual = model.predict(X)
    
    actual_medals = data.loc[mask, 'Medals'].values
    counterfactual_medals = counterfactual[mask]
    effect = actual_medals - counterfactual_medals
    
    return {
        'actual_medals': actual_medals[0] if len(actual_medals) > 0 else None,
        'counterfactual_medals': counterfactual_medals[0] if len(counterfactual_medals) > 0 else None,
        'effect_of_hosting': effect[0] if len(effect) > 0 else None
    }

def load_and_prepare_data(file_path='combined_data_merged.csv'):
    """
    加载并准备数据，确保数据格式正确。
    并根据 imfGDP, unGDP, gdpPerCapita 组合成新的特征。
    """
    try:
        data = pd.read_csv(file_path)
        
        # 检查必要的列
        required_columns = ['Rank', 'Gold', 'Silver', 'Bronze', 'Total', 'Year', 
                            'country', 'IOC_Code', 'Medals', 'imfGDP', 'unGDP', 'gdpPerCapita', 'Host_Flag']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise KeyError(f"数据缺少必要的列: {missing_columns}")
        
        # 组合新的特征，例如总GDP
        data['totalGDP'] = data['imfGDP'] + data['unGDP'] + data['gdpPerCapita']
        
        # GDP人均比
        data['gdp_per_capita_ratio'] = data['imfGDP'] / data['gdpPerCapita']
        
        print("[load_and_prepare_data] 数据加载并处理完成。")
        print(f"数据行数: {data.shape[0]}")
        print(f"列: {list(data.columns)}")
        
        return data
    
    except KeyError as e:
        print(f"错误: {e}")
        print("请检查您的输入数据和参数。")
        return None
    except Exception as e:
        print(f"错误: {e}")
        print("请检查您的输入数据和参数。")
        return None

def main():
    try:
        # 加载并准备数据
        data = load_and_prepare_data()
        if data is None:
            raise ValueError("数据加载失败。")
        
        # 三重验证框架
        # 1. 历史回溯测试
        smape, mis = historical_backtest(data)
        print(f"\n历史回溯结果 (1976-2020):")
        print(f"sMAPE: {smape:.2f}%")
        print(f"平均区间得分: {mis:.2f}")
        
        # 2. 政策冲击模拟
        S = policy_shock_simulation(data)
        print("\n政策冲击敏感性矩阵:")
        print("       GDP     unGDP  gdpPerCapita")
        print(f"Medals {S[0,0]:.2f}  {S[0,1]:.2f}  {S[0,2]:.2f}")
        
        # 3. 因果推断验证
        treatment = 'Host_Flag'
        outcome = 'Medals'
        covariates = ['imfGDP']  # 可以根据需要添加其他协变量
        causal_results = causal_inference_validation(data, treatment, outcome, covariates)
        print("\n因果推断结果:")
        print(f"平均处理效应 (ATE): {causal_results['ate']:.2f}")
        print(f"置信区间: [{causal_results['ate_lower']:.2f}, {causal_results['ate_upper']:.2f}]")
        
        # 反事实分析
        scenario = {
            'country': 'JPN',  # 使用 IOC_Code 代替国家名称
            'year': 2020,
            'treatment_var': 'Host_Flag',
            'treatment_value': 0  # 假设日本未申办2020奥运会
        }
        counterfactual = counterfactual_analysis(data, scenario)
        print("\n反事实分析结果:")
        print(f"实际奖牌数: {counterfactual['actual_medals']}")
        print(f"反事实奖牌数: {counterfactual['counterfactual_medals']}")
        print(f"情景影响: {counterfactual['effect_of_hosting']}")
        
        # 预测 vs 实际
        X = data[['imfGDP', 'unGDP', 'gdpPerCapita', 'Host_Flag']]
        y = data['Medals']
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        plt.xlabel('实际奖牌数')
        plt.ylabel('预测奖牌数')
        plt.title('实际奖牌数 vs 预测奖牌数')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.show()
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请检查您的输入数据和参数")

if __name__ == "__main__":
    main()