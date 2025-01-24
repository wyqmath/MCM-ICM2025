import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

# 在类外部定义 SimpleMovingAverage
class SimpleMovingAverage:
    def __init__(self, window=3):
        self.window = window
        self.last_value = None
    
    def predict(self):
        return np.array([self.last_value])
    
    def fit(self, data):
        if isinstance(data, pd.Series):
            self.last_value = data.rolling(window=self.window).mean().iloc[-1]
        else:
            self.last_value = pd.Series(data).rolling(window=self.window).mean().iloc[-1]
        return self

# 在STGCN_LSTM类之前添加TemporalTransformer类的定义
class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.positional_encoding = self.create_positional_encoding()
        
        # 添加输出层以匹配目标维度
        self.output_layer = nn.Linear(d_model, 1)
        
    def create_positional_encoding(self):
        max_seq_len = 1000  # 可以根据需要调整
        pe = torch.zeros(max_seq_len, self.d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, x):
        # 确保输入维度正确
        if x.dim() == 2:
            batch_size, features = x.size()
            # 如果特征维度不等于d_model，进行投影
            if features != self.d_model:
                x = nn.Linear(features, self.d_model)(x)
            # 添加序列维度
            x = x.unsqueeze(0)  # [1, batch_size, d_model]
        
        seq_len = x.size(0)
        pos_enc = self.positional_encoding[:seq_len, :].to(x.device)
        
        # 添加位置编码
        x = x + pos_enc.unsqueeze(1)  # 广播到batch维度
        
        # Transformer编码
        output = self.transformer_encoder(x)
        
        # 取序列的平均值
        output = output.mean(dim=0)  # [batch_size, d_model]
        
        # 通过输出层得到最终预测
        output = self.output_layer(output)  # [batch_size, 1]
        
        return output

class STGCN_LSTM(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers, num_heads):
        super(STGCN_LSTM, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(node_feat_dim if i == 0 else hidden_dim * num_heads,
                   hidden_dim, heads=num_heads)
            for i in range(num_layers)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(hidden_dim * num_heads, hidden_dim, 
                           num_layers=1, batch_first=True)
        
        # Output layer - predict for all nodes
        self.fc = nn.Linear(hidden_dim, node_feat_dim)
        
    def forward(self, x, edge_index, edge_attr, seq_length):
        # 空间建模
        batch_size = x.size(0) if x.dim() == 3 else 1
        num_nodes = x.size(1) if x.dim() == 3 else x.size(0)
        node_feat_dim = x.size(2) if x.dim() == 3 else x.size(1)
        
        # 确保输入维度正确 [num_nodes, node_feat_dim] 或 [batch_size, num_nodes, node_feat_dim]
        if x.dim() == 3:
            # 处理批量数据
            h = x.view(-1, node_feat_dim)  # [batch_size * num_nodes, node_feat_dim]
        else:
            h = x  # [num_nodes, node_feat_dim]
        
        # 图注意力层
        for i in range(self.num_layers):
            h = F.elu(self.gat_layers[i](h, edge_index, edge_attr))
        
        # 时间建模
        feature_dim = self.hidden_dim * self.num_heads
        
        # 恢复批量维度
        h = h.view(batch_size, num_nodes, feature_dim)
        
        # LSTM处理
        lstm_out, _ = self.lstm(h)
        
        # 最终预测 - 确保输出维度与输入节点数量匹配
        out = self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出
        # 确保输出维度正确
        if out.size(1) != num_nodes:
            # 如果维度不匹配，使用线性变换调整
            out = nn.Linear(out.size(1), num_nodes)(out)
        out = out.view(batch_size, num_nodes)  # 输出形状 [batch_size, num_nodes]
        
        return out

def handle_nan_values(data):
    """处理数据中的NaN值，确保所有NaN都被处理"""
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        # 首先尝试线性插值
        filled_data = data.interpolate(method='linear', limit_direction='both', axis=0)
        # 使用前向填充
        filled_data = filled_data.ffill()
        # 使用后向填充
        filled_data = filled_data.bfill()
        # 如果仍然有NaN，用0填充
        return filled_data.fillna(0)
    elif isinstance(data, torch.Tensor):
        # 将Tensor转换为DataFrame进行处理
        df = pd.DataFrame(data.numpy())
        filled_df = df.interpolate(method='linear', limit_direction='both', axis=0)
        # 使用前向填充
        filled_df = filled_df.ffill()
        # 使用后向填充
        filled_df = filled_df.bfill()
        # 如果仍然有NaN，用0填充
        filled_df = filled_df.fillna(0)
        return torch.tensor(filled_df.values, dtype=data.dtype)
    elif isinstance(data, np.ndarray):
        # 将numpy数组转换为DataFrame处理
        df = pd.DataFrame(data)
        filled_df = df.interpolate(method='linear', limit_direction='both', axis=0)
        # 使用前向填充
        filled_df = filled_df.ffill()
        # 使用后向填充
        filled_df = filled_df.bfill()
        # 如果仍然有NaN，用0填充
        filled_df = filled_df.fillna(0)
        return filled_df.values
    else:
        raise ValueError("Unsupported data type for NaN handling")

class DeepEnsemble:
    def __init__(self):
        self.models = [
            STGCN_LSTM(node_feat_dim=64, edge_feat_dim=32,
                      hidden_dim=128, num_layers=2, num_heads=4),
            HistGradientBoostingRegressor(),
            XGBRegressor(),
            TemporalTransformer(  # 添加时间编码Transformer
                d_model=64,
                nhead=8,
                num_encoder_layers=6,
                dim_feedforward=256,
                dropout=0.1
            )
        ]
        
    def normalize_data(self, data):
        # Normalize node features
        data.x = (data.x - data.x.mean(dim=0)) / (data.x.std(dim=0) + 1e-8)
        
        # Normalize edge attributes
        data.edge_attr = (data.edge_attr - data.edge_attr.mean()) / (data.edge_attr.std() + 1e-8)
        
        # Normalize target values
        data.y = (data.y - data.y.mean()) / (data.y.std() + 1e-8)
        return data

    def train(self, data, epochs=100, lr=0.001, batch_size=32):
        # 预处理：确保所有数据都没有NaN值
        print("开始数据预处理...")
        
        # 处理输入特征
        if torch.isnan(data.x).any():
            print(f"处理输入特征中的NaN值: {torch.isnan(data.x).sum().item()}")
            data.x = handle_nan_values(data.x)
            print(f"处理后的NaN值数量: {torch.isnan(data.x).sum().item()}")
        
        # 处理目标值
        if torch.isnan(data.y).any():
            print(f"处理目标值中的NaN值: {torch.isnan(data.y).sum().item()}")
            data.y = handle_nan_values(data.y)
            print(f"处理后的NaN值数量: {torch.isnan(data.y).sum().item()}")
        
        # 处理边属性
        if torch.isnan(data.edge_attr).any():
            print(f"处理边属性中的NaN值: {torch.isnan(data.edge_attr).sum().item()}")
            data.edge_attr = handle_nan_values(data.edge_attr)
            print(f"处理后的NaN值数量: {torch.isnan(data.edge_attr).sum().item()}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.models[0].parameters(),
                                   lr=lr,
                                   weight_decay=1e-5)
        
        # 修改 lr_scheduler，移除 verbose 参数
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # 转换数据为DataLoader
        from torch_geometric.loader import DataLoader as GeometricDataLoader
        train_loader = GeometricDataLoader([data], batch_size=1, shuffle=True)
        
        # 训练STGCN-LSTM
        self.models[0].train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_batches = 0
            
            for batch in train_loader:
                try:
                    optimizer.zero_grad()
                    
                    outputs = self.models[0](
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        seq_length=10
                    )
                    
                    target = batch.y.view(outputs.size())
                    
                    # 验证输出和目标
                    if not (torch.isnan(outputs).any() or torch.isnan(target).any()):
                        loss = criterion(outputs, target)
                        
                        # 检查损失值是否合理
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss.backward()
                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(self.models[0].parameters(), max_norm=1.0)
                            optimizer.step()
                            
                            epoch_loss += loss.item()
                            valid_batches += 1
                        else:
                            print(f"Epoch [{epoch+1}] 警告：损失值异常 {loss.item()}")
                    else:
                        print(f"Epoch [{epoch+1}] 警告：输出或目标包含NaN值")
                        # 处理NaN值
                        batch.x = handle_nan_values(batch.x)
                        batch.y = handle_nan_values(batch.y)
                    
                except Exception as e:
                    print(f"Epoch [{epoch+1}] 错误：{str(e)}")
                    continue
            
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], 平均损失: {avg_loss:.4f}')
            else:
                print(f"Epoch [{epoch+1}/{epochs}] 警告：没有有效的批次")
        
        # Train other models
        X = data.x.numpy()
        y = data.y.numpy()
        
        # 确保没有NaN值
        if np.isnan(y).any():
            print("处理目标变量中的NaN值")
            y = handle_nan_values(y)
        
        if np.isnan(X).any():
            print("处理输入特征中的NaN值")
            X = handle_nan_values(X)
        
        # 确保y是一维数组
        y = y.ravel()
        
        self.models[1].fit(X, y)
        
        # XGBoost
        self.models[2].fit(X, y)
        
        # 训练时间编码Transformer
        self.models[3].train()
        temporal_optimizer = torch.optim.Adam(self.models[3].parameters(), lr=lr)
        temporal_criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            temporal_optimizer.zero_grad()
            # 确保输入数据维度正确
            temporal_input = data.x
            if temporal_input.dim() == 2:
                temporal_input = temporal_input.float()  # 确保数据类型正确
            
            temporal_pred = self.models[3](temporal_input)
            # 确保目标维度匹配
            target = data.y.view(temporal_pred.size())
            temporal_loss = temporal_criterion(temporal_pred, target)
            temporal_loss.backward()
            temporal_optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Temporal Transformer Epoch [{epoch+1}/{epochs}], Loss: {temporal_loss.item():.4f}')
        
    def save_models(self, path):
        try:
            torch.save(self.models[0].state_dict(), f'{path}/stgcn_lstm.pth')
            joblib.dump(self.models[1], f'{path}/gbr.pkl')
            joblib.dump(self.models[2], f'{path}/xgb.pkl')
            torch.save(self.models[3].state_dict(), f'{path}/temporal_transformer.pth')
        except Exception as e:
            print(f"Error saving models: {str(e)}")
        
    def load_models(self, path):
        self.models[0].load_state_dict(torch.load(f'{path}/stgcn_lstm.pth'))
        self.models[1] = joblib.load(f'{path}/gbr.pkl')
        self.models[2] = joblib.load(f'{path}/xgb.pkl')
        self.models[3].load_state_dict(torch.load(f'{path}/temporal_transformer.pth'))
        
    def predict(self, graph_data):
        predictions = []
        target_shape = (graph_data.x.size(0), 1)

        for i, model in enumerate(self.models):
            try:
                if isinstance(model, nn.Module):
                    model.eval()
                    with torch.no_grad():
                        if isinstance(model, STGCN_LSTM):
                            pred = model(
                                graph_data.x,
                                graph_data.edge_index,
                                graph_data.edge_attr,
                                seq_length=10
                            )
                        else:  # TemporalTransformer
                            pred = model(graph_data.x)
                else:
                    pred = model.predict(graph_data.x.numpy())
                
                # 确保预测形状正确
                if pred.shape != target_shape:
                    pred = pred.reshape(target_shape)
                    
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model {i} prediction failed: {str(e)}")
                predictions.append(np.zeros(target_shape))
        
        # 堆叠预测并计算分位数
        predictions = np.stack(predictions, axis=0)
        lower = np.quantile(predictions, 0.05, axis=0)
        upper = np.quantile(predictions, 0.95, axis=0)
        return np.mean(predictions, axis=0), (lower, upper)

    def validate_data(self, data):
        """验证数据完整性和质量"""
        validation_results = {
            'missing_values': torch.isnan(data.x).sum().item(),
            'negative_values': (data.x < 0).sum().item(),
            'infinite_values': torch.isinf(data.x).sum().item()
        }
        
        print("数据验证结果：", validation_results)
        return validation_results

def align_data(primary_data, primary_countries, secondary_data, secondary_countries):
    # Create country name to index mapping
    primary_map = {country: idx for idx, country in enumerate(primary_countries)}
    secondary_map = {country: idx for idx, country in enumerate(secondary_countries)}
    
    # Handle different data dimensions
    if secondary_data.dim() == 1:
        num_features = 1
        secondary_data = secondary_data.unsqueeze(1)
    else:
        num_features = secondary_data.size(1)
    
    # Initialize aligned data with zeros
    aligned_data = torch.zeros(primary_data.size(0), num_features)
    
    # Create a unified country list based on primary data
    unified_countries = list(primary_countries)
    
    # Add missing countries from secondary data
    for country in secondary_countries:
        if country not in primary_map:
            unified_countries.append(country)
            primary_map[country] = len(unified_countries) - 1
    
    # Resize aligned data to match unified country list
    aligned_data = torch.zeros(len(unified_countries), num_features)
    
    # Align data based on unified country list
    for country, idx in primary_map.items():
        if country in secondary_map:
            aligned_data[idx] = secondary_data[secondary_map[country]]
        else:
            aligned_data[idx] = torch.zeros(num_features)
    
    # Convert primary_countries to list if it's a pandas Series
    if hasattr(primary_countries, 'tolist'):
        primary_countries = primary_countries.tolist()
    
    # Update primary countries to match unified list
    primary_countries = unified_countries
    
    return aligned_data, primary_countries

def calculate_node_features(weight_matrix, weight_countries, coach_scores, coach_countries, event_specialization):
    aligned_coach_scores, aligned_countries = align_data(weight_matrix[:, 0], weight_countries, coach_scores, coach_countries)
    W_tilde = F.normalize(weight_matrix, p=2, dim=1)
    # Ensure all tensors are 2D and have matching sizes
    if event_specialization.dim() == 3:
        event_specialization = event_specialization.squeeze(1)
        
    # Get target size from weight matrix
    target_size = W_tilde.size(0)
    
    # Pad or truncate other tensors to match target size
    aligned_coach_scores = F.pad(aligned_coach_scores, (0, 0, 0, target_size - aligned_coach_scores.size(0)))
    event_specialization = F.pad(event_specialization, (0, 0, 0, target_size - event_specialization.size(0)))
    
    # Create node features with correct dimension
    base_features = torch.cat([
        W_tilde,
        aligned_coach_scores[:target_size],
        event_specialization[:target_size]
    ], dim=1)
    
    # Project features to match model input dimension (64)
    if base_features.size(1) < 64:
        # Add padding if needed
        padding = torch.zeros(base_features.size(0), 64 - base_features.size(1))
        node_features = torch.cat([base_features, padding], dim=1)
    else:
        # Truncate if needed
        node_features = base_features[:, :64]
    return node_features

def calculate_edge_weights(medal_corr, trade_flow, gdp):
    """Calculate edge weights according to eij = MedalCorrij/σMedalCorr + TradeFlowij/GDPi"""
    # 确保所有输入张量都是2D
    if medal_corr.dim() == 1:
        medal_corr = medal_corr.unsqueeze(1)
    if trade_flow.dim() == 1:
        trade_flow = trade_flow.unsqueeze(1)
    if gdp.dim() == 1:
        gdp = gdp.unsqueeze(1)
    
    # 统一大小为144 x min(medal_corr.size(1), trade_flow.size(1))
    target_size = 144
    target_features = min(medal_corr.size(1), trade_flow.size(1))
    
    # 裁剪或填充到目标大小
    medal_corr = medal_corr[:target_size, :target_features]
    trade_flow = trade_flow[:target_size, :target_features]
    gdp = gdp[:target_size].unsqueeze(1)  # 确保gdp是2D的 [target_size, 1]
    
    # 标准化medal correlation
    medal_corr_norm = medal_corr / (torch.std(medal_corr) + 1e-8)
    
    # 通过GDP标准化trade flow
    trade_flow_norm = trade_flow / (gdp + 1e-8)  # 广播操作
    
    # 组合权重 - 取每行的平均值作为最终权重
    edge_weights = (medal_corr_norm + trade_flow_norm).mean(dim=1)
    
    return edge_weights

class GGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GGAT, self).__init__()
        self.gat = GATConv(in_dim, out_dim, heads=num_heads)
        self.gru = nn.GRUCell(out_dim * num_heads, out_dim)
        
    def forward(self, x, edge_index, edge_attr, h):
        # Graph attention
        gat_out = F.elu(self.gat(x, edge_index, edge_attr))
        
        # GRU update
        h_new = self.gru(gat_out, h)
        
        return h_new

def build_spatio_temporal_graph(
    weight_matrix,
    weight_countries,
    coach_scores,
    coach_countries,
    event_specialization,
    medal_corr,
    trade_flow,
    gdp
    ):
    # 确保所有输入数据维度一致（144个节点）
    num_nodes = 144
    
    # 计算节点特征
    node_features = calculate_node_features(weight_matrix, weight_countries, 
                                          coach_scores, coach_countries, 
                                          event_specialization)
    
    # 裁剪或填充node_features
    if node_features.size(0) > num_nodes:
        node_features = node_features[:num_nodes]
    elif node_features.size(0) < num_nodes:
        padding = torch.zeros(num_nodes - node_features.size(0), node_features.size(1))
        node_features = torch.cat([node_features, padding])
    
    # 计算边权重
    edge_weights = calculate_edge_weights(medal_corr, trade_flow, gdp)
    
    # 创建边索引和属性
    edge_index = []
    edge_attr = []
    
    # 为每对节点创建边
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # 避免自环
                edge_index.append([i, j])
                # 使用两个节点的边权重均值
                weight = torch.mean(torch.stack([edge_weights[i], edge_weights[j]]))
                edge_attr.append(weight.item())  # 使用item()获取标量值
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # 创建目标标签
    target_weights = torch.randn(node_features.size(1), 1)
    y = node_features @ target_weights
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )

class OlympicDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_weight_matrix(self):
        df = pd.read_csv(f'{self.data_dir}/dynamic_weight_matrix_2028proj.csv')
        
        # 获取数值列和国家名称
        numeric_data = df.iloc[:, 1:3].apply(pd.to_numeric, errors='coerce')
        countries = df.iloc[:, 0]
        
        # 数据验证和处理
        if numeric_data.isnull().values.any():
            print(f"发现{numeric_data.isnull().sum().sum()}个NaN值")
            # 使用列平均值填充NaN
            numeric_data = numeric_data.fillna(numeric_data.mean())
        
        # 额外验证
        if (numeric_data < 0).values.any():
            print("发现负值")
            numeric_data = numeric_data.clip(lower=0)
        
        return torch.tensor(numeric_data.values, dtype=torch.float32), countries
    
    def load_coach_scores(self):
        # Load data with country names
        data = np.genfromtxt(
            f'{self.data_dir}/sport_pagerank_scores.csv',
            delimiter=',',
            skip_header=1,
            dtype=str,
            filling_values=''
        )
        # Extract countries and scores
        countries = data[:, 0]
        scores = np.array(data[:, 1], dtype=float)
        # Remove any rows with NaN values
        valid_mask = ~np.isnan(scores)
        return torch.tensor(scores[valid_mask]), countries[valid_mask]
    
    def load_event_specialization(self):
        import pandas as pd
        # Try different encodings for Windows CSV files
        try:
            df = pd.read_csv(f'{self.data_dir}/summerOly_programs.csv', encoding='latin1')
        except UnicodeDecodeError:
            df = pd.read_csv(f'{self.data_dir}/summerOly_programs.csv', encoding='cp1252')
        
        # Get all columns that look like years (4 digits)
        year_columns = [col for col in df.columns if col.isdigit() and len(col) == 4]
        
        # Select only the numeric year columns that exist
        numeric_data = df[year_columns]
        
        # Clean and convert all values to numeric
        def clean_value(x):
            if isinstance(x, str):
                # Remove any non-numeric characters
                x = ''.join(c for c in x if c.isdigit() or c == '.')
                # If empty after cleaning, return 0
                return float(x) if x else 0.0
            return float(x)
            
        numeric_data = numeric_data.map(clean_value)
        
        return torch.tensor(numeric_data.values, dtype=torch.float32)
    
    def load_medal_correlation(self):
        import pandas as pd
        # Read CSV with header and handle non-numeric data
        df = pd.read_csv(f'{self.data_dir}/summerOly_medal_counts.csv')
        
        # Convert to numeric, coercing errors to NaN
        numeric_data = df.apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values with 0
        numeric_data = numeric_data.fillna(0)
        
        # Ensure consistent size with other datasets (212 rows)
        if numeric_data.shape[0] > 212:
            numeric_data = numeric_data.iloc[:212]
        elif numeric_data.shape[0] < 212:
            padding = pd.DataFrame(0, index=range(212 - numeric_data.shape[0]),
                                 columns=numeric_data.columns)
            numeric_data = pd.concat([numeric_data, padding])
            
        return torch.tensor(numeric_data.values, dtype=torch.float32)
    
    def load_trade_flow(self):
        import pandas as pd
        # Read CSV with header
        df = pd.read_csv(f'{self.data_dir}/world-gdp-data.csv')
        
        # Convert to numeric, coercing errors to NaN
        numeric_data = df.apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values with 0 and convert to tensor
        return torch.tensor(numeric_data.fillna(0).values, dtype=torch.float32)
    
    def load_gdp(self):
        import pandas as pd
        # Read CSV with header
        df = pd.read_csv(f'{self.data_dir}/world-gdp-data.csv')
        
        # Convert GDP column to numeric, coercing errors to NaN
        gdp_values = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        # Fill NaN values with 0 and convert to tensor
        return torch.tensor(gdp_values.fillna(0).values, dtype=torch.float32)
        
    def get_graph_data(self):
        weight_matrix, weight_countries = self.load_weight_matrix()
        coach_scores, coach_countries = self.load_coach_scores()
        event_specialization = self.load_event_specialization()
        medal_corr = self.load_medal_correlation()
        trade_flow = self.load_trade_flow()
        gdp = self.load_gdp()
        
        return build_spatio_temporal_graph(
            weight_matrix,
            weight_countries,
            coach_scores,
            coach_countries,
            event_specialization,
            medal_corr,
            trade_flow,
            gdp
        )

# Example usage
if __name__ == "__main__":
    data_loader = OlympicDataLoader('c:/Users/Administrator/Desktop/美赛')
    model = DeepEnsemble()
    graph_data = data_loader.get_graph_data()
    graph_data = model.normalize_data(graph_data)  # Add data normalization
    model.train(graph_data)
    pred, (lower, upper) = model.predict(graph_data)

    np.savetxt('predictions.csv', pred, delimiter=',')
    np.savetxt('uncertainty_lower.csv', lower, delimiter=',')
    np.savetxt('uncertainty_upper.csv', upper, delimiter=',')