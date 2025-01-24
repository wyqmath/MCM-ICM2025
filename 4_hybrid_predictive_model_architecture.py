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
            dropout=dropout,
            batch_first=True  # 确保 batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.positional_encoding = self.create_positional_encoding()
        
        # 添加输出层以匹配目标维度
        self.output_layer = nn.Linear(d_model, 1)
        
    def create_positional_encoding(self):
        max_seq_len = 1000  # 根据需要调整
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
            # 如果特征维度不等于 d_model，进行投影
            if features != self.d_model:
                x = nn.Linear(features, self.d_model)(x)
            # 添加序列维度并保持 batch_first=True 格式
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:seq_len, :].to(x.device)
        
        # 添加位置编码 (适配 batch_first=True)
        x = x + pos_enc.unsqueeze(0)  # 广播到 batch 维度
        
        # Transformer 编码
        output = self.transformer_encoder(x)
        
        # 取序列的平均值
        output = output.mean(dim=1)  # [batch_size, d_model]
        
        # 通过输出层得到最终预测
        output = self.output_layer(output)  # [batch_size, 1]
        
        # 扩展到 [batch_size, num_nodes, 1]，假设 num_nodes=144
        num_nodes = 144
        output = output.unsqueeze(1).repeat(1, num_nodes, 1)  # [batch_size, num_nodes, 1]
        
        #print(f"TemporalTransformer Output Shape: {output.shape}")  # 调试信息
        return output

class STGCN_LSTM(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers, num_heads):
        super(STGCN_LSTM, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 1. 图注意力层保持不变
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = node_feat_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(GATConv(
                in_channels,
                hidden_dim,
                heads=num_heads,
                dropout=0.1
            ))
            self.gat_layers.append(nn.LayerNorm(hidden_dim * num_heads))
            self.gat_layers.append(nn.Dropout(0.1))
            self.gat_layers.append(Swish())
        
        # 2. 特征融合门控机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim * num_heads, hidden_dim * num_heads),
            nn.LayerNorm(hidden_dim * num_heads),
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads),
            nn.Sigmoid()
        )
        
        # 3. 修改LSTM层的隐藏维度
        lstm_hidden = hidden_dim * num_heads  # 确保与GAT输出维度匹配
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden // 2,  # 减半以适应双向LSTM
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # 4. 修改时序注意力层的维度
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,  # 现在与LSTM输出维度匹配
            num_heads=4,
            dropout=0.1
        )
        
        # 5. 修改层归一化的维度
        self.layer_norm1 = nn.LayerNorm(lstm_hidden)
        self.layer_norm2 = nn.LayerNorm(lstm_hidden)
        
        # 6. 修改Highway连接的维度
        self.highway = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.Sigmoid()
        )
        
        # 7. 修改输出层的维度
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            Swish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # 8. 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """使用改进的初始化方案"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, edge_index, edge_attr, seq_length):
        # 确保输入维度正确
        if x.dim() == 2:
            batch_size = 1
            num_nodes = x.size(0)
            x = x.unsqueeze(0)  # 添加batch维度 [1, num_nodes, node_feat_dim]
        else:
            batch_size = x.size(0)
            num_nodes = x.size(1)
        
        # 1. 空间特征提取
        h = x.view(-1, self.node_feat_dim)  # [batch_size * num_nodes, node_feat_dim]
        
        # 存储每层的特征用于残差连接
        residuals = []
        
        # 图注意力层处理
        for i in range(0, len(self.gat_layers), 4):
            residual = h
            h = self.gat_layers[i](h, edge_index, edge_attr)
            h = self.gat_layers[i+1](h)
            h = self.gat_layers[i+2](h)
            h = self.gat_layers[i+3](h)
            
            if h.size() == residual.size():
                h = h + residual
            residuals.append(h)
        
        # 2. 特征融合
        h = h.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, hidden_dim * num_heads]
        
        # 3. 双向LSTM处理
        lstm_out, _ = self.lstm(h)  # [batch_size, num_nodes, lstm_hidden]
        
        # 4. 时序注意力机制
        attn_out, _ = self.temporal_attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)
        
        # 5. 残差连接和层归一化
        out = self.layer_norm1(lstm_out + attn_out)
        
        # 6. Highway连接
        gate = self.highway(out)
        highway_out = gate * out + (1 - gate) * h
        
        # 7. 最终输出层
        out = self.layer_norm2(highway_out)
        out = self.fc(out.view(batch_size * num_nodes, -1))  # [batch_size * num_nodes, 1]
        
        # 重塑输出以匹配目标维度 [batch_size, num_nodes, 1]
        out = out.view(batch_size, num_nodes, 1)
        
        #print(f"STGCN_LSTM Output Shape: {out.shape}")  # 调试信息
        return out

# 添加Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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
        # 检查数据维度
        if data.dim() == 3:
            batch_size, num_nodes, num_features = data.size()
            # 重塑为二维 [batch_size * num_nodes, num_features]
            reshaped_data = data.view(-1, data.size(-1))
            # 转换为DataFrame进行处理
            df = pd.DataFrame(reshaped_data.numpy())
            filled_df = df.interpolate(method='linear', limit_direction='both', axis=0)
            filled_df = filled_df.ffill()
            filled_df = filled_df.bfill()
            filled_df = filled_df.fillna(0)
            # 转换回Tensor并重塑为原始形状
            filled_tensor = torch.tensor(filled_df.values, dtype=data.dtype).view(batch_size, num_nodes, num_features)
            return filled_tensor
        elif data.dim() == 2:
            df = pd.DataFrame(data.numpy())
            filled_df = df.interpolate(method='linear', limit_direction='both', axis=0)
            filled_df = filled_df.ffill()
            filled_df = filled_df.bfill()
            filled_df = filled_df.fillna(0)
            return torch.tensor(filled_df.values, dtype=data.dtype)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {data.dim()}")
    elif isinstance(data, np.ndarray):
        # 将numpy数组转换为DataFrame处理
        if data.ndim == 3:
            batch_size, num_nodes, num_features = data.shape
            reshaped_data = data.reshape(-1, data.shape[-1])
            df = pd.DataFrame(reshaped_data)
            filled_df = df.interpolate(method='linear', limit_direction='both', axis=0)
            filled_df = filled_df.ffill()
            filled_df = filled_df.bfill()
            filled_df = filled_df.fillna(0)
            filled_tensor = torch.tensor(filled_df.values, dtype=torch.float32).view(batch_size, num_nodes, num_features)
            return filled_tensor
        elif data.ndim == 2:
            df = pd.DataFrame(data)
            filled_df = df.interpolate(method='linear', limit_direction='both', axis=0)
            filled_df = filled_df.ffill()
            filled_df = filled_df.bfill()
            filled_df = filled_df.fillna(0)
            return filled_df.values
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {data.ndim}")
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
        data.x = (data.x - data.x.mean(dim=(0, 1))) / (data.x.std(dim=(0, 1), unbiased=False) + 1e-8)
        
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
        
        # 数据归一化
        data = self.normalize_data(data)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.models[0].parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 转换数据为DataLoader
        from torch_geometric.loader import DataLoader as GeometricDataLoader
        train_loader = GeometricDataLoader([data], batch_size=1, shuffle=True)
        
        # 使用 OneCycleLR 调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # 训练 STGCN-LSTM
        self.models[0].train()
        grad_norms = []
        activation_stats = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_batches = 0
            
            for batch in train_loader:
                try:
                    optimizer.zero_grad()
                    
                    # 记录梯度和激活值统计
                    with torch.set_grad_enabled(True):
                        outputs = self.models[0](
                            batch.x,
                            batch.edge_index,
                            batch.edge_attr,
                            seq_length=10
                        )
                        
                        # 确保目标形状与输出一致
                        target = batch.y.view_as(outputs)  # [batch_size, num_nodes, 1]
                        loss = criterion(outputs, target)
                        
                        # 反向传播
                        loss.backward()
                        
                        # 记录梯度范数
                        total_norm = 0
                        for p in self.models[0].parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        grad_norms.append(total_norm)
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            self.models[0].parameters(),
                            max_norm=1.0
                        )
                        
                        optimizer.step()
                        scheduler.step()
                        
                        epoch_loss += loss.item()
                        valid_batches += 1
                        
                except Exception as e:
                    print(f"Epoch [{epoch+1}] 错误：{str(e)}")
                    continue
            
            # 输出训练统计信息
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, '
                          f'Grad Norm: {total_norm:.4f}, '
                          f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 训练 TemporalTransformer
        temporal_model = self.models[3]
        temporal_model.train()
        temporal_optimizer = torch.optim.Adam(temporal_model.parameters(), lr=lr)
        temporal_criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            try:
                temporal_optimizer.zero_grad()
                temporal_input = data.x.float()  # [batch_size, num_nodes, features]
                temporal_pred = temporal_model(temporal_input)  # [batch_size, num_nodes, 1]
                target = data.y.view_as(temporal_pred)  # [batch_size, num_nodes, 1]
                temporal_loss = temporal_criterion(temporal_pred, target)
                temporal_loss.backward()
                temporal_optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f'Temporal Transformer Epoch [{epoch+1}/{epochs}], Loss: {temporal_loss.item():.4f}')
            except Exception as e:
                print(f"Temporal Transformer Epoch [{epoch+1}] 错误：{str(e)}")
                continue
        
        # 训练其他模型
        X = data.x.numpy().reshape(data.x.size(0) * data.x.size(1), data.x.size(2))
        y = data.y.numpy().reshape(data.y.size(0) * data.y.size(1), 1)
        
        # 确保没有NaN值
        if np.isnan(y).any():
            print("处理目标变量中的NaN值")
            y = handle_nan_values(torch.tensor(y)).numpy()
        
        if np.isnan(X).any():
            print("处理输入特征中的NaN值")
            X = handle_nan_values(torch.tensor(X)).numpy()
        
        # 确保 y 是一维数组
        y = y.ravel()
        
        self.models[1].fit(X, y)
        
        # XGBoost
        self.models[2].fit(X, y)
        
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
        target_shape = (graph_data.x.size(0), graph_data.x.size(1), 1)  # [batch_size, num_nodes, 1]

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
                            )  # [batch_size, num_nodes, 1]
                        else:  # TemporalTransformer
                            pred = model(graph_data.x)  # [batch_size, num_nodes, 1]
                else:
                    pred = model.predict(graph_data.x.numpy().reshape(graph_data.x.size(0) * graph_data.x.size(1), graph_data.x.size(2)))
                    pred = pred.reshape(target_shape)
                
                # 确保预测形状正确
                if isinstance(model, nn.Module):
                    pred = pred.cpu().numpy()
                
                if pred.shape != target_shape:
                    pred = pred.reshape(target_shape)
                    
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model {i} prediction failed: {str(e)}")
                predictions.append(np.zeros(target_shape))
        
        # 堆叠预测并计算分位数
        predictions = np.stack(predictions, axis=0)  # [num_models, batch_size, num_nodes, 1]
        lower = np.quantile(predictions, 0.05, axis=0)
        upper = np.quantile(predictions, 0.95, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        return mean_pred, (lower, upper)

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
    y = node_features @ target_weights  # [num_nodes, 1]
    
    # 扩展 y 到 [batch_size, num_nodes, 1]，假设 batch_size=1
    y = y.unsqueeze(0)  # [1, num_nodes, 1]
    
    return Data(
        x=node_features.unsqueeze(0),  # [1, num_nodes, feature_dim]
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
    graph_data = model.normalize_data(graph_data)  # 数据归一化
    model.train(graph_data)
    pred, (lower, upper) = model.predict(graph_data)
    
    #print(f"预测结果形状: {pred.shape}")
    #print(f"下限形状: {lower.shape}")
    #print(f"上限形状: {upper.shape}")
    
    # 将预测结果从 [1, 144, 1] 转换为 [144, 1]
    pred_2d = pred.squeeze().reshape(-1, 1)  # [144, 1]
    lower_2d = lower.squeeze().reshape(-1, 1)  # [144, 1]
    upper_2d = upper.squeeze().reshape(-1, 1)  # [144, 1]
    
    np.savetxt('predictions.csv', pred_2d, delimiter=',')
    np.savetxt('uncertainty_lower.csv', lower_2d, delimiter=',')
    np.savetxt('uncertainty_upper.csv', upper_2d, delimiter=',')