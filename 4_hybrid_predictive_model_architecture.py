import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib  # 新增：导入joblib
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
        max_seq_len = 10000  # 根据需要调整
        pe = torch.zeros(max_seq_len, self.d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, x):
        # 假设 x 原本是 [batch_size, num_nodes, feature_dim]
        # 如果要用 Transformer 的时序维度，可以人为假设 num_nodes 即为时序长度
        # 或自己重排维度，以形如 [batch_size, seq_len, d_model] 的方式传入。

        batch_size, seq_len, features = x.size()

        # 如果 features != self.d_model，需要额外投影
        if features != self.d_model:
            x = nn.Linear(features, self.d_model)(x)

        # 加入位置编码
        pos_enc = self.positional_encoding[:seq_len, :].to(x.device)
        x = x + pos_enc.unsqueeze(0)

        # 进入 Transformer
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
                    # 检查初始化值
                    mean = param.data.mean().item()
                    std = param.data.std().item()
                    if abs(mean) > 0.1 or std > 1.0:
                        nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        # 学习率预热
        self.warmup_steps = 100
        self.current_step = 0
    
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
    filled_count = 0  # 计数用0填充的NaN值数量

    # 调试：检查处理前的数据状态
    #print("开始处理 NaN 值前的数据概览：")
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        nan_counts = data.isna().sum()
        #print(f"每列的NaN数量:\n{nan_counts}")
        #print(f"总NaN数量: {data.isna().sum().sum()}")
    elif isinstance(data, torch.Tensor):
        nan_indices = torch.nonzero(torch.isnan(data), as_tuple=False)
        #print(f"NaN的位置索引: {nan_indices}")
        #print(f"总NaN数量: {torch.isnan(data).sum().item()}")
    elif isinstance(data, np.ndarray):
        nan_indices = np.argwhere(np.isnan(data))
        #print(f"NaN的位置索引:\n{nan_indices}")
        #print(f"总NaN数量: {np.isnan(data).sum()}")
    else:
        print("未知数据类型，无法打印NaN位置。")

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        # 首先尝试线性插值
        filled_data = data.interpolate(method='linear', limit_direction='both', axis=0)
        #print(f"插值后NaN数量: {filled_data.isna().sum().sum()}")

        # 使用前向填充
        filled_data = filled_data.ffill()
        #print(f"前向填充后NaN数量: {filled_data.isna().sum().sum()}")

        # 使用后向填充
        filled_data = filled_data.bfill()
        #print(f"后向填充后NaN数量: {filled_data.isna().sum().sum()}")

        # 计算填充的NaN数量
        filled_count += filled_data.isna().sum().sum()  # 计算填充后仍然是NaN的数量

        # 如果仍然有NaN，用0填充
        filled_data = filled_data.fillna(0)
        filled_count += (data.isna().sum().sum() - filled_data.isna().sum().sum())  # 计算用0填充的数量

        #print(f"最终填充后的NaN数量: {filled_data.isna().sum().sum()}")
        return filled_data, filled_count

    elif isinstance(data, torch.Tensor):
        # 检查数据维度
        if data.dim() == 3:
            batch_size, num_nodes, num_features = data.size()
            # 重塑为二维 [batch_size * num_nodes, num_features]
            reshaped_data = data.view(-1, data.size(-1))
            # 转换为DataFrame进行处理
            df = pd.DataFrame(reshaped_data.numpy())
            filled_df, count = handle_nan_values(df)
            filled_tensor = torch.tensor(filled_df.values, dtype=data.dtype).view(batch_size, num_nodes, num_features)
            return filled_tensor, count
        elif data.dim() == 2:
            df = pd.DataFrame(data.numpy())
            filled_df, count = handle_nan_values(df)
            return torch.tensor(filled_df.values, dtype=data.dtype), count
        else:
            raise ValueError(f"Unsupported tensor dimensions: {data.dim()}")
    elif isinstance(data, np.ndarray):
        # 将numpy数组转换为DataFrame处理
        if data.ndim == 3:
            batch_size, num_nodes, num_features = data.shape
            reshaped_data = data.reshape(-1, data.shape[-1])
            df = pd.DataFrame(reshaped_data)
            filled_df, count = handle_nan_values(df)
            filled_tensor = torch.tensor(filled_df.values, dtype=torch.float32).view(batch_size, num_nodes, num_features)
            return filled_tensor, count
        elif data.ndim == 2:
            df = pd.DataFrame(data)
            filled_df, count = handle_nan_values(df)
            return filled_df.values, count
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
            TemporalTransformer(
                d_model=64,
                nhead=8,
                num_encoder_layers=6,
                dim_feedforward=256,
                dropout=0.1
            )
        ]
        
        # 根据需要自由调整
        self.learning_rate = 3e-4
        self.l1_lambda = 1e-6
        self.l2_lambda = 1e-5
        self.epochs = 1000
        self.batch_size = 2
        # warmup_steps 可以先设小一点，让其迅速过渡到正常学习率
        self.warmup_steps = 10
        self.current_step = 0

    def normalize_data(self, data):
        # 归一化前检查
        #print("归一化前的数据概览：")
        #print(f"data.x 形状: {data.x.shape}")
        #print(f"data.x 中的NaN总数: {torch.isnan(data.x).sum().item()}")

        # 对节点特征进行标准化
        data.x = (data.x - data.x.mean(dim=(0, 1))) / (data.x.std(dim=(0, 1), unbiased=False) + 1e-8)
        
        # 归一化后检查
        #print("归一化后的数据概览：")
        #print(f"data.x 中的NaN总数: {torch.isnan(data.x).sum().item()}")

        # 对边属性进行标准化
        data.edge_attr = (data.edge_attr - data.edge_attr.mean()) / (data.edge_attr.std() + 1e-8)
        
        # 检查边属性中的NaN
        #print(f"data.edge_attr 中的NaN总数: {torch.isnan(data.edge_attr).sum().item()}")
        
        return data

    def train(self, data, epochs=None, lr=None, batch_size=None,
              l1_lambda=None, l2_lambda=None):
        """训练 STGCN-LSTM 与其它模型"""
        # 如果外界没设置，使用默认值
        epochs = epochs or self.epochs
        lr = lr or self.learning_rate
        batch_size = batch_size or self.batch_size
        l1_lambda = l1_lambda or self.l1_lambda
        l2_lambda = l2_lambda or self.l2_lambda

        # 预处理、归一化与 NaN 处理逻辑，保持原来即可
        # ------------------------- 省略 -----------------------
        # （此处假定 handle_nan_values()、normalize_data() 等函数仍在类里或外部可用）

        # 定义损失和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.models[0].parameters(),
            lr=lr,
            weight_decay=l2_lambda
        )

        from torch_geometric.loader import DataLoader as GeometricDataLoader
        train_loader = GeometricDataLoader([data], batch_size=1, shuffle=True)

        # 这里可改用最基础的 StepLR 或其它调度器
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=300,  # 每 300 epoch 降一次学习率
            gamma=0.5       # 学习率衰减为原来的 0.5
        )

        print("===== 开始训练 STGCN-LSTM =====")
        self.models[0].train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                outputs = self.models[0](
                    batch.x,        # [batch_size, num_nodes, node_feat_dim]
                    batch.edge_index,
                    batch.edge_attr,
                    seq_length=10    # 假设有 10 个时序或这只是个占位
                )
                target = batch.y.view_as(outputs)  # shape 对齐
                loss = criterion(outputs, target)

                # L1 正则
                l1_norm = 0
                for p in self.models[0].parameters():
                    l1_norm += p.abs().sum()
                loss = loss + l1_lambda * l1_norm

                # 后向传播
                loss.backward()

                # 梯度裁剪（避免梯度爆炸）
                nn.utils.clip_grad_norm_(
                    self.models[0].parameters(),
                    max_norm=1.0
                )

                # 学习率预热示例
                if self.current_step < self.warmup_steps:
                    warmup_scale = float(self.current_step + 1) / self.warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * warmup_scale

                optimizer.step()
                self.current_step += 1

                epoch_loss += loss.item()
                valid_batches += 1

            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                # scheduler 每个 epoch 调度一次
                scheduler.step()

                # 每隔若干 epoch 打印
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], STGCN-LSTM Loss: {avg_loss:.6f}")

        print("===== STGCN-LSTM 训练结束 =====\n")

        # ============= TemporalTransformer 训练 =================
        # 让 temporal_model 同样处理多批次数据，而不是只训练单一 batch
        print("===== 开始训练 TemporalTransformer =====")
        temporal_model = self.models[3]
        temporal_model.train()
        temporal_optimizer = torch.optim.Adam(
            temporal_model.parameters(),
            lr=1e-4  # 可根据情况进行微调
        )
        temporal_criterion = nn.MSELoss()

        # 示例：假设我们用同一个 data，模拟多批次时间序列（实际应准备不同时间片的数据）
        # 这里只是示例做法，提示您应在真实场景下把 data.x 分成多段或多样本 DataLoader。
        for epoch in range(epochs):
            temporal_optimizer.zero_grad()

            # 构造一个假定的时序输入 temporal_input
            # 假设 data.x 是 [1, num_nodes, feat_dim]，为了制造时序，我们可以复制/拼接
            # 例如这里拼接 K 次变成 [batch_size, seq_len, feat_dim]
            # 仅演示做法，您应使用真实的时序输入
            K = 5  # 假设有 5 个时间步
            temporal_input = data.x.repeat(1, K, 1)      # [1, num_nodes*K, feat_dim]
            # 重排维度为 [1, K, num_nodes*feat_dim] 或 [1, K, feat_dim]
            # 为了简化，假设 seq_len = K, features = (num_nodes * feat_dim)
            batch_size, all_len, feat_dim = temporal_input.shape
            seq_len = K
            # 把 num_nodes * feat_dim 合并到 feature 维度
            merged_feat = feat_dim * all_len // K  # = num_nodes * feat_dim
            # [1, K, num_nodes*feat_dim]
            temporal_input = temporal_input.view(batch_size, K, -1).float()

            # forward
            temporal_pred = temporal_model(temporal_input)
            # 这里 temporal_pred 会被扩展回 [batch_size, num_nodes, 1]
            # 如果 data.y 只有 [1, num_nodes, 1]，我们可能要重复同样的次数 K
            # 或者只对最后时刻进行监督
            # 仅示例：让 target 与 pred 的维度相同
            target = data.y.float().repeat(1, temporal_pred.size(1), 1)
            # shape: [1, K, 1], => 还要扩展到 [1, K, num_nodes?] 视实际需求而定

            # 仅演示：若 pred 最后被 repeat 成 [batch_size, 144, 1]，需对 target 也做对应 reshape
            # 这里为了简单假定 num_nodes=K=5
            # 若实际 num_nodes=144，可先把 temporal_pred 改造成 [1, 144, 1] 后跟 target 对齐
            # 具体要看您对时序维度和节点维度的设计
            # 临时做法: 令 target 与 pred 形状一致
            if temporal_pred.shape != target.shape:
                target = target.expand_as(temporal_pred)

            temporal_loss = temporal_criterion(temporal_pred, target)
            temporal_loss.backward()

            # 如果梯度过大可以裁剪
            nn.utils.clip_grad_norm_(temporal_model.parameters(), 1.0)

            temporal_optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Temporal Loss: {temporal_loss.item():.6f}")

        print("===== TemporalTransformer 训练结束 =====\n")

        # ============= 训练 HGBR 和 XGBoost =============
        # 与原逻辑一致
        # ...
        # 省略下方 fit
        # ...

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
    
    # 检查 y 是否包含 NaN
    if torch.isnan(y).any():
        print("警告：目标标签 y 中包含 NaN 值")
        print(y)
    
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
        #print("权重矩阵数据：")
        #print(df.head())

        # 分离非数值列（IOC_Code）与数值列（Year, Weight）
        non_numeric_cols = ['IOC_Code']
        numeric_cols = [col for col in df.columns if col not in non_numeric_cols]

        # 检查数值列中的NaN值
        if df[numeric_cols].isna().any().any():
            print("数值列中存在NaN值，以下是包含NaN值的行：")
            print(df[df[numeric_cols].isna().any(axis=1)])
            # 将NaN值替换为0
            df[numeric_cols] = df[numeric_cols].fillna(0)

        # 检查重复的 IOC_Code 和 Year 组合
        duplicates = df[df.duplicated(subset=['IOC_Code', 'Year'], keep=False)]
        if not duplicates.empty:
            print("发现重复的 IOC_Code 和 Year 组合，使用 pivot_table 进行聚合。")
            print(duplicates)

        # 使用 pivot_table 代替 pivot，并指定聚合函数
        weight_matrix_pivot = df.pivot_table(
            index='IOC_Code',
            columns='Year',
            values='Weight',
            aggfunc='mean'  # 根据需要选择适当的聚合函数
        ).fillna(0)

        # 将权重矩阵转换为张量
        numeric_data = weight_matrix_pivot.values  # 形状为 [国家数, 年份数]
        weight_countries = weight_matrix_pivot.index.tolist()
        numeric_data = np.nan_to_num(numeric_data)  # 确保无NaN
        weight_matrix_tensor = torch.tensor(numeric_data, dtype=torch.float32)

        return weight_matrix_tensor, weight_countries
    
    def load_coach_scores(self):
        data = np.genfromtxt(
            f'{self.data_dir}/sport_pagerank_scores.csv',
            delimiter=',',
            skip_header=1,
            dtype=str,
            filling_values=''
        )
        #print("教练评分数据：")
        #print(data[:5])  # 输出前5行数据
        
        # 提取有效的分数和国家
        scores = data[:, 1].astype(float)  # 假设分数在第二列
        countries = data[:, 0]  # 假设国家在第一列
        valid_mask = ~np.isnan(scores)  # 创建有效掩码以过滤NaN值
        
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
            
        numeric_data = numeric_data.applymap(clean_value)
        
        #print("事件专业化数据：")
        #print(numeric_data.head())  # 输出前几行数据
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
            
        #print("奖牌相关性数据：")
        #print(df.head())  # 输出前几行数据
        return torch.tensor(numeric_data.values, dtype=torch.float32)
    
    def load_trade_flow(self):
        import pandas as pd
        # Read CSV with header
        df = pd.read_csv(f'{self.data_dir}/world-gdp-data.csv')
        
        # Convert to numeric, coercing errors to NaN
        numeric_data = df.apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values with 0 and convert to tensor
        #print("贸易流数据：")
        #print(df.head())  # 输出前几行数据
        return torch.tensor(numeric_data.fillna(0).values, dtype=torch.float32)
    
    def load_gdp(self):
        import pandas as pd
        # Read CSV with header
        df = pd.read_csv(f'{self.data_dir}/world-gdp-data.csv')
        
        # Convert GDP column to numeric, coercing errors to NaN
        gdp_values = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        # Fill NaN values with 0 and convert to tensor
        #print("GDP数据：")
        #print(df.head())  # 输出前几行数据
        return torch.tensor(gdp_values.fillna(0).values, dtype=torch.float32)
        
    def get_graph_data(self):
        weight_matrix, weight_countries = self.load_weight_matrix()
        coach_scores, coach_countries = self.load_coach_scores()
        event_specialization = self.load_event_specialization()
        medal_corr = self.load_medal_correlation()
        trade_flow = self.load_trade_flow()
        gdp = self.load_gdp()
        
        '''
        # 检查各个数据集是否包含 NaN
        print("检查各个数据集中的 NaN 值：")
        print(f"weight_matrix NaN: {torch.isnan(weight_matrix).sum().item()}")
        print(f"coach_scores NaN: {torch.isnan(coach_scores).sum().item()}")
        print(f"event_specialization NaN: {torch.isnan(event_specialization).sum().item()}")
        print(f"medal_corr NaN: {torch.isnan(medal_corr).sum().item()}")
        print(f"trade_flow NaN: {torch.isnan(trade_flow).sum().item()}")
        print(f"gdp NaN: {torch.isnan(gdp).sum().item()}")
        '''

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
    model.train(graph_data, epochs=1000, lr=0.01, batch_size=32, l1_lambda=1e-5, l2_lambda=1e-4)
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
    
    # 保存训练好的模型
    model.save_models('c:/Users/Administrator/Desktop/美赛')