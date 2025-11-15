import itertools
import pickle
import pandas as pd
import numpy as np
import torch
import torch_geometric.transforms
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import HANConv,SAGEConv,to_hetero, Linear
from torch import Tensor
import tqdm
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau  #自动调整学习率
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
# from FocalLoss import FocalLoss
from hgt_conv_edge_attr import HGTConv
import logging
import time
from torchsummary import summary


## 构建图数据集
class CustomDataset(Dataset):
    """自定义的异构图数据集类"""
    def __init__(self, root, transform=None, pre_transform=None, indices=None):
        super().__init__(root, transform, pre_transform)
        self.file_list = [f for f in os.listdir(root) if f.endswith('.pt')]
        
        if indices is not None:
            # 如果提供了索引，根据索引筛选文件
            self.file_list = [self.file_list[i] for i in indices]

    def len(self):
        """返回数据集中的图数量"""
        return len(self.file_list)

    def get(self, idx):
        """加载并返回指定索引处的图"""
        file_path = os.path.join(self.root, self.file_list[idx])
        data = torch.load(file_path)
        return data
    
    @property
    def processed_file_names(self):
        """返回处理后的文件名列表"""
        return self.filenames
    
    
## 分割图数据集    
# 数据集文件夹路径
root_path = '/home/QFRC/ZQS/Projects/Graph/Scripts_demo/Graphs_demos/Graphs5days_all1.0'

# 创建数据集实例
dataset = CustomDataset(root=root_path)
device = torch.device("cuda:0")

# 分割数据集
# 获取数据集的索引数组
batch_size= 1024
# 数据集的长度
dataset_length = len(dataset)

# 计算各个子集的索引范围
train_size = int(0.7 * dataset_length)
val_size = int(0.2 * dataset_length)
test_size = dataset_length - train_size - val_size  # 剩下的部分为测试集

# 生成顺序索引
indices = list(range(dataset_length))

# 根据比例按顺序切分数据集
train_indices = indices[:train_size]  # 前70%的索引
val_indices = indices[train_size:train_size + val_size]  # 接下来的20%
test_indices = indices[train_size + val_size:]  # 最后的10%

# 根据索引创建数据集的子集
train_dataset = CustomDataset(root=root_path, indices=train_indices)
val_dataset = CustomDataset(root=root_path, indices=val_indices)
test_dataset = CustomDataset(root=root_path, indices=test_indices)

train_data_on_gpu = [data.to(device) for data in train_dataset]
val_data_on_gpu = [data.to(device) for data in val_dataset]
test_dataset_on_gpu = [data.to(device) for data in test_dataset]

# 创建对应的DataLoader
train_loader = DataLoader(train_data_on_gpu, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data_on_gpu, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset_on_gpu, batch_size=batch_size, shuffle=False)


## 回测数据处理

# 5days
# label标签均为当天相较于前一天的涨跌情况；预测下一交易日，实际对应的标签应为下张图的第一个交易日的收盘价

label_indices_pre=[x + 5 for x in val_indices]


# 2023-4-10
# 新的列表，用于存储提取后的值
open_price_pre = []  # 预测天的开盘价
close_price_pre = []  # 预测天的收盘价
open_price_today=[]  # 当天开盘价
close_price_today=[] # 当天收盘价


# 循环遍历每个索引
for i in label_indices_pre:
    # 提取第 i 个数据集中 'stock' 的 (0,0) 位置的值
    open_value = round(dataset[i]['stock'].x[0, 0].item(),2)  # 提取并转换为 float
    open_price_pre.append(open_value)  # 保存到新的列表中
    
    close_value = round(dataset[i]['stock'].x[0, 3].item(),2)  # 提取并转换为 float
    close_price_pre.append(close_value)  # 保存到新的列表中



# 定义所有超参数的列表
hidden_channels1_list = [256, 128, 64]
hidden_channels2_list = [128, 64, 32]
hidden_channels3_list = [128, 64, 32]
out_channels1_list = [32, 16, 8]
out_channels2_list = [2]

# 生成所有的超参数组合
param_combinations = list(itertools.product(hidden_channels1_list, hidden_channels2_list, hidden_channels3_list, out_channels1_list, out_channels2_list))

# 筛选满足条件的超参数组合
filtered_combinations = [
    combination for combination in param_combinations 
    if combination[0] >= combination[1] >= combination[2] >= combination[3] >= combination[4]
]

# 定义要删除的组合
combinations_to_remove = [
    (256, 128, 128, 16, 2),
    (256, 128, 128, 32, 2),
    (256, 128, 128, 8, 2),
    (256, 128, 64, 16, 2),
    (256, 128, 64, 32, 2)
]

# 删除指定的组合
final_combinations = [comb for comb in filtered_combinations if comb not in combinations_to_remove]
final_combinations=filtered_combinations
# print(final_combinations)


# 创建保存日志的目录
log_root_dir = "/mnt/sdb/ZQS/HeteroGraph/StocksPre&Back/5days//浪潮信息/logs"
model_root_dir = "/mnt/sdb/ZQS/HeteroGraph/StocksPre&Back/5days/浪潮信息/models"

os.makedirs(log_root_dir, exist_ok=True)
os.makedirs(model_root_dir, exist_ok=True)


## 构建模型类
class HGTExplicit(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, out_channels1, out_channels2, num_heads, metadata1, metadata2):
        super().__init__()

        # 定义两层 HGT 卷积
        
        self.conv1 = HGTConv(in_channels, hidden_channels1, metadata1, num_heads)
        self.conv2 = HGTConv(hidden_channels1, hidden_channels2, metadata2, num_heads)
        self.conv3 = HGTConv(hidden_channels2, hidden_channels3, metadata2, num_heads)
        # self.conv4 = HGTConv(hidden_channels3, hidden_channels4, metadata2, num_heads)

        # 用于将 connect 节点特征从 5 维转换为 hidden_channels1
        self.connect_lin1 = torch.nn.Linear(5, hidden_channels1)
        self.connect_lin2 = torch.nn.Linear(5, hidden_channels2)
        self.connect_lin3 = torch.nn.Linear(5, hidden_channels3)
        
        self.fs_lin1= torch.nn.Linear(2, hidden_channels1)
        # 用于将 connect 节点特征从 hidden_channels1 转换为 hidden_channels2
        self.fs_lin2 = torch.nn.Linear(2, hidden_channels2)
        self.fs_lin3 = torch.nn.Linear(2, hidden_channels3)
        
        # Dropout 层
        self.dropout = torch.nn.Dropout(p=0.5)
        
        # 输出层
        self.gru=torch.nn.GRU(hidden_channels3, out_channels1)
        self.lstm=torch.nn.LSTM(hidden_channels3, out_channels1)
        #  self._initialize_weights()  # 初始化权重
        
        self.out1 = Linear(hidden_channels3, out_channels1)
        self.out2 = Linear(out_channels1, out_channels2)
        # self.out3 = Linear(out_channels2, out_channels3)
        
        # 初始化 GRU 权重
        # self._initialize_weights()
        # 定义 PyG 的特征归一化变换
        self.normalize = T.NormalizeFeatures()
        

    def forward(self, data):
        
        
        # 对整个异构图进行归一化
        # data = self.normalize(data)
        # 获取 x_dict, edge_index_dict, edge_attr_dict
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        
        connect_initial = x_dict['connect']
        financing_initial = x_dict['financing']
        selling_initial = x_dict['selling']

        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {node_type: F.leaky_relu(x) for node_type, x in x_dict.items()}
        # 应用Dropout
        x_dict = {node_type: self.dropout(x) for node_type, x in x_dict.items()} #神经元已经提取了特征，再对这些特征进行随机“丢弃”可以更好地防止过拟合。
               
        # 移除不再使用的节点
        for key in ['connect', 'financing', 'selling']:
            if key in x_dict:
                del x_dict[key]

        for edge_type in [('connect', 'invest', 'stock'), ('financing', 'invest', 'stock'), ('selling', 'invest', 'stock')]:
            if edge_type in edge_index_dict:
                del edge_index_dict[edge_type]
            if edge_type in edge_attr_dict:
                del edge_attr_dict[edge_type]
        
        # 第二层 HGT 卷积
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {node_type: F.leaky_relu(x) for node_type, x in x_dict.items()}
        x_dict = {node_type: self.dropout(x) for node_type, x in x_dict.items()} 

        # 第三层 HGT 卷积
        x_dict = self.conv3(x_dict, edge_index_dict,edge_attr_dict)
        x_dict = {node_type: F.leaky_relu(x) for node_type, x in x_dict.items()}
        x_dict = {node_type: self.dropout(x) for node_type, x in x_dict.items()} 
        
        
        
        out=F.leaky_relu(self.out1(x_dict['stock']))
        out=self.out2(out)[0::12]

        return out
    
# 传参        
in_channels = {
    'stock': 25,
    'other': 768,
    'connect': 5,
    'financing': 2,
    'selling': 2
}

metadata1 = (['stock', 'other', 'connect', 'financing', 'selling'], [
    ('stock', 'spearman', 'stock'),
    ('connect', 'invest', 'stock'),
    ('financing', 'invest', 'stock'),
    ('selling', 'invest', 'stock'),
    ('stock', 'relationship', 'stock'),
    ('stock', 'relationship', 'other'),
    ('other', 'relationship', 'stock'),
    ('other', 'relationship', 'other')
])

metadata2 = (['stock', 'other',], [
    ('stock', 'spearman', 'stock'),
    ('stock', 'relationship', 'stock'),
    ('stock', 'relationship', 'other'),
    ('other', 'relationship', 'stock'),
    ('other', 'relationship', 'other')
])

num_heads =32

## 训练、验证与回测
criterion = torch.nn.CrossEntropyLoss()
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    start_time = time.time()  # 开始计时
    
    for data in loader:
        optimizer.zero_grad()
        
        # 将数据移动到指定的GPU
        # print(data['stock'].y[:, 0].unsqueeze(1).shape)
        y = data['stock'].y[:, 0].to(device, dtype=torch.long)  # (batch_size, num_stocks) -> (num_stocks, batch_size)
        
        
        # 获取模型输出
        output = model(data)  # 输出形状: (num_stocks, batch_size)
        loss = criterion(output, y)  # 计算整体损失
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)    

def evaluate(model,loader):
    model.eval()
    total_loss = 0
    labels = []
    preds = []
    probs = []
    
    with torch.no_grad():
        for data in loader:
            
            y = data['stock'].y[:, 0].to(device, dtype=torch.long)  # (batch_size, num_stocks) -> (num_stocks, batch_size)
            
            output = model(data)  # 输出形状: (num_stocks, batch_size)
            # print(output)
            loss = criterion(output, y)
            total_loss += loss.item()
            
             # 获取预测概率
            probs_batch = torch.softmax(output, dim=1)[:, 1]  # 只取正类的概率
            
            # 将logits转为概率并转为二元预测
            preds_batch = torch.argmax(output, dim=1)  # 获取类别预测
            
            labels.extend(y.cpu().numpy())
            preds.extend(preds_batch.cpu().numpy())
            probs.extend(probs_batch.cpu().numpy())
            

    # 计算评价指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=1)
    recall = recall_score(labels, preds, zero_division=1)
    f1 = f1_score(labels, preds, zero_division=1)
    mcc= matthews_corrcoef(labels, preds)
    
    # 计算AUC
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0  # 如果计算AUC时遇到问题，例如只有一种类标签

    # print(f'Accuracy: {accuracy:.4f}, Mcc: {mcc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')
    return total_loss / len(loader) , accuracy, mcc, precision, recall, f1, auc, preds


for hidden_channels1, hidden_channels2, hidden_channels3, out_channels1, out_channels2 in final_combinations:
    # 开始计时
    start_time = time.time()
    
    # 创建文件夹用于保存日志和模型
    model_save_dir = os.path.join(model_root_dir, f"{hidden_channels1}_{hidden_channels2}_{hidden_channels3}_{out_channels1}")
    log_file = os.path.join(log_root_dir, f"{hidden_channels1}_{hidden_channels2}_{hidden_channels3}_{out_channels1}.log")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 检查是否有写入权限
    if not os.access(log_root_dir, os.W_OK):
        print(f"Permission denied: Unable to write to log directory {log_root_dir}")
        continue
    
    # 清理旧的日志处理器，避免冲突
    logger = logging.getLogger(f"{hidden_channels1}_{hidden_channels2}_{hidden_channels3}_{out_channels1}")
    if logger.hasHandlers():
        logger.handlers.clear()  # 清除现有的处理器
    
    logger.setLevel(logging.INFO)
    
    
    # 检查是否已有处理器，避免重复添加
    if not logger.hasHandlers():
        # 文件处理器
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        
        # 添加处理器到记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"Training with hidden1={hidden_channels1}, hidden2={hidden_channels2}, hidden3={hidden_channels3}, out1={out_channels1}, out2={out_channels2}")

    
    # 每次记录日志后强制刷新缓冲区
    def log_flush():
        for handler in logger.handlers:
            handler.flush()

    logger.info("Starting model training...")
    log_flush()
            
    # 检查日志文件是否创建成功
    if os.path.exists(log_file  ):
        print(f"Log file successfully created: {log_file}")
    else:
        print("Log file creation failed. Check permissions or path.")    
    # 初始化模型和优化器
    model = HGTExplicit(in_channels, hidden_channels1, hidden_channels2, hidden_channels3, out_channels1, out_channels2, num_heads, metadata1, metadata2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # best_acc=0.5
    best_mcc=0.2
    epochs=10000
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss = train(model, train_loader, optimizer)
        val_loss, val_acc, val_mcc, val_precision, val_recall, val_f1, val_auc, preds = evaluate(model, val_loader)


        # 回测代码（整合您的逻辑）
        initial_capital = 100000  # 初始资金
        capital = initial_capital
        position = 0  # 初始没有持仓
        cash = capital

        capital_history = []  # 记录每日资产
        bought = False  # 标记是否已买入

        # 假设 `preds` 已在 `evaluate` 函数中计算得到
        for i in range(len(preds) - 1):  # 循环到倒数第二天
            if not bought and preds[i] == 1:  # 如果还未买入且预测当天上涨
                # print(i)
                position = cash // open_price_pre[i]  # 以当天的开盘价买入
                cash -= position * open_price_pre[i]
                bought = True
            elif bought and preds[i] == 0:  # 如果已经持仓且预测当天下跌
                cash += position * open_price_pre[i]
                position = 0  # 清空持仓
                bought = False
            
            # 记录每日的总资金（现金 + 持仓市值）
            total_assets = cash + (position * close_price_pre[i] if bought else 0)
            capital_history.append(round(total_assets, 2))

        # 最后一天处理
        if preds[-1] == 1 and position > 0:
            cash += position * close_price_pre[-1]  # 以最后一天的收盘价卖出
            position = 0
        elif preds[-1] == 0 and position > 0:
            cash += position * open_price_pre[-1]  # 以最后一天的开盘价卖出
            position = 0

        # 最后一天的总资金记录
        capital_history.append(round(cash, 2))  # 最后一天全部资金为现金

        # 计算累计收益和收益率
        final_capital = capital_history[-1]
        cumulative_return = final_capital - initial_capital
        cumulative_rate_of_return = (final_capital / initial_capital - 1) * 100  # 百分比形式
        
        # 将资金历史转换为 NumPy 数组，便于计算
        capital_array = np.array(capital_history)

        # 最大回撤计算
        running_max = np.maximum.accumulate(capital_array)  # 历史资金的最大值
        drawdowns = (running_max - capital_array) / running_max  # 计算回撤比例
        max_drawdown = np.max(drawdowns)  # 最大回撤

        # 夏普比率计算（基于155天，不年化）
        daily_returns = np.diff(capital_array) / capital_array[:-1]  # 计算每日收益率
        mean_daily_return = np.mean(daily_returns)  # 日均收益率
        std_daily_return = np.std(daily_returns)  # 日收益率标准差

        sharpe_ratio = mean_daily_return / std_daily_return if std_daily_return > 0 else 0  # 不年化夏普比率

        # 记录训练和验证损失
        logger.info(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        logger.info(f'Accuracy: {val_acc:.4f}, MCC: {val_mcc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, CR: {cumulative_return:.4f}, CRR: {cumulative_rate_of_return:.4f}')
        log_flush()
        
        # 保存满足条件的模型
        if val_mcc > 0.3 or (val_mcc > 0.2 and val_acc>0.6 and val_recall > 0.6 and cumulative_rate_of_return>0):
        # if val_acc > 0:
            
            model_save_path = os.path.join(model_save_dir, f"Epoch_{epoch}_ACC_{val_acc:.4f}_MCC_{val_mcc:.4f}_Precision_{val_precision:.4f}_Recall_{val_recall:.4f}_F1_{val_f1:.4f}_AUC_{val_auc:.4f}_CR_{cumulative_return:.4f}_CRR_{cumulative_rate_of_return:.4f}_MDD_{max_drawdown:.4f}_SP_{sharpe_ratio:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved at epoch {epoch+1} with Accuracy: {val_acc:.4f}, MCC: {val_mcc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, CR: {cumulative_return:.4f}, CRR: {cumulative_rate_of_return:.4f}")
            
    
    log_flush()
    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time  # 计算运行时间
    logger.info(f"Total time for hidden_channels combination {hidden_channels1}_{hidden_channels2}_{hidden_channels3}_{out_channels1}: {elapsed_time:.2f} seconds")

    # 强制刷新缓冲区
    for handler in logger.handlers:
        handler.flush()