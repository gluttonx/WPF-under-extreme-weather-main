#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用已训练的模型生成预测结果，保存为CSV格式
"""
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import scipy.io as scio
import random
import model
from torch.nn.utils import weight_norm

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, mode='pre', kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [model.TemporalBlock_v2(in_channels, out_channels, kernel_size,  stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, mode=mode)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## Define forecasting model
class model_fore(nn.Module):
    def __init__(self,input_channel_fore, output_channel_fore, mode, output_size_baselearner=1,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super().__init__()
        self.mode = mode
        self.tcn = TemporalConvNet(input_channel_fore, output_channel_fore, mode, kernel_size, dropout)
        self.emb_dropout = emb_dropout
        self.drop = nn.Dropout(emb_dropout)
        self.fore_baselearner = nn.Linear(output_channel_fore[-1], output_size_baselearner)
        self.init_weights()
    def init_weights(self):
        self.fore_baselearner.bias.data.fill_(0)
        self.fore_baselearner.weight.data.normal_(0, 0.01)
    def get_trainable_params(self):
        if self.mode == 'pre':
            return self.parameters()
        else:
            trainable_params = []
            for module in self.tcn.modules():
                if hasattr(module, 'get_trainable_params'):
                    trainable_params.extend(module.get_trainable_params())
            trainable_params.extend(self.fore_baselearner.parameters())
            return trainable_params
    def forward(self, x):
        y = self.drop(x)
        y = self.tcn(y.transpose(1, 2)).transpose(1, 2)
        y = self.fore_baselearner(y)
        return y.contiguous()

print("="*70)
print("加载数据和模型...")
print("="*70)

## Load data
seed_torch(seed=1029)
dataFile = '58wf_4_train'  # 更新后的数据文件（标幺值）
wf_1  = scio.loadmat(dataFile)
p=wf_1['p_1h']
nwp=wf_1['nwp_1h']

P_load1=p[:,0]
P_load=P_load1.reshape(np.size(P_load1,axis=0),-1)
P_nwp1=nwp
nwp_index=[0,1,2,3,4]
for i in range(5):
    if i==0:
        P_nwp=P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)
    else:
        P_nwp=np.concatenate((P_nwp,P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)),axis=1)

# Define Parameters
dem_realp=1
len_realp=12
Cap=50  # 总装机容量 (MW)
m=365
d=24
ooo=365
# 数据已经是标幺值，不需要再归一化
Series_day = P_load.reshape(-1,dem_realp)
nwp_day = (P_nwp/np.max(abs(P_nwp),axis=0)).reshape(-1,dem_realp*np.size(P_nwp,axis=1))
dem_realc=np.size(P_nwp,axis=1)

# Prepare test data
test_target_p_=Series_day[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
test_target_p=test_target_p_.reshape(-1,len_realp,dem_realp)
test_input_c_=nwp_day[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
test_input_c=test_input_c_.reshape(-1,len_realp,dem_realc)

Test_target_p=torch.tensor(test_target_p,dtype=torch.float32)
Test_input_c=torch.tensor(test_input_c,dtype=torch.float32)

# Prepare training data
nwp_conven_00=wf_1['nwp_conven_']
p_conven_00=wf_1['p_conven']
p_conven_=p_conven_00  # 数据已经是标幺值
nwp_conven_=np.empty([1,5],dtype=object)
for i in range(np.size(nwp_conven_00,axis=1)):
    nwp_conven_[0,i]=nwp_conven_00[:,i].reshape(-1,1)/np.max(abs(P_nwp[:,i]),axis=0)
for i_nwp in range(np.size(nwp_conven_, axis=1)):
    if i_nwp == 0:
        nwp_conven_1 = nwp_conven_[0, i_nwp].transpose(1, 0)
        nwp_conven_1 = nwp_conven_1[:, :, np.newaxis]
    else:
        nwp_conven_0 = nwp_conven_[0, i_nwp].transpose(1, 0)
        nwp_conven_1 = np.concatenate((nwp_conven_1, nwp_conven_0[:, :, np.newaxis]), axis=2)
p_conven_1 = p_conven_.transpose(1, 0)
p_conven_1 = p_conven_1[:, :, np.newaxis]
train_target_p = p_conven_1
Train_target_p=torch.tensor(train_target_p,dtype=torch.float32)
Train_input_c=torch.tensor(nwp_conven_1,dtype=torch.float32)

# Define device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0=torch.device("cpu")
print(f"使用设备: {device}")

# Create model
model_fore_test_task_query = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='test_task_support')
model_fore_test_task_query = model_fore_test_task_query.to(device)

print("\n生成预测结果...")

# Dictionary to store predictions
predictions = {}

# Generate predictions from different models
model_names = [
    "Extreme_Weather_Model_1",
    "Extreme_Weather_Model_2", 
    "Extreme_Weather_Model_3",
    "Extreme_Weather_Model_4",
    "Meta_Learning_Model",
    "Pre_Training_Model"
]

for i_model in range(6):
    with torch.no_grad():
        if i_model<4:
            print(f"  加载模型: model_fore_test_task_support_{i_model}.pth")
            model_fore_test_task_query.load_state_dict(torch.load(f"model_fore_test_task_support_{i_model}.pth"))
        elif i_model==4:
            # Meta Learning Model：优先使用train_task_query，不存在则使用预训练
            import os
            if os.path.exists("model_fore_train_task_query.pth"):
                print(f"  加载模型: model_fore_train_task_query.pth（元训练模型）")
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_train_task_query.pth"))
            elif os.path.exists("model_fore_pre_federated.pth"):
                print(f"  加载模型: model_fore_pre_federated.pth（跳过了元训练）")
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre_federated.pth"))
            else:
                print(f"  加载模型: model_fore_pre.pth（跳过了元训练）")
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre.pth"))
        elif i_model==5:
            # Pre-training Model：优先使用联邦版
            import os
            if os.path.exists("model_fore_pre_federated.pth"):
                print(f"  加载模型: model_fore_pre_federated.pth（联邦预训练）")
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre_federated.pth"))
            else:
                print(f"  加载模型: model_fore_pre.pth（单场站预训练）")
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre.pth"))
        
        Test_input_c_device = Test_input_c.to(device)
        Test_output_query=model_fore_test_task_query(Test_input_c_device)
        test_outputs_query=Test_output_query.to(device0)
        test_outputs_query_=np.array(test_outputs_query.reshape(-1,dem_realp))
        
        # Store predictions (keep in per unit, will convert to MW later)
        predictions[model_names[i_model]] = test_outputs_query_.flatten().tolist()

# Get training predictions
print("\n  生成训练集预测...")
model_fore_pre_temp = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='pre')
model_fore_pre_temp = model_fore_pre_temp.to(device)
model_fore_pre_temp.load_state_dict(torch.load("model_fore_pre.pth"))
with torch.no_grad():
    Train_input_c_device = Train_input_c.to(device)
    train_outputs_pre = model_fore_pre_temp(Train_input_c_device).to(device0)

train_outputs_pre_=np.array(train_outputs_pre.reshape(-1,dem_realp))
train_target_p_=train_target_p.reshape(-1,dem_realp)
test_target_p_=test_target_p.reshape(-1,dem_realp)

# Create test results DataFrame
print("\n创建CSV文件...")
test_df = pd.DataFrame()
test_df['Time_Index'] = range(len(test_target_p_.flatten()))
# 标幺值和MW值都保存
test_df['True_Power_pu'] = test_target_p_.flatten().round(6)
test_df['True_Power_MW'] = (test_target_p_.flatten() * Cap).round(4)

for model_name in model_names:
    # 标幺值
    test_df[f'Pred_{model_name}_pu'] = [round(x, 6) for x in predictions[model_name]]
    # MW值
    test_df[f'Pred_{model_name}_MW'] = [round(x * Cap, 4) for x in predictions[model_name]]

# Calculate errors (in per unit)
for model_name in model_names:
    test_df[f'Error_{model_name}_pu'] = (test_df[f'Pred_{model_name}_pu'] - test_df['True_Power_pu']).round(6)
    test_df[f'Error_{model_name}_MW'] = (test_df[f'Pred_{model_name}_MW'] - test_df['True_Power_MW']).round(4)

# Create training results DataFrame
train_df = pd.DataFrame()
train_df['Time_Index'] = range(len(train_target_p_.flatten()))
train_df['True_Power_pu'] = train_target_p_.flatten().round(6)
train_df['True_Power_MW'] = (train_target_p_.flatten() * Cap).round(4)
train_df['Pred_Pre_Training_Model_pu'] = train_outputs_pre_.flatten().round(6)
train_df['Pred_Pre_Training_Model_MW'] = (train_outputs_pre_.flatten() * Cap).round(4)
train_df['Error_Pre_Training_Model_pu'] = (train_df['Pred_Pre_Training_Model_pu'] - train_df['True_Power_pu']).round(6)
train_df['Error_Pre_Training_Model_MW'] = (train_df['Pred_Pre_Training_Model_MW'] - train_df['True_Power_MW']).round(4)

# Calculate statistics (using per unit values, displayed as percentage)
print("\n计算评估指标（百分比形式）...")
stats_data = []
for model_name in model_names:
    # 使用标幺值计算
    pred_col_pu = f'Pred_{model_name}_pu'
    true_vals_pu = test_df['True_Power_pu'].values
    pred_vals_pu = test_df[pred_col_pu].values
    
    # 标幺值指标转换为百分比 (×100)
    mae_percent = np.mean(np.abs(true_vals_pu - pred_vals_pu)) * 100
    rmse_percent = np.sqrt(np.mean((true_vals_pu - pred_vals_pu)**2)) * 100
    mape = np.mean(np.abs((true_vals_pu - pred_vals_pu) / (true_vals_pu + 1e-8))) * 100
    r2 = 1 - (np.sum((true_vals_pu - pred_vals_pu)**2) / np.sum((true_vals_pu - np.mean(true_vals_pu))**2))
    
    stats_data.append({
        'Model': model_name,
        'MAE_%': round(mae_percent, 4),
        'RMSE_%': round(rmse_percent, 4),
        'MAPE_%': round(mape, 4),
        'R2_Score': round(r2, 4)
    })

stats_df = pd.DataFrame(stats_data)

# Save to CSV
print("\n保存CSV文件...")
test_df.to_csv('test_predictions.csv', index=False, encoding='utf-8-sig')
train_df.to_csv('train_predictions.csv', index=False, encoding='utf-8-sig')
stats_df.to_csv('model_performance_metrics.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*70)
print("✓✓✓ CSV文件已生成！")
print("="*70)
print("\n生成的文件:")
print(f"  📊 test_predictions.csv ({len(test_df)} 行)")
print("     - Time_Index: 时间索引")
print("     - True_Power_MW: 真实功率 (MW)")
print("     - Pred_*_MW: 各模型预测功率 (MW)")
print("     - Error_*_MW: 预测误差 (MW)")
print(f"\n  📊 train_predictions.csv ({len(train_df)} 行)")
print("     - Time_Index: 时间索引")
print("     - True_Power_MW: 真实功率 (MW)")
print("     - Pred_Pre_Training_Model_MW: 预训练模型预测 (MW)")
print("     - Error_Pre_Training_Model_MW: 预测误差 (MW)")
print(f"\n  📈 model_performance_metrics.csv (评估指标)")
print("     - MAE_%: 平均绝对误差（百分比形式）")
print("     - RMSE_%: 均方根误差（百分比形式）")
print("     - MAPE_%: 平均绝对百分比误差")
print("     - R2_Score: 决定系数")
print("\n数据说明:")
print("  - 功率值同时提供标幺值(_pu)和实际功率(_MW)")
print("  - 评估指标采用百分比形式 (%)")
print("  - Cap = 50 MW (总装机容量)")
print("  - 测试集: 365天数据")
print("  - 训练集: 常规场景数据")
print("="*70)

# Display statistics
print("\n模型性能对比:")
print(stats_df.to_string(index=False))
print("="*70)
