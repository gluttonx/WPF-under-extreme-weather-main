#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成多场站测试结果CSV
支持3场站×6模型的完整评估
"""
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import scipy.io as scio
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
print("生成多场站测试结果CSV")
print("="*70)

# 参数
Cap = 50  # MW
dem_realc = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0 = torch.device("cpu")

# 检查是否已有保存的测试结果
if os.path.exists('all_stations_test_results.mat'):
    print("\n从已保存的结果加载...")
    results_mat = scio.loadmat('all_stations_test_results.mat')
    # 这里简化处理，直接重新加载模型生成
    print("（跳过，直接重新生成）")

# 场站列表
station_ids = ['58', '59', '60']

# 模型名称
model_names = [
    'Extreme_Weather_Class1',
    'Extreme_Weather_Class2',
    'Extreme_Weather_Class3',
    'Extreme_Weather_Class4',
    'Meta_Learning',
    'Pre_Training'
]

# 创建模型
model_test = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8], mode='test_task_support')
model_test = model_test.to(device)

# 收集所有场站的结果
all_results = []

print("\n生成预测...")
for station_id in station_ids:
    print(f"\n场站 {station_id}:")
    
    # 加载该场站的测试数据
    dataFile = f'{station_id}wf_4_train'
    wf_1 = scio.loadmat(dataFile)
    p = wf_1['p_1h']
    nwp = wf_1['nwp_1h']
    
    P_load1 = p[:,0]
    P_load = P_load1.reshape(np.size(P_load1,axis=0),-1)
    P_nwp1 = nwp
    nwp_index = [0,1,2,3,4]
    
    for i in range(5):
        if i==0:
            P_nwp = P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)
        else:
            P_nwp = np.concatenate((P_nwp,P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)),axis=1)
    
    dem_realp = 1
    len_realp = 12
    m = 365
    d = 24
    ooo = 365
    
    Series_day = P_load.reshape(-1,dem_realp)
    nwp_day = (P_nwp/np.max(abs(P_nwp),axis=0)).reshape(-1,dem_realp*np.size(P_nwp,axis=1))
    
    # 测试集（2023年）
    test_target_p_ = Series_day[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
    test_target_p = test_target_p_.reshape(-1,len_realp,dem_realp)
    test_input_c_ = nwp_day[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
    test_input_c = test_input_c_.reshape(-1,len_realp,dem_realc)
    
    Test_target_p = torch.tensor(test_target_p, dtype=torch.float32)
    Test_input_c = torch.tensor(test_input_c, dtype=torch.float32)
    
    # 真实值（标幺值）
    true_pu = test_target_p_.flatten()
    
    # 生成该场站各模型的预测
    station_predictions = {}
    
    # 1-4: 极端天气模型
    for i_class in range(4):
        model_file = f'model_fore_station{station_id}_extreme{i_class}.pth'
        if os.path.exists(model_file):
            model_test.load_state_dict(torch.load(model_file))
            with torch.no_grad():
                Test_input_device = Test_input_c.to(device)
                output = model_test(Test_input_device)
                output_np = output.to(device0).numpy().reshape(-1,dem_realp).flatten()
                station_predictions[model_names[i_class]] = output_np
            print(f"  ✓ {model_names[i_class]}")
        else:
            print(f"  ✗ {model_file} 不存在")
            station_predictions[model_names[i_class]] = np.zeros_like(true_pu)
    
    # 5: 元学习模型
    if os.path.exists('model_fore_train_task_query.pth'):
        model_test.load_state_dict(torch.load('model_fore_train_task_query.pth'))
        with torch.no_grad():
            Test_input_device = Test_input_c.to(device)
            output = model_test(Test_input_device)
            output_np = output.to(device0).numpy().reshape(-1,dem_realp).flatten()
            station_predictions[model_names[4]] = output_np
        print(f"  ✓ {model_names[4]}")
    
    # 6: 预训练模型
    pre_model_file = 'model_fore_pre_federated.pth' if os.path.exists('model_fore_pre_federated.pth') else 'model_fore_pre.pth'
    model_test.load_state_dict(torch.load(pre_model_file))
    with torch.no_grad():
        Test_input_device = Test_input_c.to(device)
        output = model_test(Test_input_device)
        output_np = output.to(device0).numpy().reshape(-1,dem_realp).flatten()
        station_predictions[model_names[5]] = output_np
    print(f"  ✓ {model_names[5]}")
    
    # 计算该场站各模型的指标
    mape_threshold = 0.05
    
    for model_name in model_names:
        pred_pu = station_predictions[model_name]
        
        # 标幺值指标转换为百分比
        mae_percent = np.mean(np.abs(true_pu - pred_pu)) * 100
        rmse_percent = np.sqrt(np.mean((true_pu - pred_pu)**2)) * 100
        
        # MAPE（排除低功率）
        mask = true_pu >= mape_threshold
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((true_pu[mask] - pred_pu[mask]) / true_pu[mask])) * 100
        else:
            mape = np.nan
        
        # R2
        r2 = 1 - (np.sum((true_pu - pred_pu)**2) / np.sum((true_pu - np.mean(true_pu))**2))
        
        all_results.append({
            'Station': station_id,
            'Model': model_name,
            'MAE_%': round(mae_percent, 4),
            'RMSE_%': round(rmse_percent, 4),
            'MAPE_%': round(mape, 2) if not np.isnan(mape) else 'N/A',
            'R2_Score': round(r2, 4)
        })

# 计算Overall Average
print("\n计算Overall Average（3场站平均）...")
for model_name in model_names:
    # 提取该模型在3个场站的指标
    model_results = [r for r in all_results if r['Model'] == model_name]
    
    mae_avg = np.mean([r['MAE_%'] for r in model_results])
    rmse_avg = np.mean([r['RMSE_%'] for r in model_results])
    
    mape_values = [r['MAPE_%'] for r in model_results if r['MAPE_%'] != 'N/A']
    mape_avg = np.mean(mape_values) if len(mape_values) > 0 else 'N/A'
    
    r2_avg = np.mean([r['R2_Score'] for r in model_results])
    
    all_results.append({
        'Station': 'Overall_Average',
        'Model': model_name,
        'MAE_%': round(mae_avg, 4),
        'RMSE_%': round(rmse_avg, 4),
        'MAPE_%': round(mape_avg, 2) if mape_avg != 'N/A' else 'N/A',
        'R2_Score': round(r2_avg, 4)
    })

# 保存为CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv('multi_station_performance.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*70)
print("✓✓✓ 多场站结果已生成！")
print("="*70)
print(f"\n生成文件: multi_station_performance.csv")
print(f"总行数: {len(results_df)}")
print(f"  - 每场站6个模型 × 3场站 = 18行")
print(f"  - Overall Average: 6行")
print(f"  - 总计: 24行")

print("\n" + "="*70)
print("性能对比表格:")
print("="*70)
print(results_df.to_string(index=False))

# 只显示Overall Average
print("\n" + "="*70)
print("Overall Average（3场站平均）:")
print("="*70)
overall_df = results_df[results_df['Station'] == 'Overall_Average']
print(overall_df.to_string(index=False))
print("="*70)
