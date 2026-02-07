#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用已训练的模型生成预测结果
"""
import os
import numpy as np
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
dataFile = 'wf_4_train'
wf_1  = scio.loadmat(dataFile)
p=wf_1['p_1h']
p_conven_00=wf_1['p_conven']
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
Cap=400.5
m=365
d=24
ooo=365
Series_day = P_load.reshape(-1,dem_realp)/Cap
nwp_day = (P_nwp/np.max(abs(P_nwp),axis=0)).reshape(-1,dem_realp*np.size(P_nwp,axis=1))
dem_realc=np.size(P_nwp,axis=1)

# Prepare test data
test_target_p_=Series_day[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
test_target_p=test_target_p_.reshape(-1,len_realp,dem_realp)
test_input_c_=nwp_day[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
test_input_c=test_input_c_.reshape(-1,len_realp,dem_realc)

Test_target_p=torch.tensor(test_target_p,dtype=torch.float32)
Test_input_c=torch.tensor(test_input_c,dtype=torch.float32)

# Prepare training data (for comparison)
nwp_conven_00=wf_1['nwp_conven_']
p_conven_=p_conven_00/Cap
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

# Load extreme weather support data for later use
p_extre_class1_00=wf_1['p_extre_class1']
p_extre_class2_00=wf_1['p_extre_class2']
p_extre_class3_00=wf_1['p_extre_class3']
p_extre_class4_00=wf_1['p_extre_class4']
p_extre_class__=np.empty([1,4],dtype=object)
p_extre_class__[0,0]=p_extre_class1_00
p_extre_class__[0,1]=p_extre_class2_00
p_extre_class__[0,2]=p_extre_class3_00
p_extre_class__[0,3]=p_extre_class4_00
p_extre_class_=p_extre_class__/Cap

test_outputs_query_00= np.empty([1, 6], dtype=object)
test_outputs_support_00= np.empty([1, 4], dtype=object)

# Generate predictions from different models
for i_model in range(6):
    with torch.no_grad():
        if i_model<4:
            print(f"  加载模型: model_fore_test_task_support_{i_model}.pth")
            model_fore_test_task_query.load_state_dict(torch.load(f"model_fore_test_task_support_{i_model}.pth"))
        elif i_model==4:
            print(f"  加载模型: model_fore_train_task_query.pth")
            model_fore_test_task_query.load_state_dict(torch.load("model_fore_train_task_query.pth"))
        elif i_model==5:
            print(f"  加载模型: model_fore_pre.pth")
            model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre.pth"))
        
        Test_input_c_device = Test_input_c.to(device)
        Test_output_query=model_fore_test_task_query(Test_input_c_device)
        test_outputs_query=Test_output_query.to(device0)
        test_outputs_query_=np.array(test_outputs_query.reshape(-1,dem_realp))
        test_outputs_query_00[0,i_model]=test_outputs_query_
        
        if i_model < 4:
            # For extreme weather models, also save their training data predictions
            p_data = p_extre_class_[0, i_model]
            num_samples = p_data.shape[0] // len_realp
            p_extre_reshaped = p_data[:num_samples*len_realp].reshape(num_samples, len_realp, 1)
            test_outputs_support_00[0,i_model] = np.array(p_extre_reshaped.reshape(-1, dem_realp))

# Get training outputs for comparison
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

# Save results
print("\n保存预测结果...")
results = {
    'test_outputs_query': test_outputs_query_00,
    'test_outputs_support': test_outputs_support_00,
    'train_outputs_pre': train_outputs_pre_,
    'train_target': train_target_p_,
    'test_target': test_target_p_,
    'Cap': Cap
}
scio.savemat('forecast_results.mat', results)

print("\n" + "="*70)
print("✓✓✓ 预测结果已保存！")
print("="*70)
print("\n生成的文件:")
print("  📁 forecast_results.mat - 包含以下数据:")
print("     - test_outputs_query[0,0-3]: 4个极端天气模型的测试集预测")
print("     - test_outputs_query[0,4]:   元训练模型的测试集预测")
print("     - test_outputs_query[0,5]:   预训练模型的测试集预测")
print("     - test_outputs_support:      极端天气样本预测")
print("     - train_outputs_pre:         训练集预测")
print("     - train_target:              训练集真实值")
print("     - test_target:               测试集真实值")
print("     - Cap:                       归一化系数 (400.5)")
print("="*70)
