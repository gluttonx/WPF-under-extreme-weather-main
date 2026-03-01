import os
import time
import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
import random
import model
from torch.nn.utils import weight_norm

# ========== [联邦新增] 联邦学习开关 ==========
USE_FEDERATION = True  # True=联邦多场站, False=单场站原方法
# 说明：设为False时完全退化为原始单场站元学习方法

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


## data processing
seed_torch(seed=1029)

# ========== [联邦修改] 多场站数据加载 ==========
if USE_FEDERATION:
    print("="*70)
    print("联邦模式：加载3个场站数据（58/59/60）")
    print("="*70)
    station_ids = ['58', '59', '60']  # [联邦新增] 3个场站作为客户端
else:
    print("="*70)
    print("单场站模式：加载场站58数据（原方法）")
    print("="*70)
    station_ids = ['58']  # [原代码] 单场站

# [联邦修改] 循环加载所有场站数据
station_data = {}
for station_id in station_ids:
    dataFile = f'{station_id}wf_4_train'
    print(f"  加载 {dataFile}.mat...")
    wf_1 = scio.loadmat(dataFile)
    
    # [原代码保留] 数据提取逻辑完全不变
    station_data[station_id] = {
        'p': wf_1['p_1h'],
        'p_conven_00': wf_1['p_conven'],
        'p_conven_class_00': wf_1['p_conven_class'],
        'p_extre_class1_00': wf_1['p_extre_class1'],
        'p_extre_class2_00': wf_1['p_extre_class2'],
        'p_extre_class3_00': wf_1['p_extre_class3'],
        'p_extre_class4_00': wf_1['p_extre_class4'],
        'nwp': wf_1['nwp_1h'],
        'nwp_conven_00': wf_1['nwp_conven_'],
        'nwp_conven_class_00': wf_1['nwp_conven_class_'],
        'nwp_extre_class1_00': wf_1['nwp_extre_class1_'],
        'nwp_extre_class2_00': wf_1['nwp_extre_class2_'],
        'nwp_extre_class3_00': wf_1['nwp_extre_class3_'],
        'nwp_extre_class4_00': wf_1['nwp_extre_class4_']
    }

# ========== [联邦修改] 删除"主场站"概念，3场站一视同仁 ==========
# 为第一个场站准备变量（用于参数初始化，后续会处理所有场站）
primary_station = station_ids[0]
p = station_data[primary_station]['p']
nwp = station_data[primary_station]['nwp']
p_conven_00 = station_data[primary_station]['p_conven_00']
nwp_conven_00 = station_data[primary_station]['nwp_conven_00']
p_conven_class_00 = station_data[primary_station]['p_conven_class_00']
nwp_conven_class_00 = station_data[primary_station]['nwp_conven_class_00']
P_load1=p[:,0]
P_load=P_load1.reshape(np.size(P_load1,axis=0),-1)
P_nwp1=nwp
nwp_index=[0,1,2,3,4]
for i in range(np.size(nwp_conven_class_00)):
    if i==0:
        P_nwp=P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)
    else:
        P_nwp=np.concatenate((P_nwp,P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)),axis=1)


# Define training equipment
device=torch.device("cuda")
device0=torch.device("cpu")


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
train_input_c = nwp_conven_1  # 用于参数初始化
train_target_p = p_conven_1

# ========== [联邦新增] 准备所有场站的数据（元训练、测试都用） ==========
print("\n" + "="*70)
print("准备所有场站的完整数据（元训练+极端天气+测试集）")
print("="*70)

# 准备所有客户端的常规天气数据（已在预训练中准备）
# ========== [联邦新增] 准备所有客户端的常规天气数据 ==========
if USE_FEDERATION:
    print("\n准备联邦客户端数据（常规天气）...")
    clients_train_data = {}
    
    for station_id in station_ids:
        print(f"  处理场站 {station_id}...")
        # [原代码逻辑] 每个场站使用相同的处理流程
        p_st = station_data[station_id]['p']
        nwp_st = station_data[station_id]['nwp']
        P_nwp1_st = nwp_st
        
        # 构建P_nwp_st（用于归一化）
        for i in range(5):
            if i==0:
                P_nwp_st = P_nwp1_st[:,nwp_index[i]].reshape(np.size(P_nwp1_st,axis=0),-1)
            else:
                P_nwp_st = np.concatenate((P_nwp_st,P_nwp1_st[:,nwp_index[i]].reshape(np.size(P_nwp1_st,axis=0),-1)),axis=1)
        
        # 处理常规天气数据
        p_conven_st = station_data[station_id]['p_conven_00']
        nwp_conven_st = station_data[station_id]['nwp_conven_00']
        
        p_conven_st_ = p_conven_st
        nwp_conven_st_ = np.empty([1,5],dtype=object)
        for i in range(np.size(nwp_conven_st,axis=1)):
            nwp_conven_st_[0,i] = nwp_conven_st[:,i].reshape(-1,1)/np.max(abs(P_nwp_st[:,i]),axis=0)
        
        for i_nwp in range(np.size(nwp_conven_st_, axis=1)):
            if i_nwp == 0:
                nwp_conven_1_st = nwp_conven_st_[0, i_nwp].transpose(1, 0)
                nwp_conven_1_st = nwp_conven_1_st[:, :, np.newaxis]
            else:
                nwp_conven_0_st = nwp_conven_st_[0, i_nwp].transpose(1, 0)
                nwp_conven_1_st = np.concatenate((nwp_conven_1_st, nwp_conven_0_st[:, :, np.newaxis]), axis=2)
        
        p_conven_1_st = p_conven_st_.transpose(1, 0)
        p_conven_1_st = p_conven_1_st[:, :, np.newaxis]
        
        # [联邦新增] 存储客户端数据
        clients_train_data[station_id] = {
            'input': nwp_conven_1_st,
            'target': p_conven_1_st
        }
        print(f"    shape: {nwp_conven_1_st.shape} → {p_conven_1_st.shape}")
    
    print(f"  总数据量: {sum([clients_train_data[s]['input'].shape[1] for s in station_ids])} 样本")
# ========== [联邦新增] 结束 ==========

# ========== [联邦修改] 准备所有场站的元训练、极端天气和测试数据 ==========
print("\n准备所有场站的元训练和测试数据（3场站一视同仁）...")
all_stations_full_data = {}

for station_id in station_ids:
    print(f"\n  场站 {station_id}:")
    
    # 获取该场站数据
    p_st = station_data[station_id]['p']
    nwp_st = station_data[station_id]['nwp']
    
    P_load1_st = p_st[:,0]
    P_load_st = P_load1_st.reshape(np.size(P_load1_st,axis=0),-1)
    P_nwp1_st = nwp_st
    
    for i in range(5):
        if i==0:
            P_nwp_st = P_nwp1_st[:,nwp_index[i]].reshape(np.size(P_nwp1_st,axis=0),-1)
        else:
            P_nwp_st = np.concatenate((P_nwp_st,P_nwp1_st[:,nwp_index[i]].reshape(np.size(P_nwp1_st,axis=0),-1)),axis=1)
    
    Series_day_st = P_load_st.reshape(-1,dem_realp)
    nwp_day_st = (P_nwp_st/np.max(abs(P_nwp_st),axis=0)).reshape(-1,dem_realp*np.size(P_nwp_st,axis=1))
    
    # 测试集（2023年）
    test_target_p_st = Series_day_st[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
    test_target_p_st = test_target_p_st.reshape(-1,len_realp,dem_realp)
    test_input_c_st = nwp_day_st[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
    test_input_c_st = test_input_c_st.reshape(-1,len_realp,dem_realc)
    
    # 聚类类别（用于元训练）
    p_conven_class_st = station_data[station_id]['p_conven_class_00']
    nwp_conven_class_st = station_data[station_id]['nwp_conven_class_00'].copy()
    for i in range(np.size(nwp_conven_class_st,axis=1)):
        nwp_conven_class_st[0,i] = nwp_conven_class_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)
    
    # 极端天气类别（用于Few-shot）
    p_extre_st = np.empty([1,4],dtype=object)
    nwp_extre_st = np.empty([1,5],dtype=object)
    
    for i_class in range(4):
        p_extre_st[0,i_class] = station_data[station_id][f'p_extre_class{i_class+1}_00']
    
    for i_nwp in range(5):
        nwp_extre_st[0,i_nwp] = np.empty([1,4],dtype=object)
        for i_class in range(4):
            nwp_extre_st[0, i_nwp][0, i_class] = station_data[station_id][f'nwp_extre_class{i_class+1}_00'][0, i_nwp]
    
    for i in range(np.size(nwp_extre_st,axis=1)):
        nwp_extre_st[0,i] = nwp_extre_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)
    
    # 存储该场站的完整数据
    all_stations_full_data[station_id] = {
        'P_nwp': P_nwp_st,
        'test_input': test_input_c_st,
        'test_target': test_target_p_st,
        'p_conven_class': p_conven_class_st,
        'nwp_conven_class': nwp_conven_class_st,
        'p_extre': p_extre_st,
        'nwp_extre': nwp_extre_st
    }
    print(f"    测试集2023: {test_target_p_st.shape}")
    print(f"    聚类类别: 10类")
    print(f"    极端天气: 4类")

print(f"\n✓ 所有场站数据准备完成！")

# [保留] 为了兼容部分原代码，保留变量
test_target_p = all_stations_full_data[station_ids[0]]['test_target']
test_input_c = all_stations_full_data[station_ids[0]]['test_input']
p_conven_class_00 = all_stations_full_data[station_ids[0]]['p_conven_class']
nwp_conven_class_00 = all_stations_full_data[station_ids[0]]['nwp_conven_class']
P_nwp = all_stations_full_data[station_ids[0]]['P_nwp']
p_conven_class_ = p_conven_class_00
nwp_conven_class_ = nwp_conven_class_00


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
        """根据mode返回需要训练的参数"""
        if self.mode == 'pre':
            # 预训练：训练所有参数
            return self.parameters()
        else:
            # 元学习：只训练TCN中的LWP层和最后的预测层
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
model_fore_pre = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='pre')
model_fore_train_task_support = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='train_task_support')
model_fore_train_task_query = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='train_task_query')
model_fore_test_task_support = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='test_task_support')
model_fore_test_task_query = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='test_task_support')


## Define loss
loss_fn_1=nn.MSELoss()
def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss1 = loss_fn_1(logits[0::2] * scale, y[0::2])
    loss2 = loss_fn_1(logits[1::2] * scale, y[1::2])
    # grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    # return torch.sum(grad ** 2)
    grad_batch1 = torch.autograd.grad(loss1, [scale], create_graph=True)[0]
    grad_batch2 = torch.autograd.grad(loss2, [scale], create_graph=True)[0]
    return torch.sum(grad_batch1 * grad_batch2)


## Define optimizer
optimizer_fore_pre = torch.optim.Adam(model_fore_pre.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_train_task_support = torch.optim.Adam(model_fore_train_task_support.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_train_task_query = torch.optim.Adam(model_fore_train_task_query.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_test_task_support = torch.optim.Adam(model_fore_test_task_support.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))

# [原代码保留] 主场站数据转为Tensor
Train_target_p=torch.tensor(train_target_p,dtype=torch.float32)
Train_input_c=torch.tensor(train_input_c,dtype=torch.float32)
Test_target_p=torch.tensor(test_target_p,dtype=torch.float32)
Test_input_c=torch.tensor(test_input_c,dtype=torch.float32)

# ========== [联邦新增] 所有客户端数据转为Tensor ==========
if USE_FEDERATION:
    clients_train_tensor = {}
    for station_id in station_ids:
        clients_train_tensor[station_id] = {
            'input': torch.tensor(clients_train_data[station_id]['input'], dtype=torch.float32),
            'target': torch.tensor(clients_train_data[station_id]['target'], dtype=torch.float32)
        }
    task_num = len(station_ids)  # [联邦新增] 客户端数量=3
else:
    task_num = 1  # [原代码] 单场站
# ========== [联邦新增] 结束 ==========

model_fore_pre = model_fore_pre.to(device)
model_fore_train_task_support = model_fore_train_task_support.to(device)
model_fore_train_task_query = model_fore_train_task_query.to(device)
model_fore_test_task_support = model_fore_test_task_support.to(device)
model_fore_test_task_query = model_fore_test_task_query.to(device)


## pre-train
if USE_FEDERATION:
    print("#########################################################################——————————联邦预训练（Federation Pre-train）——————————############################################################")
    print(f"客户端数量: {task_num}, 场站: {', '.join(station_ids)}")
else:
    print( "#########################################################################——————————预训练（Pre-train）——————————############################################################")

total_train_step=0
total_test_step=0
epoch1_pre = 80000
writer1=SummaryWriter("./logs_train/loss1")
writer2=SummaryWriter("./logs_train/loss2")
start_time=time.time()

for i in range(epoch1_pre):
    # [原代码保留] penalty系数调度
    if i<10000:
        k = 0
    elif 10000<=i<20000:
        k = 1
    elif 20000<=i<30000:
        k = 5
    elif 30000 <= i:
        k = 10
    
    model_fore_pre.train()
    
    if USE_FEDERATION:
        # ========== [联邦核心] 联邦梯度平均 ==========
        optimizer_fore_pre.zero_grad()
        total_loss1 = 0
        total_loss2 = 0
        
        # [联邦关键] 遍历所有客户端，累积梯度
        for station_id in station_ids:
            Train_target = clients_train_tensor[station_id]['target'].to(device)
            Train_input = clients_train_tensor[station_id]['input'].to(device)
            
            # [原代码保留] 相同的前向传播和loss计算
            Train_outputs_pre = model_fore_pre(Train_input)
            loss1 = penalty(Train_outputs_pre, Train_target)
            loss2 = loss_fn_1(Train_outputs_pre, Train_target)
            loss_en = k * loss1 + loss2
            
            # [联邦关键] 除以客户端数量实现联邦平均
            # 公式: θ = θ - α × (1/K) × Σ∇L_k
            loss_en_avg = loss_en / task_num  # ← 联邦平均的核心
            loss_en_avg.backward()  # 累积梯度，不清零
            
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
        
        # [联邦关键] 统一更新全局模型（包含所有客户端的平均梯度）
        optimizer_fore_pre.step()
        
        # 用于显示的平均loss
        loss1_display = total_loss1 / task_num
        loss2_display = total_loss2 / task_num
        
    else:
        # ========== [原代码保留] 单场站训练逻辑 ==========
        Train_target_p = Train_target_p.to(device)
        Train_input_c = Train_input_c.to(device)
        
        Train_outputs_pre=model_fore_pre(Train_input_c)
        loss1 = penalty(Train_outputs_pre,Train_target_p)
        loss2=loss_fn_1(Train_outputs_pre,Train_target_p)
        loss_en=k * loss1 + loss2
        
        optimizer_fore_pre.zero_grad()
        loss_en.backward()
        optimizer_fore_pre.step()
        
        loss1_display = loss1.item()
        loss2_display = loss2.item()
    
    # [原代码保留] 日志记录
    if (i + 1) % 100 == 0:
        end_time = time.time()
        print(end_time - start_time)
        print("[Epoch %d/%d] [loss_mse: %f] " % (i, epoch1_pre, loss2_display))
        writer1.add_scalar("loss_mse_pre", loss1_display, i)
        writer2.add_scalar("loss_mse_pre", loss2_display, i)

model_fore_pre.eval()

# [联邦修改] 根据模式保存不同的模型文件
if USE_FEDERATION:
    torch.save(model_fore_pre.state_dict(),"./model_fore_pre_federated.pth")
    print("\n✓ 联邦预训练完成: model_fore_pre_federated.pth")
else:
    torch.save(model_fore_pre.state_dict(),"./model_fore_pre.pth")
    print("\n✓ 预训练完成: model_fore_pre.pth")


## train_task_support
epoch_train_task=70000
for i_t in range(epoch_train_task):
    # ========== [联邦修改] 场站分组：每个场站独立贡献任务 ==========
    # 从每个场站随机选2个类别（3场站×2类=6个任务）
    selected_tasks = []
    
    for station_id in station_ids:
        # 每场站随机选2个类别
        station_classes = random.sample(range(0, 10), 2)
        
        # 处理该场站的选中类别
        nwp_conven_class_st = all_stations_full_data[station_id]['nwp_conven_class']
        p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']
        P_nwp_st = all_stations_full_data[station_id]['P_nwp']
        
        for i_class in station_classes:
            # 处理NWP数据
            for i_nwp in range(np.size(nwp_conven_class_st, axis=1)):
                nwp_data = nwp_conven_class_st[0,i_nwp][0,i_class]
                num_samples = nwp_data.shape[0] // len_realp
                nwp_reshaped = nwp_data[:num_samples*len_realp].reshape(num_samples, len_realp, 1)
                if i_nwp==0:
                    nwp_conven_class_1 = nwp_reshaped
                else:
                    nwp_conven_class_1 = np.concatenate((nwp_conven_class_1, nwp_reshaped), axis=2)
            
            # 处理功率数据
            p_data = p_conven_class_st[0, i_class]
            num_samples = p_data.shape[0] // len_realp
            p_conven_class_1 = p_data[:num_samples*len_realp].reshape(num_samples, len_realp, 1)
            
            selected_tasks.append({
                'station': station_id,
                'class': i_class,
                'nwp': nwp_conven_class_1,
                'p': p_conven_class_1
            })
    
    # 构建训练数据集
    train_input_dataset=list()
    train_target_dataset = list()
    for task in selected_tasks:
        train_input_dataset.append(task['nwp'])
        train_target_dataset.append(task['p'])
    
    # [联邦修改] 使用任务数量（6个任务）
    num_tasks = len(selected_tasks)
    train_input_support_=np.empty([10, 12, 5])
    train_input_query_ = np.empty([10, 12, 5])
    train_target_support_=np.empty([10, 12, 1])
    train_target_query_ = np.empty([10, 12, 1])
    
    for i_task in range(num_tasks):
        index_shot = random.sample(range(0, np.size(train_input_dataset[i_task],axis=0)), 20)
        train_input_support_=train_input_dataset[i_task][index_shot[0:10],:,:]
        train_input_query_ = train_input_dataset[i_task][index_shot[10:20], :, :]
        train_target_support_=train_target_dataset[i_task][index_shot[0:10],:,:]
        train_target_query_ = train_target_dataset[i_task][index_shot[10:20], :, :]
        if i_task==0:
            train_input_support=train_input_support_
            train_input_query = train_input_query_
            train_target_support=train_target_support_
            train_target_query = train_target_query_
        else:
            train_input_support=np.concatenate((train_input_support,train_input_support_),axis=0)
            train_input_query = np.concatenate((train_input_query, train_input_query_), axis=0)
            train_target_support=np.concatenate((train_target_support,train_target_support_),axis=0)
            train_target_query = np.concatenate((train_target_query, train_target_query_), axis=0)
    Train_target_support = torch.tensor(train_target_support, dtype=torch.float32)
    Train_input_support = torch.tensor(train_input_support, dtype=torch.float32)
    Train_target_query = torch.tensor(train_target_query, dtype=torch.float32)
    Train_input_query = torch.tensor(train_input_query, dtype=torch.float32)
    print(
        "[##################################################################——————————train_task_support_Epoch %d/%d——————————############################################################]"
        % (i_t, epoch_train_task)
    )
    # [联邦修改] 根据模式加载对应的预训练模型
    if i_t==0:
        if USE_FEDERATION:
            model_fore_train_task_support.load_state_dict(torch.load("model_fore_pre_federated.pth"))
        else:
            model_fore_train_task_support.load_state_dict(torch.load("model_fore_pre.pth"))
    else:
        model_fore_train_task_support.load_state_dict(torch.load("model_fore_train_task_query.pth"))
    total_train_step=0
    total_test_step=0
    epoch1_train_task_support = 1
    start_time=time.time()
    for i in range(epoch1_train_task_support):
        k=10
        Train_target_support = Train_target_support.to(device)
        Train_input_support = Train_input_support.to(device)
        model_fore_train_task_support.train()
        Train_outputs_support=model_fore_train_task_support(Train_input_support)
        loss1=penalty(Train_outputs_support,Train_target_support)
        loss2=loss_fn_1(Train_outputs_support,Train_target_support)
        loss_en=k*loss1+loss2
        optimizer_fore_train_task_support.zero_grad()
        loss_en.backward()
        optimizer_fore_train_task_support.step()
        #
        if (i + 1) % 2 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(
                "[Epoch %d/%d] [loss_mse: %f]"
                % (i, epoch1_train_task_support, loss2.item())
            )
            writer1.add_scalar("loss_mse_train_task_support", loss1.item(),
                              epoch1_pre + i_t * epoch1_train_task_support + i)
            writer2.add_scalar("loss_mse_train_task_support", loss2.item(), epoch1_pre+i_t*epoch1_train_task_support+i)
    model_fore_train_task_support.eval()
    torch.save(model_fore_train_task_support.state_dict(),"./model_fore_train_task_support.pth")


    ## train_task_query
    print(
        "[##################################################################——————————train_task_query_Epoch %d/%d——————————############################################################]"
        % (i_t, epoch_train_task)
    )
    # [联邦修改] 根据模式加载对应的预训练模型
    if i_t==0:
        if USE_FEDERATION:
            model_fore_train_task_query.load_state_dict(torch.load("model_fore_pre_federated.pth"))
        else:
            model_fore_train_task_query.load_state_dict(torch.load("model_fore_pre.pth"))
        model_fore_train_task_support.load_state_dict(torch.load("model_fore_train_task_support.pth"))
    else:
        model_fore_train_task_query.load_state_dict(torch.load("model_fore_train_task_query.pth"))
        model_fore_train_task_support.load_state_dict(torch.load("model_fore_train_task_support.pth"))
    total_train_step=0
    total_test_step=0
    epoch1_train_task_query = 1
    start_time=time.time()
    for i in range(epoch1_train_task_query):
        k=10
        Train_target_query = Train_target_query.to(device)
        Train_input_query = Train_input_query.to(device)
        model_fore_train_task_support.train()
        Train_outputs_query_=model_fore_train_task_support(Train_input_query)
        loss1 = penalty(Train_outputs_query_, Train_target_query)
        loss2=loss_fn_1(Train_outputs_query_,Train_target_query)
        loss_en=k*loss1+loss2
        optimizer_fore_train_task_query.zero_grad()
        loss_en.backward()
        optimizer_fore_train_task_query.step()
        Train_outputs_query=model_fore_train_task_query(Train_input_query)
        #
        if (i + 1) % 1 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(
                "[Epoch %d/%d] [loss_mse: %f] "
                % (i, epoch1_train_task_query, loss2.item())
            )

            writer1.add_scalar("loss_mse_train_task_query", loss1.item(), epoch1_pre+i_t*(epoch1_train_task_support+epoch1_train_task_query)+epoch1_train_task_support+i)
            writer2.add_scalar("loss_mse_train_task_query", loss2.item(), epoch1_pre + i_t * (
                        epoch1_train_task_support + epoch1_train_task_query) + epoch1_train_task_support + i)
    model_fore_train_task_query.eval()
    torch.save(model_fore_train_task_query.state_dict(),"./model_fore_train_task_query.pth")


## test_task_support
print("##################################################################——————————test_task_support（Few-shot适应：3场站×4类=12个模型）——————————############################################################")

# ========== [联邦修改] 为所有场站的所有极端天气类别训练个性化模型 ==========
all_personalized_models = {}  # 存储所有个性化模型

for station_id in station_ids:
    print(f"\n{'='*70}")
    print(f"场站 {station_id} 的Few-shot适应")
    print(f"{'='*70}")
    
    for i_class in range(4):
        print(f"\n  极端天气类别 {i_class+1}:")
        
        # 加载元训练模型作为初始化
        model_fore_test_task_support.load_state_dict(torch.load("model_fore_train_task_query.pth"))
        # [联邦修改] 获取该场站该类的极端天气数据
        nwp_extre_st = all_stations_full_data[station_id]['nwp_extre']
        p_extre_st = all_stations_full_data[station_id]['p_extre']
        P_nwp_st = all_stations_full_data[station_id]['P_nwp']
        
        # 处理NWP数据
        for i_nwp in range(np.size(nwp_extre_st, axis=1)):
            nwp_data = nwp_extre_st[0, i_nwp][0, i_class]
            num_samples = nwp_data.shape[0] // len_realp
            nwp_reshaped = nwp_data[:num_samples*len_realp].reshape(num_samples, len_realp, 1)
            if i_nwp == 0:
                nwp_extre_class_1 = nwp_reshaped
            else:
                nwp_extre_class_1 = np.concatenate((nwp_extre_class_1, nwp_reshaped), axis=2)
        
        # 处理功率数据
        p_data = p_extre_st[0, i_class]
        num_samples = p_data.shape[0] // len_realp
        p_extre_class_1 = p_data[:num_samples*len_realp].reshape(num_samples, len_realp, 1)
        
        nwp_extre_class=nwp_extre_class_1
        p_extre_class=p_extre_class_1
        Test_target_support = torch.tensor(p_extre_class, dtype=torch.float32)
        Test_input_support = torch.tensor(nwp_extre_class, dtype=torch.float32)
        
        print(f"    样本数: {num_samples}")
        total_train_step=0
        total_test_step=0
        # [修改] 增加few-shot训练轮数
        epoch1_test_task_support = 100  # 从10改为100，增加训练充分性
        start_time=time.time()
        
        for i in range(epoch1_test_task_support):
            # [修改] 添加penalty正则化
            k=5  # 从0改为5，添加正则化避免过拟合
            Test_target_support = Test_target_support.to(device)
            Test_input_support = Test_input_support.to(device)
            model_fore_test_task_support.train()
            Test_outputs_support=model_fore_test_task_support(Test_input_support)
            loss1 = penalty(Test_outputs_support, Test_target_support)
            loss2=loss_fn_1(Test_outputs_support,Test_target_support)
            loss_en=k*loss1+loss2
            optimizer_fore_test_task_support.zero_grad()
            loss_en.backward()
            optimizer_fore_test_task_support.step()
            
            if (i + 1) % 20 == 0:
                end_time = time.time()
                print(f"      [Epoch {i+1}/{epoch1_test_task_support}] [loss_mse: {loss2.item():.6f}]")
                writer2.add_scalar(f"loss_mse_station{station_id}_class{i_class}", loss2.item(), i)
        
        model_fore_test_task_support.eval()
        
        # [联邦修改] 保存该场站该类的个性化模型
        model_name = f"./model_fore_station{station_id}_extreme{i_class}.pth"
        torch.save(model_fore_test_task_support.state_dict(), model_name)
        print(f"    ✓ 保存: {model_name}")
        
        # 存储该模型（用于后续评估）
        all_personalized_models[f'{station_id}_class{i_class}'] = model_name

writer1.close()
writer2.close()
print(f"\n✓ Few-shot训练完成，生成了 {len(all_personalized_models)} 个个性化模型")
# ========== [联邦修改] 保存所有场站的测试结果 ==========
print("\n" + "="*70)
print("生成所有场站的测试预测结果")
print("="*70)

all_test_results = {}  # 存储所有场站的测试结果

for station_id in station_ids:
    print(f"\n场站 {station_id} 预测:")
    
    # 获取该场站的测试数据
    Test_input_c_st = torch.tensor(all_stations_full_data[station_id]['test_input'], dtype=torch.float32)
    Test_target_p_st = all_stations_full_data[station_id]['test_target']
    
    all_test_results[station_id] = {
        'test_input': Test_input_c_st,
        'test_target': Test_target_p_st,
        'predictions': {}
    }
    
    # 预测：该场站的4个极端天气模型
    for i_class in range(4):
        model_name = f"model_fore_station{station_id}_extreme{i_class}.pth"
        model_fore_test_task_query.load_state_dict(torch.load(model_name))
        
        with torch.no_grad():
            Test_input_device = Test_input_c_st.to(device)
            Test_output = model_fore_test_task_query(Test_input_device)
            test_output = Test_output.to(device0)
            test_output_np = np.array(test_output.reshape(-1,dem_realp))
            all_test_results[station_id]['predictions'][f'extreme_{i_class}'] = test_output_np
        
        print(f"  ✓ 极端类别{i_class+1}")
    
    # 预测：元学习模型
    model_fore_test_task_query.load_state_dict(torch.load("model_fore_train_task_query.pth"))
    with torch.no_grad():
        Test_input_device = Test_input_c_st.to(device)
        Test_output = model_fore_test_task_query(Test_input_device)
        test_output = Test_output.to(device0)
        test_output_np = np.array(test_output.reshape(-1,dem_realp))
        all_test_results[station_id]['predictions']['meta'] = test_output_np
    print(f"  ✓ 元学习模型")
    
    # 预测：预训练模型
    if USE_FEDERATION:
        model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre_federated.pth"))
    else:
        model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre.pth"))
    with torch.no_grad():
        Test_input_device = Test_input_c_st.to(device)
        Test_output = model_fore_test_task_query(Test_input_device)
        test_output = Test_output.to(device0)
        test_output_np = np.array(test_output.reshape(-1,dem_realp))
        all_test_results[station_id]['predictions']['pre'] = test_output_np
    print(f"  ✓ 预训练模型")

# 保存所有结果
print("\n保存所有场站测试结果...")
scio.savemat('all_stations_test_results.mat', {'all_test_results': all_test_results, 'Cap': Cap})
print("✓ 已保存: all_stations_test_results.mat")

print("\n" + "="*70)
print("✓✓✓ 训练和测试全部完成！")
print(f"生成的模型: {len(all_personalized_models)}个个性化模型 + 1个元学习模型 + 1个预训练模型")
print("="*70)

# [删除] 原来的单场站保存代码
# 以下代码不再需要
if False:  # 禁用原代码
    test_outputs_query_00= np.empty([1, 6], dtype=object)
    test_outputs_support_00= np.empty([1, 4], dtype=object)
    
    ## save
    for i_model in range(6):
        with torch.no_grad():
            if i_model<4:
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_test_task_support_%d.pth"%(i_model)))
            elif i_model==4:
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_train_task_query.pth" ))
            elif i_model==5:
                model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre.pth" ))
            Test_input_c = Test_input_c.to(device)
            Test_output_query=model_fore_test_task_query(Test_input_c)
            test_outputs_query=Test_output_query.to(device0)
            test_outputs_query_=np.array(test_outputs_query.reshape(-1,dem_realp))
            test_outputs_query_00[0,i_model]=test_outputs_query_
            if i_model < 4:
                test_outputs_support_list=Test_outputs_support_list[i_model].to(device0)
                test_outputs_support_ = np.array(test_outputs_support_list.reshape(-1, dem_realp))
                test_outputs_support_00[0,i_model]=test_outputs_support_
            train_outputs_pre=Train_outputs_pre.to(device0)
            train_outputs_support=Train_outputs_support.to(device0)
            train_outputs_query=Train_outputs_query.to(device0)
            pass  # [联邦修改] 原单场站保存逻辑已被新的多场站逻辑替代

