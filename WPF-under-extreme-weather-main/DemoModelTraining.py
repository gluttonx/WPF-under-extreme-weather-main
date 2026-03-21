import os
import time
import numpy as np
import torch.nn as nn
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
import random
import model
from torch.nn.utils import weight_norm


def env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


def env_float(name, default):
    value = os.getenv(name)
    return default if value is None else float(value)


# ========== [联邦新增] 联邦学习开关 ==========
USE_FEDERATION = env_flag("USE_FEDERATION", True)  # True=联邦多场站, False=单场站原方法
# 说明：设为False时完全退化为原始单场站元学习方法

# ========== 论文口径关键开关 ==========
TRAIN_META_ONLY_BASELINE = env_flag("TRAIN_META_ONLY_BASELINE", True)  # 新增：训练真正的 meta-learning only 基线
FEW_SHOT_EPOCHS = env_int("FEW_SHOT_EPOCHS", 50)             # 论文口径：每个极端天气 fine-tune 50 epochs
FEW_SHOT_USE_CDRM = env_flag("FEW_SHOT_USE_CDRM", False)
FEW_SHOT_CDRM_WEIGHT = 5.0
# 联邦场景下保持3场站，但按论文口径保持每轮总任务数 k*=5
META_TASKS_PER_EPOCH = env_int("META_TASKS_PER_EPOCH", 5)
PRETRAIN_EPOCHS = env_int("PRETRAIN_EPOCHS", 35000)
PROPOSED_META_EPOCHS = env_int("PROPOSED_META_EPOCHS", 30000)
META_ONLY_META_EPOCHS = env_int("META_ONLY_META_EPOCHS", 30000)
# 论文消融口径：Meta-only = 去掉 pre-training，其余训练机制保持一致
META_ONLY_USE_CDRM = env_flag("META_ONLY_USE_CDRM", True)
META_ONLY_TRAIN_ALL_PARAMS = env_flag("META_ONLY_TRAIN_ALL_PARAMS", False)
META_ONLY_DISABLE_LWP = env_flag("META_ONLY_DISABLE_LWP", False)
FED_PRETRAIN_REGIME_ALPHA = env_float("FED_PRETRAIN_REGIME_ALPHA", 1.0)
FED_PRETRAIN_AGGREGATION_GAMMA = env_float("FED_PRETRAIN_AGGREGATION_GAMMA", 0.5)
PROPOSED_META_SHARED_ANCHOR_BETA = env_float("PROPOSED_META_SHARED_ANCHOR_BETA", 0.01)
PROPOSED_META_SHARED_LR_SCALE = env_float("PROPOSED_META_SHARED_LR_SCALE", 0.3)

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
    if torch.cuda.is_available():
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
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0=torch.device("cpu")

# 模型文件路径（避免混淆）
PRETRAIN_MODEL_PATH = "model_fore_pre_federated.pth" if USE_FEDERATION else "model_fore_pre.pth"
PROPOSED_SUPPORT_MODEL_PATH = "model_fore_train_task_support_proposed.pth"
PROPOSED_META_MODEL_PATH = "model_fore_train_task_query_proposed.pth"
META_ONLY_SUPPORT_MODEL_PATH = "model_fore_train_task_support_meta_only.pth"
META_ONLY_MODEL_PATH = "model_fore_train_task_query_meta_only.pth"


def get_local_pretrain_model_path(station_id):
    return f"model_fore_pre_station{station_id}_local.pth"


def get_proposed_support_model_path(station_id):
    return f"model_fore_train_task_support_proposed_station{station_id}.pth"


def get_proposed_meta_model_path(station_id):
    return f"model_fore_train_task_query_proposed_station{station_id}.pth"


def get_local_meta_support_model_path(station_id):
    return f"model_fore_train_task_support_local_meta_station{station_id}.pth"


def get_local_meta_model_path(station_id):
    return f"model_fore_train_task_query_local_meta_station{station_id}.pth"


def get_meta_only_support_model_path(station_id):
    return f"model_fore_train_task_support_meta_only_station{station_id}.pth"


def get_meta_only_model_path(station_id):
    return f"model_fore_train_task_query_meta_only_station{station_id}.pth"


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
# 保存一份随机初始化权重，供 meta-learning only 使用（不经过 pre-train）
meta_only_random_init_state = copy.deepcopy(model_fore_train_task_query.state_dict())


## Define loss
loss_fn_1=nn.MSELoss()
def penalty(logits, y):
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
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


def clone_state_dict(state_dict):
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


def average_state_dicts(weighted_states):
    total_weight = float(sum(weight for _, weight in weighted_states))
    averaged_state = {}
    reference_state = weighted_states[0][0]
    for name in reference_state.keys():
        accumulator = None
        for state_dict, weight in weighted_states:
            weighted_tensor = state_dict[name].detach() * float(weight)
            accumulator = weighted_tensor if accumulator is None else accumulator + weighted_tensor
        averaged_state[name] = (accumulator / total_weight).clone()
    return averaged_state


def weighted_mse_loss(predictions, targets, sample_weights):
    per_sample_mse = torch.mean((predictions - targets) ** 2, dim=(1, 2))
    return torch.sum(per_sample_mse * sample_weights) / torch.sum(sample_weights).clamp_min(1e-6)


def compute_regime_sample_weights(train_input, train_target, alpha=1.0):
    target_flat = train_target.squeeze(-1)
    ramp_score = torch.mean(torch.abs(target_flat[:, 1:] - target_flat[:, :-1]), dim=1) if target_flat.shape[1] > 1 else torch.zeros(
        target_flat.shape[0], device=train_target.device
    )
    volatility_score = torch.std(target_flat, dim=1, unbiased=False)

    input_flat = train_input.reshape(train_input.shape[0], -1)
    input_center = input_flat.mean(dim=0, keepdim=True)
    input_scale = input_flat.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    rarity_score = torch.mean(torch.abs((input_flat - input_center) / input_scale), dim=1)

    raw_score = ramp_score + volatility_score + rarity_score
    normalized_score = raw_score - raw_score.min()
    normalized_score = normalized_score / normalized_score.mean().clamp_min(1e-6)
    sample_weights = 1.0 + alpha * normalized_score

    topk = max(1, int(normalized_score.shape[0] * 0.2))
    regime_factor = torch.topk(normalized_score, k=topk).values.mean().item()
    return sample_weights.detach(), float(regime_factor)


def get_pretrain_penalty_weight(epoch_idx):
    if epoch_idx < 10000:
        return 0
    if epoch_idx < 20000:
        return 1
    if epoch_idx < 30000:
        return 5
    return 10


def client_local_pretrain_update(global_state_dict, station_id, penalty_weight):
    client_model = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8], mode='pre').to(device)
    client_model.load_state_dict(copy.deepcopy(global_state_dict))
    client_optimizer = torch.optim.Adam(client_model.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))

    train_target = clients_train_tensor[station_id]['target'].to(device)
    train_input = clients_train_tensor[station_id]['input'].to(device)
    sample_weights, regime_factor = compute_regime_sample_weights(
        train_input,
        train_target,
        alpha=FED_PRETRAIN_REGIME_ALPHA
    )

    client_model.train()
    train_outputs = client_model(train_input)
    loss_penalty = penalty(train_outputs, train_target)
    loss_mse = weighted_mse_loss(train_outputs, train_target, sample_weights)
    loss_total = penalty_weight * loss_penalty + loss_mse

    client_optimizer.zero_grad()
    loss_total.backward()
    client_optimizer.step()

    updated_state = clone_state_dict(client_model.state_dict())
    aggregation_weight = int(train_input.shape[0]) * max(
        0.5,
        min(2.0, 1.0 + FED_PRETRAIN_AGGREGATION_GAMMA * (regime_factor - 1.0))
    )
    return {
        'state_dict': updated_state,
        'num_samples': int(train_input.shape[0]),
        'regime_factor': regime_factor,
        'aggregation_weight': float(aggregation_weight),
        'loss_penalty': float(loss_penalty.item()),
        'loss_mse': float(loss_mse.item()),
    }


def run_local_pretrain(station_id, save_path, epoch1_pre=35000):
    print("\n" + "=" * 70)
    print(f"开始场站 {station_id} 的本地 conventional pretrain")
    print(f"  epochs={epoch1_pre}")
    print("=" * 70)

    local_model = model_fore(
        input_channel_fore=dem_realc,
        output_channel_fore=[128, 96, 64, 48, 32, 16, 8],
        mode='pre'
    ).to(device)
    local_optimizer = torch.optim.Adam(local_model.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))

    train_target = clients_train_tensor[station_id]['target'].to(device)
    train_input = clients_train_tensor[station_id]['input'].to(device)

    for i in range(epoch1_pre):
        penalty_weight = get_pretrain_penalty_weight(i)
        local_model.train()
        train_outputs = local_model(train_input)
        loss_penalty = penalty(train_outputs, train_target)
        loss_mse = loss_fn_1(train_outputs, train_target)
        loss_total = penalty_weight * loss_penalty + loss_mse

        local_optimizer.zero_grad()
        loss_total.backward()
        local_optimizer.step()

        if (i + 1) % 100 == 0:
            writer1.add_scalar(f"loss_penalty_pre_local_station{station_id}", loss_penalty.item(), i)
            writer2.add_scalar(f"loss_mse_pre_local_station{station_id}", loss_mse.item(), i)

    local_model.eval()
    local_state = clone_state_dict(local_model.state_dict())
    torch.save(local_state, save_path)
    print(f"✓ 场站 {station_id} 本地 conventional pretrain 完成: {save_path}")
    return local_state


def server_aggregate_client_states(client_updates):
    weighted_states = [
        (client_update['state_dict'], client_update['aggregation_weight'])
        for client_update in client_updates
    ]
    return average_state_dicts(weighted_states)


def sample_station_meta_batch(station_id):
    station_tasks = []
    nwp_conven_class_st = all_stations_full_data[station_id]['nwp_conven_class']
    p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']
    total_station_classes = np.size(p_conven_class_st, axis=1)

    for i_class in range(total_station_classes):
        for i_nwp in range(np.size(nwp_conven_class_st, axis=1)):
            nwp_data = nwp_conven_class_st[0, i_nwp][0, i_class]
            num_samples = nwp_data.shape[0] // len_realp
            nwp_reshaped = nwp_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)
            if i_nwp == 0:
                nwp_conven_class_1 = nwp_reshaped
            else:
                nwp_conven_class_1 = np.concatenate((nwp_conven_class_1, nwp_reshaped), axis=2)

        p_data = p_conven_class_st[0, i_class]
        num_samples = p_data.shape[0] // len_realp
        p_conven_class_1 = p_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)
        station_tasks.append({
            'nwp': nwp_conven_class_1,
            'p': p_conven_class_1,
        })

    tasks_per_epoch = min(META_TASKS_PER_EPOCH, len(station_tasks))
    selected_tasks = random.sample(station_tasks, tasks_per_epoch)

    for i_task, task in enumerate(selected_tasks):
        index_shot = random.sample(range(0, np.size(task['nwp'], axis=0)), 20)
        train_input_support_ = task['nwp'][index_shot[0:10], :, :]
        train_input_query_ = task['nwp'][index_shot[10:20], :, :]
        train_target_support_ = task['p'][index_shot[0:10], :, :]
        train_target_query_ = task['p'][index_shot[10:20], :, :]
        if i_task == 0:
            train_input_support = train_input_support_
            train_input_query = train_input_query_
            train_target_support = train_target_support_
            train_target_query = train_target_query_
        else:
            train_input_support = np.concatenate((train_input_support, train_input_support_), axis=0)
            train_input_query = np.concatenate((train_input_query, train_input_query_), axis=0)
            train_target_support = np.concatenate((train_target_support, train_target_support_), axis=0)
            train_target_query = np.concatenate((train_target_query, train_target_query_), axis=0)

    return (
        torch.tensor(train_target_support, dtype=torch.float32),
        torch.tensor(train_input_support, dtype=torch.float32),
        torch.tensor(train_target_query, dtype=torch.float32),
        torch.tensor(train_input_query, dtype=torch.float32)
    )


## pre-train
if USE_FEDERATION:
    print("#########################################################################——————————联邦预训练（Federation Pre-train）——————————############################################################")
    print(f"客户端数量: {task_num}, 场站: {', '.join(station_ids)}")
else:
    print( "#########################################################################——————————预训练（Pre-train）——————————############################################################")

total_train_step=0
total_test_step=0
epoch1_pre = PRETRAIN_EPOCHS
writer1=SummaryWriter("./logs_train/loss1")
writer2=SummaryWriter("./logs_train/loss2")
start_time=time.time()

if USE_FEDERATION:
    global_pretrain_state = clone_state_dict(model_fore_pre.state_dict())
    for i in range(epoch1_pre):
        k = get_pretrain_penalty_weight(i)

        client_updates = []
        total_loss1 = 0.0
        total_loss2 = 0.0
        for station_id in station_ids:
            client_update = client_local_pretrain_update(global_pretrain_state, station_id, penalty_weight=k)
            client_updates.append(client_update)
            total_loss1 += client_update['loss_penalty']
            total_loss2 += client_update['loss_mse']

        aggregated_state = server_aggregate_client_states(client_updates)
        global_pretrain_state = clone_state_dict(aggregated_state)
        model_fore_pre.load_state_dict(copy.deepcopy(global_pretrain_state))

        loss1_display = total_loss1 / task_num
        loss2_display = total_loss2 / task_num

        if (i + 1) % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("[Epoch %d/%d] [loss_mse: %f] " % (i, epoch1_pre, loss2_display))
            writer1.add_scalar("loss_mse_pre", loss1_display, i)
            writer2.add_scalar("loss_mse_pre", loss2_display, i)
else:
    for i in range(epoch1_pre):
        k = get_pretrain_penalty_weight(i)

        model_fore_pre.train()
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

        if (i + 1) % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("[Epoch %d/%d] [loss_mse: %f] " % (i, epoch1_pre, loss2_display))
            writer1.add_scalar("loss_mse_pre", loss1_display, i)
            writer2.add_scalar("loss_mse_pre", loss2_display, i)

model_fore_pre.eval()
torch.save(model_fore_pre.state_dict(), PRETRAIN_MODEL_PATH)
if USE_FEDERATION:
    print(f"\n✓ 联邦预训练完成: {PRETRAIN_MODEL_PATH}")
else:
    print(f"\n✓ 预训练完成: {PRETRAIN_MODEL_PATH}")


local_pretrain_state_dicts = {}
for station_id in station_ids:
    local_pretrain_state_dicts[station_id] = run_local_pretrain(
        station_id=station_id,
        save_path=get_local_pretrain_model_path(station_id),
        epoch1_pre=PRETRAIN_EPOCHS
    )


def freeze_lwp_as_identity(model_instance):
    """
    将 LWP 固定为恒等变换(scale=1, shift=0)，用于传统 meta-learning 基线。
    """
    for module in model_instance.modules():
        if isinstance(module, model.LWP):
            with torch.no_grad():
                module.scale.fill_(1.0)
                module.shift.zero_()
            module.scale.requires_grad = False
            module.shift.requires_grad = False


def get_meta_trainable_params(model_instance, train_all_params=False, disable_lwp=False):
    """
    元训练参数选择：
    - train_all_params=True: 训练全部参数（传统 meta-learning）
    - disable_lwp=True: 从可训练参数中移除 LWP 参数
    """
    if train_all_params:
        if disable_lwp:
            return [p for name, p in model_instance.named_parameters() if "lwp" not in name]
        return list(model_instance.parameters())

    if disable_lwp:
        return list(model_instance.fore_baselearner.parameters())

    return list(model_instance.get_trainable_params())


def build_meta_optimizer(model_instance, train_all_params=False, disable_lwp=False, shared_lr_scale=1.0):
    local_params = []
    shared_params = []

    for name, parameter in model_instance.named_parameters():
        if not parameter.requires_grad:
            continue
        if train_all_params:
            if disable_lwp and "lwp" in name:
                continue
            if "fore_baselearner" in name:
                shared_params.append(parameter)
            else:
                local_params.append(parameter)
            continue

        if "lwp" in name and not disable_lwp:
            local_params.append(parameter)
        elif "fore_baselearner" in name:
            shared_params.append(parameter)

    parameter_groups = []
    if local_params:
        parameter_groups.append({'params': local_params, 'lr': 0.0002})
    if shared_params:
        parameter_groups.append({'params': shared_params, 'lr': 0.0002 * shared_lr_scale})
    return torch.optim.Adam(parameter_groups, betas=(0.5, 0.999))


def compute_shared_anchor_loss(model_instance, anchor_state_dict):
    anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
    shared_param_count = 0
    for name, parameter in model_instance.named_parameters():
        if "fore_baselearner" not in name:
            continue
        anchor_tensor = anchor_state_dict[name].to(parameter.device)
        anchor_loss = anchor_loss + torch.mean((parameter - anchor_tensor) ** 2)
        shared_param_count += 1
    if shared_param_count == 0:
        return anchor_loss
    return anchor_loss / shared_param_count


def run_local_meta_training(
    station_id,
    meta_tag,
    init_state_dict,
    support_model_path,
    query_model_path,
    epoch_train_task=70000,
    use_cdrm=True,
    train_all_params=False,
    disable_lwp=False,
    shared_anchor_beta=0.0,
    shared_lr_scale=1.0
):
    """
    单场站本地元训练过程：
    - proposed: init_state_dict 为 federated pre-train 权重
    - meta_only: init_state_dict 为随机初始化
    """
    print("\n" + "=" * 70)
    print(f"开始场站 {station_id} 的本地元训练: {meta_tag}")
    print(
        f"  use_cdrm={use_cdrm}, train_all_params={train_all_params}, "
        f"disable_lwp={disable_lwp}, shared_anchor_beta={shared_anchor_beta}, "
        f"shared_lr_scale={shared_lr_scale}"
    )
    total_task_pool = np.size(all_stations_full_data[station_id]['p_conven_class'], axis=1)
    print(f"  tasks_per_epoch={min(META_TASKS_PER_EPOCH, total_task_pool)}, station_task_pool={total_task_pool}")
    print("=" * 70)

    get_meta_trainable_params(
        model_fore_train_task_support,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp
    )
    get_meta_trainable_params(
        model_fore_train_task_query,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp
    )
    optimizer_support = build_meta_optimizer(
        model_fore_train_task_support,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp,
        shared_lr_scale=shared_lr_scale
    )
    optimizer_query = build_meta_optimizer(
        model_fore_train_task_query,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp,
        shared_lr_scale=shared_lr_scale
    )
    prior_anchor_state = clone_state_dict(init_state_dict)

    for i_t in range(epoch_train_task):
        Train_target_support, Train_input_support, Train_target_query, Train_input_query = sample_station_meta_batch(station_id)

        print(
            "[##################################################################"
            f"——{meta_tag}:train_task_support_Epoch {i_t}/{epoch_train_task}——"
            "############################################################]"
        )

        if i_t == 0:
            base_state = copy.deepcopy(init_state_dict)
        else:
            base_state = torch.load(query_model_path)
        model_fore_train_task_support.load_state_dict(copy.deepcopy(base_state))

        if disable_lwp:
            freeze_lwp_as_identity(model_fore_train_task_support)

        model_fore_train_task_support.train()
        Train_target_support = Train_target_support.to(device)
        Train_input_support = Train_input_support.to(device)
        Train_outputs_support = model_fore_train_task_support(Train_input_support)
        if use_cdrm:
            loss1 = penalty(Train_outputs_support, Train_target_support)
        else:
            loss1 = torch.zeros((), dtype=torch.float32, device=device)
        loss2 = loss_fn_1(Train_outputs_support, Train_target_support)
        loss_en = 10 * loss1 + loss2 if use_cdrm else loss2
        anchor_loss_support = torch.zeros((), dtype=torch.float32, device=device)
        if shared_anchor_beta > 0:
            anchor_loss_support = compute_shared_anchor_loss(model_fore_train_task_support, prior_anchor_state)
            loss_en = loss_en + shared_anchor_beta * anchor_loss_support
        optimizer_support.zero_grad()
        loss_en.backward()
        optimizer_support.step()
        model_fore_train_task_support.eval()
        support_state = copy.deepcopy(model_fore_train_task_support.state_dict())
        torch.save(support_state, support_model_path)

        writer1.add_scalar(f"loss_penalty_train_task_support_{meta_tag}", loss1.item(), i_t)
        writer2.add_scalar(f"loss_mse_train_task_support_{meta_tag}", loss2.item(), i_t)
        writer2.add_scalar(f"loss_anchor_train_task_support_{meta_tag}", anchor_loss_support.item(), i_t)

        print(
            "[##################################################################"
            f"——{meta_tag}:train_task_query_Epoch {i_t}/{epoch_train_task}——"
            "############################################################]"
        )

        model_fore_train_task_query.load_state_dict(copy.deepcopy(support_state))

        if disable_lwp:
            freeze_lwp_as_identity(model_fore_train_task_query)

        model_fore_train_task_query.train()
        Train_target_query = Train_target_query.to(device)
        Train_input_query = Train_input_query.to(device)
        Train_outputs_query_ = model_fore_train_task_query(Train_input_query)
        if use_cdrm:
            loss1_q = penalty(Train_outputs_query_, Train_target_query)
        else:
            loss1_q = torch.zeros((), dtype=torch.float32, device=device)
        loss2_q = loss_fn_1(Train_outputs_query_, Train_target_query)
        loss_en_q = 10 * loss1_q + loss2_q if use_cdrm else loss2_q
        anchor_loss_q = torch.zeros((), dtype=torch.float32, device=device)
        if shared_anchor_beta > 0:
            anchor_loss_q = compute_shared_anchor_loss(model_fore_train_task_query, prior_anchor_state)
            loss_en_q = loss_en_q + shared_anchor_beta * anchor_loss_q
        optimizer_query.zero_grad()
        loss_en_q.backward()
        optimizer_query.step()
        model_fore_train_task_query.eval()
        torch.save(model_fore_train_task_query.state_dict(), query_model_path)

        writer1.add_scalar(f"loss_penalty_train_task_query_{meta_tag}", loss1_q.item(), i_t)
        writer2.add_scalar(f"loss_mse_train_task_query_{meta_tag}", loss2_q.item(), i_t)
        writer2.add_scalar(f"loss_anchor_train_task_query_{meta_tag}", anchor_loss_q.item(), i_t)

    print(f"✓ 场站 {station_id} 元训练完成: {query_model_path}")


# 1) Proposed: Federated pre-train 初始化后，各场站独立做 local meta-training
proposed_init_state = torch.load(PRETRAIN_MODEL_PATH)
for station_id in station_ids:
    run_local_meta_training(
        station_id=station_id,
        meta_tag=f"proposed_station{station_id}",
        init_state_dict=proposed_init_state,
        support_model_path=get_proposed_support_model_path(station_id),
        query_model_path=get_proposed_meta_model_path(station_id),
        epoch_train_task=PROPOSED_META_EPOCHS,
        use_cdrm=True,
        train_all_params=False,
        disable_lwp=False,
        shared_anchor_beta=PROPOSED_META_SHARED_ANCHOR_BETA,
        shared_lr_scale=PROPOSED_META_SHARED_LR_SCALE
    )

# 2) Local_Meta_Transfer: 本地 conventional pretrain 初始化后，各场站独立 local meta-training
for station_id in station_ids:
    run_local_meta_training(
        station_id=station_id,
        meta_tag=f"local_meta_station{station_id}",
        init_state_dict=local_pretrain_state_dicts[station_id],
        support_model_path=get_local_meta_support_model_path(station_id),
        query_model_path=get_local_meta_model_path(station_id),
        epoch_train_task=PROPOSED_META_EPOCHS,
        use_cdrm=True,
        train_all_params=False,
        disable_lwp=False,
        shared_anchor_beta=0.0,
        shared_lr_scale=1.0
    )

# 3) Meta-only: 随机初始化后各场站独立 local meta-training
if TRAIN_META_ONLY_BASELINE:
    for station_id in station_ids:
        run_local_meta_training(
            station_id=station_id,
            meta_tag=f"meta_only_station{station_id}",
            init_state_dict=meta_only_random_init_state,
            support_model_path=get_meta_only_support_model_path(station_id),
            query_model_path=get_meta_only_model_path(station_id),
            epoch_train_task=META_ONLY_META_EPOCHS,
            use_cdrm=META_ONLY_USE_CDRM,
            train_all_params=META_ONLY_TRAIN_ALL_PARAMS,
            disable_lwp=META_ONLY_DISABLE_LWP,
            shared_anchor_beta=0.0,
            shared_lr_scale=1.0
        )


## test_task_support
few_shot_model_count = len(station_ids) * 4 * (2 if TRAIN_META_ONLY_BASELINE else 1)
print(f"##################################################################——————————test_task_support（Few-shot适应：共{few_shot_model_count}个模型）——————————############################################################")

# ========== [联邦修改] 为所有场站的所有极端天气类别训练个性化模型 ==========
all_personalized_models = {}  # 存储所有个性化模型

def run_few_shot_adaptation(base_model_path, save_path, log_tag, model_label, test_input_tensor, test_target_tensor):
    """针对某个初始化模型执行一次 few-shot 适应并保存。"""
    model_fore_test_task_support.load_state_dict(torch.load(base_model_path))
    optimizer = torch.optim.Adam(
        model_fore_test_task_support.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999)
    )

    test_input_device = test_input_tensor.to(device)
    test_target_device = test_target_tensor.to(device)

    for i in range(FEW_SHOT_EPOCHS):
        model_fore_test_task_support.train()
        test_outputs_support = model_fore_test_task_support(test_input_device)
        loss2 = loss_fn_1(test_outputs_support, test_target_device)
        loss_en = loss2
        optimizer.zero_grad()
        loss_en.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            print(
                f"      [{model_label}] [Epoch {i+1}/{FEW_SHOT_EPOCHS}] "
                f"[loss_mse: {loss2.item():.6f}]"
            )
            writer1.add_scalar(f"loss_penalty_{log_tag}", 0.0, i)
            writer2.add_scalar(f"loss_mse_{log_tag}", loss2.item(), i)

    model_fore_test_task_support.eval()
    torch.save(model_fore_test_task_support.state_dict(), save_path)
    print(f"    ✓ 保存({model_label}): {save_path}")

for station_id in station_ids:
    print(f"\n{'='*70}")
    print(f"场站 {station_id} 的Few-shot适应")
    print(f"{'='*70}")
    
    for i_class in range(4):
        print(f"\n  极端天气类别 {i_class+1}:")

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
        print(
            f"    训练轮数: {FEW_SHOT_EPOCHS}, "
            f"few-shot loss={'CDRM+MSE' if FEW_SHOT_USE_CDRM else 'MSE'}"
        )

        # Proposed：按论文流程用 proposed meta-model 做 per-class few-shot
        proposed_model_name = f"./model_fore_station{station_id}_extreme{i_class}.pth"
        run_few_shot_adaptation(
            base_model_path=get_proposed_meta_model_path(station_id),
            save_path=proposed_model_name,
            log_tag=f"station{station_id}_class{i_class}",
            model_label="Proposed",
            test_input_tensor=Test_input_support,
            test_target_tensor=Test_target_support
        )
        all_personalized_models[f'proposed_{station_id}_class{i_class}'] = proposed_model_name

        # Local_Meta_Transfer：本地 pretrain + local meta 后做 per-class few-shot
        local_meta_model_name = f"./model_fore_station{station_id}_extreme{i_class}_local_meta.pth"
        run_few_shot_adaptation(
            base_model_path=get_local_meta_model_path(station_id),
            save_path=local_meta_model_name,
            log_tag=f"local_meta_station{station_id}_class{i_class}",
            model_label="Local_Meta_Transfer",
            test_input_tensor=Test_input_support,
            test_target_tensor=Test_target_support
        )
        all_personalized_models[f'local_meta_{station_id}_class{i_class}'] = local_meta_model_name

        # Transfer_Learning：本地 pretrain 后直接 few-shot
        transfer_model_name = f"./model_fore_station{station_id}_extreme{i_class}_transfer_only.pth"
        run_few_shot_adaptation(
            base_model_path=get_local_pretrain_model_path(station_id),
            save_path=transfer_model_name,
            log_tag=f"transfer_station{station_id}_class{i_class}",
            model_label="Transfer_Learning",
            test_input_tensor=Test_input_support,
            test_target_tensor=Test_target_support
        )
        all_personalized_models[f'transfer_{station_id}_class{i_class}'] = transfer_model_name

        # Meta-only：同口径执行 step-11 few-shot，确保与论文消融对齐
        if TRAIN_META_ONLY_BASELINE:
            meta_only_model_name = f"./model_fore_station{station_id}_extreme{i_class}_meta_only.pth"
            run_few_shot_adaptation(
                base_model_path=get_meta_only_model_path(station_id),
                save_path=meta_only_model_name,
                log_tag=f"meta_only_station{station_id}_class{i_class}",
                model_label="Meta-only",
                test_input_tensor=Test_input_support,
                test_target_tensor=Test_target_support
            )
            all_personalized_models[f'meta_only_{station_id}_class{i_class}'] = meta_only_model_name

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
    
    # 预测：元学习模型（meta-only baseline）
    meta_model_path = get_meta_only_model_path(station_id) if TRAIN_META_ONLY_BASELINE else get_proposed_meta_model_path(station_id)
    model_fore_test_task_query.load_state_dict(torch.load(meta_model_path))
    with torch.no_grad():
        Test_input_device = Test_input_c_st.to(device)
        Test_output = model_fore_test_task_query(Test_input_device)
        test_output = Test_output.to(device0)
        test_output_np = np.array(test_output.reshape(-1,dem_realp))
        all_test_results[station_id]['predictions']['meta'] = test_output_np
    print(f"  ✓ 元学习模型")
    
    # 预测：本地预训练模型
    model_fore_test_task_query.load_state_dict(torch.load(get_local_pretrain_model_path(station_id)))
    with torch.no_grad():
        Test_input_device = Test_input_c_st.to(device)
        Test_output = model_fore_test_task_query(Test_input_device)
        test_output = Test_output.to(device0)
        test_output_np = np.array(test_output.reshape(-1,dem_realp))
        all_test_results[station_id]['predictions']['local_pre'] = test_output_np
    print(f"  ✓ 本地预训练模型")

    # 预测：联邦预训练模型（辅助口径，非主表）
    model_fore_test_task_query.load_state_dict(torch.load(PRETRAIN_MODEL_PATH))
    with torch.no_grad():
        Test_input_device = Test_input_c_st.to(device)
        Test_output = model_fore_test_task_query(Test_input_device)
        test_output = Test_output.to(device0)
        test_output_np = np.array(test_output.reshape(-1,dem_realp))
        all_test_results[station_id]['predictions']['fed_pre'] = test_output_np
    print(f"  ✓ 联邦预训练模型")

# 保存所有结果
print("\n保存所有场站测试结果...")
scio.savemat('all_stations_test_results.mat', {'all_test_results': all_test_results, 'Cap': Cap})
print("✓ 已保存: all_stations_test_results.mat")

print("\n" + "="*70)
print("✓✓✓ 训练和测试全部完成！")
if TRAIN_META_ONLY_BASELINE:
    print(f"生成的模型: {len(all_personalized_models)}个个性化模型（Proposed+Local_Meta_Transfer+Transfer_Learning+Meta-only） + 3类元模型 + 联邦/本地预训练模型")
else:
    print(f"生成的模型: {len(all_personalized_models)}个个性化模型（Proposed+Local_Meta_Transfer+Transfer_Learning） + 2类元模型 + 联邦/本地预训练模型")
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
