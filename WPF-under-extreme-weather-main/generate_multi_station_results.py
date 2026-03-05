#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成多场站测试结果CSV
支持3场站×6模型的完整评估
"""
import os
import glob
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import scipy.io as scio
from scipy import stats
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

def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR 校正。
    输入/输出均为一维 p-value 数组，输出为校正后的 q-value。
    """
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    if n == 0:
        return p_values

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = sorted_p[i] * n / rank
        if val > prev:
            val = prev
        prev = min(val, 1.0)
        adjusted_sorted[i] = prev

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def calc_fdr_significance_ratio(true_events, pred_events, alpha=0.05):
    """
    论文口径中的 R_{p<0.05}:
    逐事件做配对 t 检验得到 p-value，做 BH-FDR 校正后，统计 q<0.05 的事件占比。
    返回百分比（0-100）。
    """
    p_values = []
    for true_event, pred_event in zip(true_events, pred_events):
        if np.allclose(true_event, pred_event):
            p_values.append(1.0)
            continue
        _, p_val = stats.ttest_rel(pred_event, true_event, nan_policy='omit')
        if np.isnan(p_val):
            p_val = 1.0
        p_values.append(float(p_val))

    if len(p_values) == 0:
        return np.nan

    q_values = benjamini_hochberg(np.array(p_values, dtype=float))
    return float(np.mean(q_values < alpha) * 100.0)


def calc_paper_metrics(true_events, pred_events, cap_norm=1.0):
    """
    论文口径:
    nRMSE = mean_e sqrt(mean_j(((y-ŷ)/Cap)^2)) * 100
    nMAE  = mean_e mean_j(|(y-ŷ)/Cap|) * 100
    WD    = mean_e mean_j(|(sort(y)-sort(ŷ))/Cap|) * 100
    Rp<0.05 = FDR校正后显著样本占比 * 100
    注: 当前数据为标幺值, 等价于 Cap=1.
    """
    err = (true_events - pred_events) / cap_norm
    nrmse_per_event = np.sqrt(np.mean(err ** 2, axis=1)) * 100
    nmae_per_event = np.mean(np.abs(err), axis=1) * 100
    wd_per_event = np.mean(
        np.abs((np.sort(true_events, axis=1) - np.sort(pred_events, axis=1)) / cap_norm),
        axis=1
    ) * 100
    rp_less_005 = calc_fdr_significance_ratio(true_events, pred_events, alpha=0.05)
    return (
        float(np.mean(nmae_per_event)),
        float(np.mean(nrmse_per_event)),
        float(np.mean(wd_per_event)),
        rp_less_005
    )


def infer_training_durations_from_tensorboard():
    """
    从最新 TensorBoard 事件文件推断训练时长（秒）。
    口径：
    - Pre_Training = pre-train 阶段时长
    - Meta_Learning = meta-only 阶段时长
    - Proposed = pre-train + proposed meta-training + few-shot 适配总时长
    """
    duration_map = {
        'Proposed': np.nan,
        'Meta_Learning': np.nan,
        'Pre_Training': np.nan
    }

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return duration_map

    event_files = glob.glob(os.path.join('logs_train', 'loss2', 'events.out.tfevents.*'))
    if not event_files:
        return duration_map

    latest_event_file = max(event_files, key=os.path.getmtime)
    ea = event_accumulator.EventAccumulator(latest_event_file, size_guidance={'scalars': 0})
    ea.Reload()
    scalar_tags = set(ea.Tags().get('scalars', []))

    def tag_span_seconds(tag):
        if tag not in scalar_tags:
            return np.nan
        events = ea.Scalars(tag)
        if len(events) < 2:
            return np.nan
        return float(events[-1].wall_time - events[0].wall_time)

    def max_valid(*vals):
        valid = [v for v in vals if not np.isnan(v)]
        return float(max(valid)) if valid else np.nan

    pretrain_sec = tag_span_seconds('loss_mse_pre')
    proposed_meta_sec = max_valid(
        tag_span_seconds('loss_mse_train_task_support_proposed'),
        tag_span_seconds('loss_mse_train_task_query_proposed')
    )
    meta_only_sec = max_valid(
        tag_span_seconds('loss_mse_train_task_support_meta_only'),
        tag_span_seconds('loss_mse_train_task_query_meta_only')
    )

    few_shot_tags = sorted([t for t in scalar_tags if t.startswith('loss_mse_station')])
    few_shot_sec = np.nan
    if few_shot_tags:
        starts = []
        ends = []
        for t in few_shot_tags:
            events = ea.Scalars(t)
            if events:
                starts.append(events[0].wall_time)
                ends.append(events[-1].wall_time)
        if starts and ends:
            few_shot_sec = float(max(ends) - min(starts))

    duration_map['Pre_Training'] = pretrain_sec
    duration_map['Meta_Learning'] = meta_only_sec
    if not np.isnan(pretrain_sec) and not np.isnan(proposed_meta_sec):
        duration_map['Proposed'] = pretrain_sec + proposed_meta_sec + (0.0 if np.isnan(few_shot_sec) else few_shot_sec)

    return duration_map

# 参数
# 当前训练/评估数据为标幺值，因此按论文公式中的 Cap 取 1.0
cap_norm = 1.0
dem_realc = 5
dem_realp = 1
len_realp = 12
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

# 输出到表格的模型名称（按论文表风格）
model_names = [
    'Proposed',
    'Meta_Learning',
    'Pre_Training'
]

# 创建模型
model_test = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8], mode='test_task_support')
model_test = model_test.to(device)
model_test.eval()  # 推理必须eval，否则dropout会导致结果随机漂移

# 收集所有场站结果（论文口径）
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
    
    # 生成该场站模型文件映射
    # Proposed 在各天气类别上使用对应的 class 模型
    proposed_model_files = {}

    # 1-4: 类别对应的 Proposed 子模型
    for i_class in range(4):
        model_file = f'model_fore_station{station_id}_extreme{i_class}.pth'
        proposed_model_files[i_class] = model_file if os.path.exists(model_file) else None
        if os.path.exists(model_file):
            print(f"  ✓ Proposed(Class{i_class+1})")
        else:
            print(f"  ✗ {model_file} 不存在")

    baseline_model_files = {}

    # Meta-learning（论文 Table IV: meta-learning only）
    meta_model_candidates = [
        'model_fore_train_task_query_meta_only.pth',
        'model_fore_meta_only.pth',
        'model_fore_train_task_query.pth'
    ]
    meta_model_file = None
    for cand in meta_model_candidates:
        if os.path.exists(cand):
            meta_model_file = cand
            break

    if meta_model_file is not None:
        baseline_model_files['Meta_Learning'] = meta_model_file
        print(f"  ✓ Meta_Learning ({meta_model_file})")
    else:
        baseline_model_files['Meta_Learning'] = None
        print(f"  ✗ Meta_Learning模型不存在（尝试: {meta_model_candidates}）")

    # Pre-training
    pre_model_file = 'model_fore_pre_federated.pth' if os.path.exists('model_fore_pre_federated.pth') else 'model_fore_pre.pth'
    baseline_model_files['Pre_Training'] = pre_model_file if os.path.exists(pre_model_file) else None
    if baseline_model_files['Pre_Training'] is None:
        print(f"  ✗ {pre_model_file} 不存在")
    else:
        print("  ✓ Pre_Training")

    # 论文口径：在每个极端天气类别子集上评估各方法
    for eval_class in range(4):
        p_extre = wf_1[f'p_extre_class{eval_class+1}']
        nwp_extre = wf_1[f'nwp_extre_class{eval_class+1}_']
        num_samples = p_extre.shape[0] // len_realp
        if num_samples == 0:
            continue
        
        nwp_extre_list = []
        for i_nwp in range(dem_realc):
            nwp_data = nwp_extre[0, i_nwp].reshape(-1, 1)
            nwp_norm = nwp_data / np.max(abs(P_nwp[:, i_nwp]), axis=0)
            nwp_extre_list.append(nwp_norm)
        
        nwp_extre_concat = np.concatenate(nwp_extre_list, axis=1)
        nwp_extre_class = nwp_extre_concat[:num_samples*len_realp].reshape(
            num_samples, len_realp, dem_realc
        )
        true_events = p_extre[:num_samples*len_realp].reshape(num_samples, len_realp)
        input_class_tensor = torch.tensor(nwp_extre_class, dtype=torch.float32)
        
        for model_name in model_names:
            if model_name == 'Proposed':
                model_file = proposed_model_files.get(eval_class)
            else:
                model_file = baseline_model_files.get(model_name)
            if model_file is None:
                pred_events = np.zeros_like(true_events)
            else:
                model_test.load_state_dict(torch.load(model_file, map_location=device))
                model_test.eval()
                with torch.no_grad():
                    output_class = model_test(input_class_tensor.to(device))
                    pred_events = output_class.to(device0).numpy().reshape(num_samples, len_realp)

            nmae_percent, nrmse_percent, wd_percent, rp_less_005_percent = calc_paper_metrics(
                true_events, pred_events, cap_norm=cap_norm
            )
            all_results.append({
                'Station': station_id,
                'Extreme_Class': f'Extreme_Weather_Class{eval_class+1}',
                'Model': model_name,
                'Samples': int(num_samples),
                'nMAE_%': round(nmae_percent, 4),
                'nRMSE_%': round(nrmse_percent, 4),
                'WD_%': round(wd_percent, 4),
                'R_p<0.05_%': round(rp_less_005_percent, 4)
            })

print("\n计算Overall Average（按Extreme_Class + Model跨3场站平均）...")
results_long_df = pd.DataFrame(all_results)
overall_long_df = (
    results_long_df.groupby(['Extreme_Class', 'Model'], as_index=False)[
        ['Samples', 'nMAE_%', 'nRMSE_%', 'WD_%', 'R_p<0.05_%']
    ]
    .mean()
    .round(4)
)
overall_long_df.insert(0, 'Station', 'Overall_Average')
results_long_df = pd.concat([results_long_df, overall_long_df], ignore_index=True)

# 转为论文 Table III/IV 风格：每个模型一行，四类天气横向展开
weather_name_map = {
    'Extreme_Weather_Class1': 'HighWind',
    'Extreme_Weather_Class2': 'HighTemperature',
    'Extreme_Weather_Class3': 'ColdWave',
    'Extreme_Weather_Class4': 'Frost'
}
weather_order = ['HighWind', 'HighTemperature', 'ColdWave', 'Frost']
metric_order = ['nMAE_%', 'nRMSE_%', 'WD_%']

table_df = results_long_df.copy()
table_df['Weather'] = table_df['Extreme_Class'].map(weather_name_map)
table_df['Model'] = pd.Categorical(table_df['Model'], categories=model_names, ordered=True)

wide_df = table_df.pivot_table(
    index=['Station', 'Model'],
    columns='Weather',
    values=metric_order,
    aggfunc='first'
)

ordered_columns = []
for weather in weather_order:
    for metric in metric_order:
        key = (metric, weather)
        if key in wide_df.columns:
            ordered_columns.append(key)
wide_df = wide_df[ordered_columns]
wide_df.columns = [f'{weather}_{metric}' for metric, weather in wide_df.columns]
wide_df = wide_df.reset_index()

# 追加一个全类别加权的 R_p<0.05（按样本数加权）
rp_all_class_df = (
    table_df.groupby(['Station', 'Model'], as_index=False)
    .apply(lambda g: np.average(g['R_p<0.05_%'], weights=g['Samples']))
    .rename(columns={None: 'AllClasses_R_p<0.05_%'})
)
wide_df = wide_df.merge(rp_all_class_df, on=['Station', 'Model'], how='left')

# 追加训练时长（秒）
duration_map = infer_training_durations_from_tensorboard()
wide_df['Training_duration_s'] = wide_df['Model'].map(duration_map)
wide_df = wide_df.rename(columns={'AllClasses_R_p<0.05_%': 'R_p<0.05_%'})

# 排序
station_order = ['58', '59', '60', 'Overall_Average']
wide_df['Station'] = pd.Categorical(wide_df['Station'], categories=station_order, ordered=True)
wide_df = wide_df.sort_values(['Station', 'Model']).reset_index(drop=True)
wide_df['Station'] = wide_df['Station'].astype(str)

# 输出列顺序：四类天气指标 + 训练时长 + 总R_p<0.05
output_cols = ['Station', 'Model']
for weather in weather_order:
    for metric in metric_order:
        output_cols.append(f'{weather}_{metric}')
output_cols.extend(['Training_duration_s', 'R_p<0.05_%'])
wide_df = wide_df[output_cols]

# 保存为CSV（论文表格风格）
metric_cols = [c for c in wide_df.columns if c not in ['Station', 'Model', 'Training_duration_s']]
wide_df[metric_cols] = wide_df[metric_cols].round(4)
wide_df['Training_duration_s'] = pd.to_numeric(wide_df['Training_duration_s'], errors='coerce')
wide_df['Training_duration_s'] = wide_df['Training_duration_s'].round(2)
wide_df.to_csv('multi_station_performance.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*70)
print("✓✓✓ 多场站结果已生成（Table III/IV 风格）！")
print("="*70)
print(f"\n生成文件: multi_station_performance.csv")
print(f"总行数: {len(wide_df)}")
print(f"  - 每场站: 3模型 = 3行")
print(f"  - 3场站: 9行")
print(f"  - Overall Average: 3行")
print(f"  - 总计: 12行")

print("\n" + "="*70)
print("性能对比表格（横向展开）:")
print("="*70)
print(wide_df.to_string(index=False))

print("\n" + "="*70)
print("Overall Average（论文口径，横向展开）:")
print("="*70)
print(wide_df[wide_df['Station'] == 'Overall_Average'].to_string(index=False))
print("="*70)
