#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成多场站测试结果CSV
支持多场站 extreme 主表评估
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

PREFER_TUNED_PROPOSED_MODELS = os.getenv("PREFER_TUNED_PROPOSED_MODELS", "0") != "0"


def load_protocol_metadata(metadata_path):
    if not metadata_path or not os.path.exists(metadata_path):
        return {}
    import json
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


PROTOCOL_METADATA_PATH = os.getenv("PROTOCOL_METADATA_PATH", os.getenv("YEARLY_PROTOCOL_METADATA_PATH", ""))
protocol_metadata = load_protocol_metadata(PROTOCOL_METADATA_PATH)
PROTOCOL_NAME = os.getenv("PROTOCOL_NAME", protocol_metadata.get("protocol_name", "legacy_1h_12point"))
PROTOCOL_DATA_DIR = os.getenv("PROTOCOL_DATA_DIR", protocol_metadata.get("protocol_data_dir", ""))
LEN_REALP = int(os.getenv("LEN_REALP", str(protocol_metadata.get("len_realp", 12))))
POINTS_PER_DAY = int(os.getenv("POINTS_PER_DAY", str(protocol_metadata.get("points_per_day", 24))))
SAMPLE_INTERVAL_HOURS = int(
    os.getenv(
        "SAMPLE_INTERVAL_HOURS",
        str(protocol_metadata.get("sample_interval_hours", max(1, 24 // max(1, POINTS_PER_DAY)))),
    )
)
DOWNSAMPLE_OFFSET = int(os.getenv("DOWNSAMPLE_OFFSET", str(protocol_metadata.get("downsample_offset", 0))))
WINDOW_SPAN_HOURS = int(
    os.getenv(
        "WINDOW_SPAN_HOURS",
        str(protocol_metadata.get("window_span_hours", SAMPLE_INTERVAL_HOURS * LEN_REALP)),
    )
)
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", ".")


def resolve_artifact_path(filename):
    if os.path.isabs(filename):
        return filename
    if ARTIFACT_DIR in ("", "."):
        return filename
    return os.path.join(ARTIFACT_DIR, filename)


MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", resolve_artifact_path("models") if ARTIFACT_DIR not in ("", ".") else ".")
LOGS_TRAIN_DIR = os.getenv("LOGS_TRAIN_DIR", resolve_artifact_path("logs_train"))
TASK_RESULTS_OUTPUT_PATH = os.getenv("TASK_RESULTS_OUTPUT_PATH", "multi_station_performance_task_level.csv")
RESULTS_OUTPUT_PATH = os.getenv("RESULTS_OUTPUT_PATH", "multi_station_performance.csv")


def resolve_model_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(MODEL_OUTPUT_DIR, filename)


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def resolve_station_ids():
    eval_station_ids = os.getenv("EVAL_STATION_IDS", "")
    if eval_station_ids:
        return [station_id.strip() for station_id in eval_station_ids.split(",") if station_id.strip()]
    metadata_stations = protocol_metadata.get("stations", [])
    if metadata_stations:
        return [str(station["station_id"]) for station in metadata_stations]
    return ['58', '59', '60']


def resolve_station_mat_path(station_id):
    filename = f"{station_id}wf_4_train.mat"
    if PROTOCOL_DATA_DIR:
        candidate = os.path.join(PROTOCOL_DATA_DIR, filename)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Protocol mat missing: {candidate}")
        return candidate
    return filename


def get_extreme_eval_payload(wf_1, class_index):
    test_power_key = f"p_test_extre_class{class_index + 1}"
    test_nwp_key = f"nwp_test_extre_class{class_index + 1}_"
    support_power_key = f"p_extre_class{class_index + 1}"
    support_nwp_key = f"nwp_extre_class{class_index + 1}_"
    power_key = test_power_key if test_power_key in wf_1 else support_power_key
    nwp_key = test_nwp_key if test_nwp_key in wf_1 else support_nwp_key
    return wf_1[power_key], wf_1[nwp_key], power_key


print("\n数据协议:")
print(f"  Protocol={PROTOCOL_NAME}")
print(f"  PROTOCOL_DATA_DIR={PROTOCOL_DATA_DIR or '(legacy root mats)'}")
print(f"  PROTOCOL_METADATA_PATH={PROTOCOL_METADATA_PATH or '(none)'}")
print(f"  SAMPLE_INTERVAL_HOURS={SAMPLE_INTERVAL_HOURS}")
print(f"  DOWNSAMPLE_OFFSET={DOWNSAMPLE_OFFSET}")
print(f"  LEN_REALP={LEN_REALP}")
print(f"  POINTS_PER_DAY={POINTS_PER_DAY}")
print(f"  WINDOW_SPAN_HOURS={WINDOW_SPAN_HOURS}")

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


def get_local_pretrain_model_path(station_id):
    return resolve_model_path(f"model_fore_pre_station{station_id}_local.pth")


def resolve_lmt_model_path(station_id, class_idx):
    """
    LMT 子模型优先级：
    1) *_tuned.pth（若存在）
    2) 默认 few-shot 模型
    """
    candidates = []
    if PREFER_TUNED_PROPOSED_MODELS:
        candidates.append(resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_tuned.pth"))
    candidates.append(resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}.pth"))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def get_extreme_fedavg_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_extreme_fedavg.pth")


def get_proposed_a_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_proposed_a.pth")


def infer_training_durations_from_tensorboard():
    """
    从最新 TensorBoard 事件文件推断训练时长（秒）。
    口径：
    - LMT = local pre-train + local meta-training + target-only extreme few-shot
    - Extreme-FedAvg = shared prefix + source update + fedavg target refine
    - Proposed-A = shared prefix + source update + proposed target refine
    """
    duration_map = {
        'LMT': np.nan,
        'Extreme-FedAvg': np.nan,
        'Proposed-A': np.nan
    }

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return duration_map

    event_files = glob.glob(os.path.join(LOGS_TRAIN_DIR, 'loss2', 'events.out.tfevents.*'))
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
    local_meta_tags = sorted([t for t in scalar_tags if t.startswith('loss_mse_train_task_query_local_meta_station')])
    local_meta_support_tags = sorted([t for t in scalar_tags if t.startswith('loss_mse_train_task_support_local_meta_station')])
    local_meta_sec = max_valid(
        *[tag_span_seconds(t) for t in local_meta_tags],
        *[tag_span_seconds(t) for t in local_meta_support_tags]
    )

    local_pretrain_tags = sorted([t for t in scalar_tags if t.startswith('loss_mse_pre_station')])
    local_pretrain_sec = max_valid(
        pretrain_sec,
        *[tag_span_seconds(t) for t in local_pretrain_tags]
    )

    def prefix_span_seconds(prefix):
        matched_tags = sorted([t for t in scalar_tags if t.startswith(prefix)])
        if not matched_tags:
            return np.nan
        starts = []
        ends = []
        for tag in matched_tags:
            events = ea.Scalars(tag)
            if events:
                starts.append(events[0].wall_time)
                ends.append(events[-1].wall_time)
        if not starts or not ends:
            return np.nan
        return float(max(ends) - min(starts))

    lmt_few_shot_sec = prefix_span_seconds('loss_mse_lmt_station')
    shared_source_update_sec = prefix_span_seconds('loss_mse_extreme_source_station')
    fedavg_refine_sec = prefix_span_seconds('loss_mse_extreme_fedavg_station')
    proposed_refine_sec = prefix_span_seconds('loss_mse_proposed_a_station')

    if not np.isnan(local_pretrain_sec) and not np.isnan(local_meta_sec):
        shared_prefix_sec = local_pretrain_sec + local_meta_sec
        duration_map['LMT'] = shared_prefix_sec + (0.0 if np.isnan(lmt_few_shot_sec) else lmt_few_shot_sec)
        duration_map['Extreme-FedAvg'] = shared_prefix_sec + (0.0 if np.isnan(shared_source_update_sec) else shared_source_update_sec) + (0.0 if np.isnan(fedavg_refine_sec) else fedavg_refine_sec)
        duration_map['Proposed-A'] = shared_prefix_sec + (0.0 if np.isnan(shared_source_update_sec) else shared_source_update_sec) + (0.0 if np.isnan(proposed_refine_sec) else proposed_refine_sec)

    return duration_map

# 参数
# 当前训练/评估数据为标幺值，因此按论文公式中的 Cap 取 1.0
cap_norm = 1.0
dem_realc = 5
dem_realp = 1
len_realp = LEN_REALP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0 = torch.device("cpu")

# 检查是否已有保存的测试结果
if os.path.exists('all_stations_test_results.mat'):
    print("\n从已保存的结果加载...")
    results_mat = scio.loadmat('all_stations_test_results.mat')
    # 这里简化处理，直接重新加载模型生成
    print("（跳过，直接重新生成）")

# 场站列表
station_ids = resolve_station_ids()

# 输出到表格的模型名称（按论文表风格）
model_names = [
    'LMT',
    'Extreme-FedAvg',
    'Proposed-A'
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
    wf_1 = scio.loadmat(resolve_station_mat_path(station_id))
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
    
    lmt_model_files = {}
    extreme_fedavg_model_files = {}
    proposed_a_model_files = {}

    # 1-4: 类别对应的 LMT 子模型（优先 tuned）
    for i_class in range(4):
        model_file = resolve_lmt_model_path(station_id, i_class)
        lmt_model_files[i_class] = model_file
        if model_file is not None:
            if model_file.endswith("_tuned.pth"):
                print(f"  ✓ LMT(Class{i_class+1}) 使用 tuned: {model_file}")
            else:
                print(f"  ✓ LMT(Class{i_class+1}) 使用默认: {model_file}")
        else:
            print(f"  ✗ LMT(Class{i_class+1}) 模型不存在（默认/调优均缺失）")

    for i_class in range(4):
        fedavg_model_file = get_extreme_fedavg_model_path(station_id, i_class)
        proposed_model_file = get_proposed_a_model_path(station_id, i_class)
        extreme_fedavg_model_files[i_class] = fedavg_model_file if os.path.exists(fedavg_model_file) else None
        proposed_a_model_files[i_class] = proposed_model_file if os.path.exists(proposed_model_file) else None
        if extreme_fedavg_model_files[i_class] is not None:
            print(f"  ✓ Extreme-FedAvg(Class{i_class+1}) 使用: {fedavg_model_file}")
        else:
            print(f"  ✗ Extreme-FedAvg(Class{i_class+1}) 模型不存在: {fedavg_model_file}")
        if proposed_a_model_files[i_class] is not None:
            print(f"  ✓ Proposed-A(Class{i_class+1}) 使用: {proposed_model_file}")
        else:
            print(f"  ✗ Proposed-A(Class{i_class+1}) 模型不存在: {proposed_model_file}")

    # 论文口径：在每个极端天气类别子集上评估各方法
    for eval_class in range(4):
        p_extre, nwp_extre, eval_payload_key = get_extreme_eval_payload(wf_1, eval_class)
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
            if model_name == 'LMT':
                model_file = lmt_model_files.get(eval_class)
            elif model_name == 'Extreme-FedAvg':
                model_file = extreme_fedavg_model_files.get(eval_class)
            else:
                model_file = proposed_a_model_files.get(eval_class)
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
                'Protocol': PROTOCOL_NAME,
                'Sample_Interval_Hours': SAMPLE_INTERVAL_HOURS,
                'Window_Points': LEN_REALP,
                'Window_Span_Hours': WINDOW_SPAN_HOURS,
                'Station': station_id,
                'Extreme_Class': f'Extreme_Weather_Class{eval_class+1}',
                'Eval_Payload_Key': eval_payload_key,
                'Model': model_name,
                'Samples': int(num_samples),
                'nMAE_%': round(nmae_percent, 4),
                'nRMSE_%': round(nrmse_percent, 4),
                'WD_%': round(wd_percent, 4),
                'R_p<0.05_%': round(rp_less_005_percent, 4)
            })

print(f"\n计算Overall Average（按Extreme_Class + Model跨{len(station_ids)}场站平均）...")
results_long_df = pd.DataFrame(all_results)
overall_long_df = (
    results_long_df.groupby(['Extreme_Class', 'Model'], as_index=False)[
        ['Samples', 'nMAE_%', 'nRMSE_%', 'WD_%', 'R_p<0.05_%']
    ]
    .mean()
    .round(4)
)
overall_long_df.insert(0, 'Station', 'Overall_Average')
overall_long_df.insert(0, 'Window_Span_Hours', WINDOW_SPAN_HOURS)
overall_long_df.insert(0, 'Window_Points', LEN_REALP)
overall_long_df.insert(0, 'Sample_Interval_Hours', SAMPLE_INTERVAL_HOURS)
overall_long_df.insert(0, 'Protocol', PROTOCOL_NAME)
overall_long_df.insert(6, 'Eval_Payload_Key', 'Overall_Average')
results_long_df = pd.concat([results_long_df, overall_long_df], ignore_index=True)
ensure_parent_dir(TASK_RESULTS_OUTPUT_PATH)
results_long_df.to_csv(TASK_RESULTS_OUTPUT_PATH, index=False, encoding='utf-8-sig')

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
wide_df.insert(0, 'Protocol', PROTOCOL_NAME)
wide_df.insert(1, 'Sample_Interval_Hours', SAMPLE_INTERVAL_HOURS)
wide_df.insert(2, 'Window_Points', LEN_REALP)
wide_df.insert(3, 'Window_Span_Hours', WINDOW_SPAN_HOURS)

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
station_order = station_ids + ['Overall_Average']
wide_df['Station'] = pd.Categorical(wide_df['Station'], categories=station_order, ordered=True)
wide_df = wide_df.sort_values(['Station', 'Model']).reset_index(drop=True)
wide_df['Station'] = wide_df['Station'].astype(str)

# 输出列顺序：四类天气指标 + 训练时长 + 总R_p<0.05
output_cols = ['Protocol', 'Sample_Interval_Hours', 'Window_Points', 'Window_Span_Hours', 'Station', 'Model']
for weather in weather_order:
    for metric in metric_order:
        output_cols.append(f'{weather}_{metric}')
output_cols.extend(['Training_duration_s', 'R_p<0.05_%'])
wide_df = wide_df[output_cols]

# 保存为CSV（论文表格风格）
metric_cols = [
    c for c in wide_df.columns
    if c not in ['Protocol', 'Sample_Interval_Hours', 'Window_Points', 'Window_Span_Hours', 'Station', 'Model', 'Training_duration_s']
]
wide_df[metric_cols] = wide_df[metric_cols].round(4)
wide_df['Training_duration_s'] = pd.to_numeric(wide_df['Training_duration_s'], errors='coerce')
wide_df['Training_duration_s'] = wide_df['Training_duration_s'].round(2)
ensure_parent_dir(RESULTS_OUTPUT_PATH)
wide_df.to_csv(RESULTS_OUTPUT_PATH, index=False, encoding='utf-8-sig')

print("\n" + "="*70)
print("✓✓✓ 多场站结果已生成（Table III/IV 风格）！")
print("="*70)
print(f"\n生成文件: {RESULTS_OUTPUT_PATH}")
print(f"总行数: {len(wide_df)}")
print(f"  - 每场站: {len(model_names)}模型 = {len(model_names)}行")
print(f"  - {len(station_ids)}场站: {len(station_ids) * len(model_names)}行")
print(f"  - Overall Average: 3行")
print(f"  - 总计: {len(wide_df)}行")

print("\n" + "="*70)
print("性能对比表格（横向展开）:")
print("="*70)
print(wide_df.to_string(index=False))

print("\n" + "="*70)
print("Overall Average（论文口径，横向展开）:")
print("="*70)
print(wide_df[wide_df['Station'] == 'Overall_Average'].to_string(index=False))
print("="*70)
