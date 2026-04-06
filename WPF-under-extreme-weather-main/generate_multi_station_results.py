#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成多场站测试结果CSV
支持3场站×6模型的完整评估
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import scipy.io as scio
from scipy import stats
import model


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
STRICT_PAPER_ORDER = os.getenv("STRICT_PAPER_ORDER", "1") != "0"
SEASONAL_PROTOCOL_ENABLED = os.getenv("SEASONAL_PROTOCOL_ENABLED", "0") != "0"
SEASONAL_PROTOCOL_METADATA_PATH = os.getenv(
    "SEASONAL_PROTOCOL_METADATA_PATH",
    "seasonal_protocol_data/seasonal_protocol_metadata.json",
)
YEARLY_PROTOCOL_ENABLED = os.getenv("YEARLY_PROTOCOL_ENABLED", "0") != "0"
YEARLY_PROTOCOL_METADATA_PATH = os.getenv(
    "YEARLY_PROTOCOL_METADATA_PATH",
    "three_station_yearly_protocol_data/three_station_yearly_protocol_metadata.json",
)
FED_PRETRAIN_MODEL_PATH = "model_fore_pre_federated.pth"
LEGACY_PRETRAIN_MODEL_PATH = "model_fore_pre.pth"
LOCAL_PRETRAIN_MODEL_TEMPLATE = "model_fore_pre_station{station_id}_local.pth"
STATION_LOCAL_PROPOSED_META_MODEL_TEMPLATE = "model_fore_train_task_query_proposed_station{station_id}.pth"
STATION_LOCAL_LOCAL_META_MODEL_TEMPLATE = "model_fore_train_task_query_local_meta_station{station_id}.pth"
STATION_LOCAL_META_ONLY_MODEL_TEMPLATE = "model_fore_train_task_query_meta_only_station{station_id}.pth"
LEGACY_META_ONLY_MODEL_PATH = "model_fore_train_task_query_meta_only.pth"
STATION_RESULT_TEMPLATE = "station{station_id}_test_results.mat"
SEASONAL_EXTREME_CLASS_NAMES = ["high_wind", "high_temp", "cold_wave", "frost"]
YEARLY_EXTREME_CLASS_NAMES = ["high_wind", "high_temp", "cold_wave", "frost"]
YEARLY_TABLE_IV_COLUMNS = [
    "Station",
    "Model",
    "HighWind_E_M_%",
    "HighWind_E_R_%",
    "HighWind_WD",
    "HighTemperature_E_M_%",
    "HighTemperature_E_R_%",
    "HighTemperature_WD",
    "ColdWave_E_M_%",
    "ColdWave_E_R_%",
    "ColdWave_WD",
    "Frost_E_M_%",
    "Frost_E_R_%",
    "Frost_WD",
    "Training_duration_s",
    "R_p<0.05_%",
]

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


def resolve_proposed_model_path(station_id, class_idx):
    """
    Proposed 子模型优先级：
    1) *_tuned.pth（若存在）
    2) 默认 few-shot 模型
    """
    candidates = []
    if PREFER_TUNED_PROPOSED_MODELS:
        candidates.append(f"model_fore_station{station_id}_extreme{class_idx}_tuned.pth")
    candidates.append(f"model_fore_station{station_id}_extreme{class_idx}.pth")

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_local_meta_transfer_model_path(station_id, class_idx):
    candidates = [
        f"model_fore_station{station_id}_extreme{class_idx}_local_meta_tuned.pth",
        f"model_fore_station{station_id}_extreme{class_idx}_local_meta.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_transfer_learning_model_path(station_id, class_idx):
    candidates = [
        f"model_fore_station{station_id}_extreme{class_idx}_transfer_only_tuned.pth",
        f"model_fore_station{station_id}_extreme{class_idx}_transfer_only.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_lmt_new_model_path(station_id, class_idx):
    candidates = [
        f"model_fore_station{station_id}_extreme{class_idx}_lmt_new_tuned.pth",
        f"model_fore_station{station_id}_extreme{class_idx}_lmt_new.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_extreme_fedavg_model_path(station_id, class_idx):
    candidates = [
        f"model_fore_station{station_id}_extreme{class_idx}_extreme_fedavg_tuned.pth",
        f"model_fore_station{station_id}_extreme{class_idx}_extreme_fedavg.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_proposed_a_model_path(station_id, class_idx):
    candidates = []
    if PREFER_TUNED_PROPOSED_MODELS:
        candidates.append(f"model_fore_station{station_id}_extreme{class_idx}_tuned.pth")
    candidates.append(f"model_fore_station{station_id}_extreme{class_idx}.pth")
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_output_model_names():
    return [
        'Proposed',
        'Local_Meta_Transfer',
        'Transfer_Learning',
        'Meta_Learning',
        'Local_PreTraining',
    ]


def resolve_yearly_output_model_names():
    return [
        "LMT-new",
        "Extreme-FedAvg",
        "Proposed-A",
    ]


def ensure_expected_model_rows(wide_df, station_order, model_names):
    existing = set(zip(wide_df['Station'].astype(str), wide_df['Model'].astype(str)))
    filler_rows = []
    fill_cols = [c for c in wide_df.columns if c not in {'Station', 'Model'}]

    for station in station_order:
        for model_name in model_names:
            if (station, model_name) in existing:
                continue
            row = {'Station': station, 'Model': model_name}
            for col in fill_cols:
                row[col] = np.nan
            filler_rows.append(row)

    if not filler_rows:
        return wide_df

    return pd.concat([wide_df, pd.DataFrame(filler_rows)], ignore_index=True, sort=False)


def resolve_station_pretrain_model_path(station_id):
    candidates = [
        LOCAL_PRETRAIN_MODEL_TEMPLATE.format(station_id=station_id),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def resolve_meta_learning_model_path(station_id, class_idx):
    meta_model_candidates = [
        f"model_fore_station{station_id}_extreme{class_idx}_meta_only_tuned.pth",
        f"model_fore_station{station_id}_extreme{class_idx}_meta_only.pth",
        STATION_LOCAL_META_ONLY_MODEL_TEMPLATE.format(station_id=station_id),
        LEGACY_META_ONLY_MODEL_PATH,
    ]
    for cand in meta_model_candidates:
        if os.path.exists(cand):
            return cand
    return None


def load_seasonal_protocol_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    client_map = {}
    for client in metadata.get("clients", []):
        client_copy = dict(client)
        client_copy["asset_path"] = os.path.join(
            os.path.dirname(metadata_path),
            client_copy["asset_path"],
        )
        client_copy["valid_class_index_set"] = set(client_copy.get("valid_class_indices", []))
        client_map[str(client_copy["client_id"])] = client_copy

    metadata["client_map"] = client_map
    return metadata


def load_yearly_protocol_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    station_map = {}
    for station in metadata.get("stations", []):
        station_copy = dict(station)
        station_copy["asset_path"] = os.path.join(
            os.path.dirname(metadata_path),
            station_copy["asset_path"],
        )
        support_counts = station_copy.get("extreme_support_window_counts", {})
        test_counts = station_copy.get("extreme_test_window_counts", {})
        valid_class_indices = []
        for class_index, class_name in enumerate(YEARLY_EXTREME_CLASS_NAMES):
            if int(support_counts.get(class_name, 0)) > 0 and int(test_counts.get(class_name, 0)) > 0:
                valid_class_indices.append(class_index)
        station_copy["valid_class_indices"] = valid_class_indices
        station_copy["valid_class_index_set"] = set(valid_class_indices)
        station_map[str(station_copy["station_id"])] = station_copy

    metadata["station_map"] = station_map
    return metadata


def iter_valid_protocol_tasks(client_metadata):
    client_id = str(client_metadata["client_id"])
    for class_index in client_metadata.get("valid_class_indices", []):
        class_index = int(class_index)
        yield {
            "client_id": client_id,
            "extreme_class_index": class_index,
            "extreme_class": SEASONAL_EXTREME_CLASS_NAMES[class_index],
        }


def iter_valid_yearly_protocol_tasks(station_metadata):
    station_id = str(station_metadata["station_id"])
    for class_index in station_metadata.get("valid_class_indices", []):
        class_index = int(class_index)
        yield {
            "station_id": station_id,
            "extreme_class_index": class_index,
            "extreme_class": YEARLY_EXTREME_CLASS_NAMES[class_index],
        }


def validate_paper_ablation_order(wide_df, weather_order):
    """
    联邦多场站主消融排序校验（误差类指标越小越好）：
    - Proposed <= Local_Meta_Transfer
    - Local_Meta_Transfer <= Transfer_Learning
    - Local_Meta_Transfer <= Meta_Learning
    - Local_Meta_Transfer <= Local_PreTraining
    """
    metric_suffixes = ["nMAE_%", "nRMSE_%", "WD_%"]
    issues = []

    station_values = [str(s) for s in wide_df["Station"].astype(str).tolist()]
    for station_id in sorted(set(station_values)):
        if station_id == "Overall_Average":
            continue

        station_rows = wide_df[wide_df["Station"].astype(str) == station_id].set_index("Model")
        required_models = {"Proposed", "Local_Meta_Transfer", "Transfer_Learning", "Meta_Learning", "Local_PreTraining"}
        if not required_models.issubset(set(station_rows.index)):
            continue

        for weather in weather_order:
            for metric_suffix in metric_suffixes:
                col = f"{weather}_{metric_suffix}"
                proposed_v = float(station_rows.loc["Proposed", col])
                local_meta_v = float(station_rows.loc["Local_Meta_Transfer", col])
                transfer_v = float(station_rows.loc["Transfer_Learning", col])
                meta_v = float(station_rows.loc["Meta_Learning", col])
                local_pre_v = float(station_rows.loc["Local_PreTraining", col])

                if not np.isnan(proposed_v) and not np.isnan(local_meta_v) and proposed_v > local_meta_v:
                    issues.append(
                        f"Station {station_id} {col}: Proposed({proposed_v:.4f}) > Local_Meta_Transfer({local_meta_v:.4f})"
                    )
                if not np.isnan(local_meta_v) and not np.isnan(transfer_v) and local_meta_v > transfer_v:
                    issues.append(
                        f"Station {station_id} {col}: Local_Meta_Transfer({local_meta_v:.4f}) > Transfer_Learning({transfer_v:.4f})"
                    )
                if not np.isnan(local_meta_v) and not np.isnan(meta_v) and local_meta_v > meta_v:
                    issues.append(
                        f"Station {station_id} {col}: Local_Meta_Transfer({local_meta_v:.4f}) > Meta_Learning({meta_v:.4f})"
                    )
                if not np.isnan(local_meta_v) and not np.isnan(local_pre_v) and local_meta_v > local_pre_v:
                    issues.append(
                        f"Station {station_id} {col}: Local_Meta_Transfer({local_meta_v:.4f}) > Local_PreTraining({local_pre_v:.4f})"
                    )

    return issues


def infer_training_durations_from_tensorboard(station_ids=None):
    """
    从最新 TensorBoard 事件文件推断训练时长（秒）。
    口径：
    - Proposed = fed pretrain + proposed meta + few-shot
    - Local_Meta_Transfer = local pretrain + local meta + few-shot
    - Transfer_Learning = local pretrain + few-shot
    - Meta_Learning = meta-only + few-shot
    - Local_PreTraining = local pretrain
    """
    duration_map = {
        'Proposed': np.nan,
        'Local_Meta_Transfer': np.nan,
        'Transfer_Learning': np.nan,
        'Meta_Learning': np.nan,
        'Local_PreTraining': np.nan,
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

    def span_for_tags(tags):
        starts = []
        ends = []
        for tag in tags:
            if tag not in scalar_tags:
                continue
            events = ea.Scalars(tag)
            if events:
                starts.append(events[0].wall_time)
                ends.append(events[-1].wall_time)
        if starts and ends:
            return float(max(ends) - min(starts))
        return np.nan

    pretrain_sec = tag_span_seconds('loss_mse_pre')
    if station_ids is None:
        station_ids = ['58', '59', '60']
    local_pretrain_sec = span_for_tags([
        f'loss_mse_pre_local_station{station_id}'
        for station_id in station_ids
    ])
    proposed_meta_sec = span_for_tags([
        tag
        for station_id in station_ids
        for tag in [
            f'loss_mse_train_task_support_proposed_station{station_id}',
            f'loss_mse_train_task_query_proposed_station{station_id}',
        ]
    ])
    local_meta_sec = span_for_tags([
        tag
        for station_id in station_ids
        for tag in [
            f'loss_mse_train_task_support_local_meta_station{station_id}',
            f'loss_mse_train_task_query_local_meta_station{station_id}',
        ]
    ])
    meta_only_sec = span_for_tags([
        tag
        for station_id in station_ids
        for tag in [
            f'loss_mse_train_task_support_meta_only_station{station_id}',
            f'loss_mse_train_task_query_meta_only_station{station_id}',
        ]
    ])

    proposed_few_shot_tags = sorted(
        [
            t for t in scalar_tags
            if t.startswith('loss_mse_station') and 'meta_only' not in t
        ]
    )
    proposed_few_shot_sec = np.nan
    if proposed_few_shot_tags:
        starts = []
        ends = []
        for t in proposed_few_shot_tags:
            events = ea.Scalars(t)
            if events:
                starts.append(events[0].wall_time)
                ends.append(events[-1].wall_time)
        if starts and ends:
            proposed_few_shot_sec = float(max(ends) - min(starts))

    local_meta_few_shot_sec = span_for_tags(
        [t for t in scalar_tags if t.startswith('loss_mse_local_meta_station')]
    )
    transfer_few_shot_sec = span_for_tags(
        [t for t in scalar_tags if t.startswith('loss_mse_transfer_station')]
    )
    meta_only_few_shot_tags = sorted(
        [t for t in scalar_tags if t.startswith('loss_mse_meta_only_station')]
    )
    meta_only_few_shot_sec = np.nan
    if meta_only_few_shot_tags:
        starts = []
        ends = []
        for t in meta_only_few_shot_tags:
            events = ea.Scalars(t)
            if events:
                starts.append(events[0].wall_time)
                ends.append(events[-1].wall_time)
        if starts and ends:
            meta_only_few_shot_sec = float(max(ends) - min(starts))

    duration_map['Local_PreTraining'] = local_pretrain_sec
    if not np.isnan(local_pretrain_sec) and not np.isnan(local_meta_sec):
        duration_map['Local_Meta_Transfer'] = local_pretrain_sec + local_meta_sec + (0.0 if np.isnan(local_meta_few_shot_sec) else local_meta_few_shot_sec)
    if not np.isnan(local_pretrain_sec):
        duration_map['Transfer_Learning'] = local_pretrain_sec + (0.0 if np.isnan(transfer_few_shot_sec) else transfer_few_shot_sec)
    if not np.isnan(meta_only_sec):
        duration_map['Meta_Learning'] = meta_only_sec + (0.0 if np.isnan(meta_only_few_shot_sec) else meta_only_few_shot_sec)
    if not np.isnan(pretrain_sec) and not np.isnan(proposed_meta_sec):
        duration_map['Proposed'] = pretrain_sec + proposed_meta_sec + (0.0 if np.isnan(proposed_few_shot_sec) else proposed_few_shot_sec)
    elif not np.isnan(pretrain_sec):
        duration_map['Proposed'] = pretrain_sec + (0.0 if np.isnan(proposed_few_shot_sec) else proposed_few_shot_sec)

    return duration_map


def run_seasonal_protocol_evaluation():
    print("\n检测到 seasonal protocol 模式，切换为逐 (client_id, extreme_class) 任务导出。")
    seasonal_protocol_metadata = load_seasonal_protocol_metadata(SEASONAL_PROTOCOL_METADATA_PATH)
    station_ids = [str(client["client_id"]) for client in seasonal_protocol_metadata.get("clients", [])]
    model_names = resolve_output_model_names()
    cap_norm = 1.0
    dem_realc = 5
    len_realp = int(seasonal_protocol_metadata.get("len_realp", 12))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device0 = torch.device("cpu")
    duration_map = infer_training_durations_from_tensorboard(station_ids=station_ids)

    model_test = model_fore(
        input_channel_fore=dem_realc,
        output_channel_fore=[128, 96, 64, 48, 32, 16, 8],
        mode='test_task_support',
    )
    model_test = model_test.to(device)
    model_test.eval()

    all_results = []

    for client_id in station_ids:
        client_metadata = seasonal_protocol_metadata["client_map"][client_id]
        print(f"\nclient {client_id} ({client_metadata.get('client_name', client_id)}):")
        wf_1 = scio.loadmat(client_metadata["asset_path"])
        reference_nwp = wf_1["nwp_1h"]

        proposed_model_files = {}
        local_meta_model_files = {}
        transfer_model_files = {}
        meta_model_files = {}
        for task in iter_valid_protocol_tasks(client_metadata):
            class_index = task["extreme_class_index"]
            proposed_model_files[class_index] = resolve_proposed_model_path(client_id, class_index)
            local_meta_model_files[class_index] = resolve_local_meta_transfer_model_path(client_id, class_index)
            transfer_model_files[class_index] = resolve_transfer_learning_model_path(client_id, class_index)
            meta_model_files[class_index] = resolve_meta_learning_model_path(client_id, class_index)
        local_pretrain_file = resolve_station_pretrain_model_path(client_id)

        for task in iter_valid_protocol_tasks(client_metadata):
            class_index = task["extreme_class_index"]
            p_extre = wf_1[f"p_test_extre_class{class_index+1}"]
            nwp_extre = wf_1[f"nwp_test_extre_class{class_index+1}_"]
            num_samples = int(p_extre.shape[0] // len_realp)
            if num_samples == 0:
                continue

            nwp_extre_list = []
            for feature_index in range(dem_realc):
                nwp_data = nwp_extre[0, feature_index].reshape(-1, 1)
                scale = float(np.max(np.abs(reference_nwp[:, feature_index]), axis=0))
                if scale < 1e-8:
                    scale = 1.0
                nwp_extre_list.append(nwp_data / scale)

            nwp_extre_concat = np.concatenate(nwp_extre_list, axis=1)
            nwp_extre_class = nwp_extre_concat[:num_samples * len_realp].reshape(
                num_samples, len_realp, dem_realc
            )
            true_events = p_extre[:num_samples * len_realp].reshape(num_samples, len_realp)
            input_class_tensor = torch.tensor(nwp_extre_class, dtype=torch.float32)

            for model_name in model_names:
                if model_name == 'Proposed':
                    model_file = proposed_model_files.get(class_index)
                elif model_name == 'Local_Meta_Transfer':
                    model_file = local_meta_model_files.get(class_index)
                elif model_name == 'Transfer_Learning':
                    model_file = transfer_model_files.get(class_index)
                elif model_name == 'Meta_Learning':
                    model_file = meta_model_files.get(class_index)
                else:
                    model_file = local_pretrain_file

                row = {
                    'client_id': client_id,
                    'client_name': client_metadata.get('client_name', client_id),
                    'source_station_id': client_metadata.get('source_station_id'),
                    'extreme_class': task["extreme_class"],
                    'Extreme_Class': f'Extreme_Weather_Class{class_index+1}',
                    'Model': model_name,
                    'support_windows': int(client_metadata.get('support_window_counts', {}).get(task["extreme_class"], 0)),
                    'test_windows': int(client_metadata.get('test_window_counts', {}).get(task["extreme_class"], num_samples)),
                    'Samples': int(num_samples),
                    'Training_duration_s': duration_map.get(model_name, np.nan),
                }

                if model_file is None:
                    row.update({
                        'nMAE_%': np.nan,
                        'nRMSE_%': np.nan,
                        'WD_%': np.nan,
                        'R_p<0.05_%': np.nan,
                    })
                    all_results.append(row)
                    continue

                model_test.load_state_dict(torch.load(model_file, map_location=device))
                model_test.eval()
                with torch.no_grad():
                    output_class = model_test(input_class_tensor.to(device))
                    pred_events = output_class.to(device0).numpy().reshape(num_samples, len_realp)

                nmae_percent, nrmse_percent, wd_percent, rp_less_005_percent = calc_paper_metrics(
                    true_events, pred_events, cap_norm=cap_norm
                )
                row.update({
                    'nMAE_%': round(nmae_percent, 4),
                    'nRMSE_%': round(nrmse_percent, 4),
                    'WD_%': round(wd_percent, 4),
                    'R_p<0.05_%': round(rp_less_005_percent, 4),
                })
                all_results.append(row)

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        raise RuntimeError("seasonal protocol 模式下没有生成任何有效测试任务结果。")

    model_order = {name: index for index, name in enumerate(model_names)}
    class_order = {name: index for index, name in enumerate(SEASONAL_EXTREME_CLASS_NAMES)}
    results_df["client_sort_key"] = results_df["client_id"].astype(int)
    results_df["class_sort_key"] = results_df["extreme_class"].map(class_order)
    results_df["model_sort_key"] = results_df["Model"].map(model_order)
    results_df = results_df.sort_values(
        ["client_sort_key", "class_sort_key", "model_sort_key"]
    ).drop(columns=["client_sort_key", "class_sort_key", "model_sort_key"])

    metric_cols = ['nMAE_%', 'nRMSE_%', 'WD_%', 'R_p<0.05_%', 'Training_duration_s']
    for col in metric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').round(4)

    output_columns = [
        'client_id',
        'client_name',
        'source_station_id',
        'extreme_class',
        'Model',
        'support_windows',
        'test_windows',
        'Samples',
        'nMAE_%',
        'nRMSE_%',
        'WD_%',
        'R_p<0.05_%',
        'Training_duration_s',
    ]
    results_df = results_df[output_columns]
    results_df.to_csv('multi_station_performance.csv', index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print("✓✓✓ six-client seasonal protocol 结果已生成")
    print("=" * 70)
    print("输出文件: multi_station_performance.csv")
    print(f"任务行数: {len(results_df)}")
    print(results_df.to_string(index=False))


def infer_yearly_training_durations_from_tensorboard(station_ids=None):
    duration_map = {
        "LMT-new": np.nan,
        "Extreme-FedAvg": np.nan,
        "Proposed-A": np.nan,
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

    if station_ids is None:
        station_ids = ['58', '59', '60']

    def span_for_tags(tags):
        starts = []
        ends = []
        for tag in tags:
            if tag not in scalar_tags:
                continue
            events = ea.Scalars(tag)
            if events:
                starts.append(events[0].wall_time)
                ends.append(events[-1].wall_time)
        if starts and ends:
            return float(max(ends) - min(starts))
        return np.nan

    local_pretrain_sec = span_for_tags([
        f'loss_mse_pre_local_station{station_id}'
        for station_id in station_ids
    ])
    local_meta_sec = span_for_tags([
        tag
        for station_id in station_ids
        for tag in [
            f'loss_mse_train_task_support_local_meta_station{station_id}',
            f'loss_mse_train_task_query_local_meta_station{station_id}',
        ]
    ])
    lmt_new_few_shot_sec = span_for_tags(
        [tag for tag in scalar_tags if tag.startswith('loss_mse_lmt_new_station')]
    )
    extreme_fedavg_few_shot_sec = span_for_tags(
        [tag for tag in scalar_tags if tag.startswith('loss_mse_extreme_fedavg_target')]
    )
    proposed_a_few_shot_sec = span_for_tags(
        [tag for tag in scalar_tags if tag.startswith('loss_mse_proposed_a_target')]
    )

    if np.isnan(local_pretrain_sec) or np.isnan(local_meta_sec):
        return duration_map

    shared_prefix_sec = local_pretrain_sec + local_meta_sec
    duration_map["LMT-new"] = shared_prefix_sec + (0.0 if np.isnan(lmt_new_few_shot_sec) else lmt_new_few_shot_sec)
    duration_map["Extreme-FedAvg"] = shared_prefix_sec + (0.0 if np.isnan(extreme_fedavg_few_shot_sec) else extreme_fedavg_few_shot_sec)
    duration_map["Proposed-A"] = shared_prefix_sec + (0.0 if np.isnan(proposed_a_few_shot_sec) else proposed_a_few_shot_sec)
    return duration_map


def build_yearly_task_level_results_df(all_results, model_names, duration_map):
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        raise RuntimeError("yearly protocol 模式下没有生成任何有效测试任务结果。")

    model_order = {name: index for index, name in enumerate(model_names)}
    class_order = {name: index for index, name in enumerate(YEARLY_EXTREME_CLASS_NAMES)}
    results_df["station_sort_key"] = results_df["station_id"].astype(int)
    results_df["class_sort_key"] = results_df["extreme_class"].map(class_order)
    results_df["model_sort_key"] = results_df["Model"].map(model_order)
    results_df = results_df.sort_values(
        ["station_sort_key", "class_sort_key", "model_sort_key"]
    ).drop(columns=["station_sort_key", "class_sort_key", "model_sort_key"])

    metric_cols = ['nMAE_%', 'nRMSE_%', 'WD_%', 'R_p<0.05_%']
    for col in metric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').round(4)
    results_df['Training_duration_s'] = pd.to_numeric(
        results_df['Model'].map(duration_map),
        errors='coerce',
    ).round(2)

    output_columns = [
        'station_id',
        'source_station_id',
        'extreme_class',
        'Model',
        'support_windows',
        'test_windows',
        'Samples',
        'nMAE_%',
        'nRMSE_%',
        'WD_%',
        'R_p<0.05_%',
        'Training_duration_s',
    ]
    return results_df[output_columns]


def build_yearly_table_iv_results_df(station_ids, yearly_task_records, model_names, duration_map, cap_norm=1.0):
    weather_name_map = {
        "high_wind": "HighWind",
        "high_temp": "HighTemperature",
        "cold_wave": "ColdWave",
        "frost": "Frost",
    }
    station_order = {str(station_id): index for index, station_id in enumerate(station_ids)}
    model_order = {name: index for index, name in enumerate(model_names)}
    wide_rows = []

    for station_id in station_ids:
        station_id = str(station_id)
        for model_name in model_names:
            row = {"Station": station_id, "Model": model_name}
            station_model_records = [
                record for record in yearly_task_records
                if record["station_id"] == station_id and record["Model"] == model_name
            ]

            for extreme_class, weather_key in weather_name_map.items():
                class_records = [
                    record for record in station_model_records
                    if record["extreme_class"] == extreme_class
                ]
                if class_records:
                    true_events = np.concatenate([record["true_events"] for record in class_records], axis=0)
                    pred_events = np.concatenate([record["pred_events"] for record in class_records], axis=0)
                    nmae_percent, nrmse_percent, wd_percent, _ = calc_paper_metrics(
                        true_events,
                        pred_events,
                        cap_norm=cap_norm,
                    )
                    row[f"{weather_key}_E_M_%"] = round(nmae_percent, 4)
                    row[f"{weather_key}_E_R_%"] = round(nrmse_percent, 4)
                    row[f"{weather_key}_WD"] = round(wd_percent, 4)
                else:
                    row[f"{weather_key}_E_M_%"] = np.nan
                    row[f"{weather_key}_E_R_%"] = np.nan
                    row[f"{weather_key}_WD"] = np.nan

            if station_model_records:
                true_events = np.concatenate([record["true_events"] for record in station_model_records], axis=0)
                pred_events = np.concatenate([record["pred_events"] for record in station_model_records], axis=0)
                _, _, _, rp_less_005_percent = calc_paper_metrics(
                    true_events,
                    pred_events,
                    cap_norm=cap_norm,
                )
                row["R_p<0.05_%"] = round(rp_less_005_percent, 4)
            else:
                row["R_p<0.05_%"] = np.nan

            row["Training_duration_s"] = duration_map.get(model_name, np.nan)
            wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    wide_df["station_sort_key"] = wide_df["Station"].map(lambda value: station_order.get(str(value), len(station_order)))
    wide_df["model_sort_key"] = wide_df["Model"].map(lambda value: model_order.get(value, len(model_order)))
    wide_df = wide_df.sort_values(["station_sort_key", "model_sort_key"]).drop(
        columns=["station_sort_key", "model_sort_key"]
    )
    wide_df = wide_df[YEARLY_TABLE_IV_COLUMNS]

    metric_cols = [col for col in wide_df.columns if col not in {"Station", "Model", "Training_duration_s"}]
    wide_df[metric_cols] = wide_df[metric_cols].apply(pd.to_numeric, errors='coerce').round(4)
    wide_df["Station"] = wide_df["Station"].astype(str)
    wide_df["Training_duration_s"] = pd.to_numeric(wide_df["Training_duration_s"], errors='coerce').round(2)
    return wide_df


def run_yearly_protocol_evaluation():
    print("\n检测到 yearly protocol 模式，切换为 task-level + TABLE IV 导出。")
    yearly_protocol_metadata = load_yearly_protocol_metadata(YEARLY_PROTOCOL_METADATA_PATH)
    station_ids = [str(station["station_id"]) for station in yearly_protocol_metadata.get("stations", [])]
    model_names = resolve_yearly_output_model_names()
    duration_map = infer_yearly_training_durations_from_tensorboard(station_ids=station_ids)
    cap_norm = 1.0
    dem_realc = 5
    len_realp = int(yearly_protocol_metadata.get("len_realp", 12))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device0 = torch.device("cpu")

    model_test = model_fore(
        input_channel_fore=dem_realc,
        output_channel_fore=[128, 96, 64, 48, 32, 16, 8],
        mode='test_task_support',
    )
    model_test = model_test.to(device)
    model_test.eval()

    all_results = []
    yearly_task_records = []

    for station_id in station_ids:
        station_metadata = yearly_protocol_metadata["station_map"][station_id]
        print(f"\nstation {station_id}:")
        wf_1 = scio.loadmat(station_metadata["asset_path"])
        reference_nwp = wf_1["nwp_1h"]

        lmt_new_model_files = {}
        extreme_fedavg_model_files = {}
        proposed_a_model_files = {}
        for task in iter_valid_yearly_protocol_tasks(station_metadata):
            class_index = task["extreme_class_index"]
            lmt_new_model_files[class_index] = resolve_lmt_new_model_path(station_id, class_index)
            extreme_fedavg_model_files[class_index] = resolve_extreme_fedavg_model_path(station_id, class_index)
            proposed_a_model_files[class_index] = resolve_proposed_a_model_path(station_id, class_index)

        for task in iter_valid_yearly_protocol_tasks(station_metadata):
            class_index = task["extreme_class_index"]
            p_extre = wf_1[f"p_test_extre_class{class_index+1}"]
            nwp_extre = wf_1[f"nwp_test_extre_class{class_index+1}_"]
            num_samples = int(p_extre.shape[0] // len_realp)
            if num_samples == 0:
                continue

            nwp_extre_list = []
            for feature_index in range(dem_realc):
                nwp_data = nwp_extre[0, feature_index].reshape(-1, 1)
                scale = float(np.max(np.abs(reference_nwp[:, feature_index]), axis=0))
                if scale < 1e-8:
                    scale = 1.0
                nwp_extre_list.append(nwp_data / scale)

            nwp_extre_concat = np.concatenate(nwp_extre_list, axis=1)
            nwp_extre_class = nwp_extre_concat[:num_samples * len_realp].reshape(
                num_samples, len_realp, dem_realc
            )
            true_events = p_extre[:num_samples * len_realp].reshape(num_samples, len_realp)
            input_class_tensor = torch.tensor(nwp_extre_class, dtype=torch.float32)

            for model_name in model_names:
                if model_name == "LMT-new":
                    model_file = lmt_new_model_files.get(class_index)
                elif model_name == "Extreme-FedAvg":
                    model_file = extreme_fedavg_model_files.get(class_index)
                else:
                    model_file = proposed_a_model_files.get(class_index)

                row = {
                    'station_id': station_id,
                    'source_station_id': station_metadata.get('source_station_id', station_id),
                    'extreme_class': task["extreme_class"],
                    'Model': model_name,
                    'support_windows': int(station_metadata.get('extreme_support_window_counts', {}).get(task["extreme_class"], 0)),
                    'test_windows': int(station_metadata.get('extreme_test_window_counts', {}).get(task["extreme_class"], num_samples)),
                    'Samples': int(num_samples),
                }

                if model_file is None:
                    row.update({
                        'nMAE_%': np.nan,
                        'nRMSE_%': np.nan,
                        'WD_%': np.nan,
                        'R_p<0.05_%': np.nan,
                    })
                    all_results.append(row)
                    continue

                model_test.load_state_dict(torch.load(model_file, map_location=device))
                model_test.eval()
                with torch.no_grad():
                    output_class = model_test(input_class_tensor.to(device))
                    pred_events = output_class.to(device0).numpy().reshape(num_samples, len_realp)

                nmae_percent, nrmse_percent, wd_percent, rp_less_005_percent = calc_paper_metrics(
                    true_events,
                    pred_events,
                    cap_norm=cap_norm,
                )
                row.update({
                    'nMAE_%': round(nmae_percent, 4),
                    'nRMSE_%': round(nrmse_percent, 4),
                    'WD_%': round(wd_percent, 4),
                    'R_p<0.05_%': round(rp_less_005_percent, 4),
                })
                all_results.append(row)
                yearly_task_records.append(
                    {
                        "station_id": station_id,
                        "extreme_class": task["extreme_class"],
                        "Model": model_name,
                        "true_events": true_events,
                        "pred_events": pred_events,
                    }
                )

    task_level_df = build_yearly_task_level_results_df(all_results, model_names, duration_map)
    table_iv_df = build_yearly_table_iv_results_df(
        station_ids,
        yearly_task_records,
        model_names,
        duration_map,
        cap_norm=cap_norm,
    )
    task_level_df.to_csv('multi_station_performance_task_level.csv', index=False, encoding='utf-8-sig')
    table_iv_df.to_csv('multi_station_performance.csv', index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print("✓✓✓ yearly extreme protocol 结果已生成")
    print("=" * 70)
    print("输出文件: multi_station_performance.csv")
    print("附加文件: multi_station_performance_task_level.csv")
    print(f"task-level 行数: {len(task_level_df)}")
    print(table_iv_df.to_string(index=False))

# 参数
# 当前训练/评估数据为标幺值，因此按论文公式中的 Cap 取 1.0
cap_norm = 1.0
dem_realc = 5
dem_realp = 1
len_realp = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0 = torch.device("cpu")

if YEARLY_PROTOCOL_ENABLED and SEASONAL_PROTOCOL_ENABLED:
    raise RuntimeError("YEARLY_PROTOCOL_ENABLED 与 SEASONAL_PROTOCOL_ENABLED 不能同时开启。")

if YEARLY_PROTOCOL_ENABLED:
    run_yearly_protocol_evaluation()
    raise SystemExit(0)

if SEASONAL_PROTOCOL_ENABLED:
    run_seasonal_protocol_evaluation()
    raise SystemExit(0)

# 场站列表
station_ids = ['58', '59', '60']
meta_learning_available = all(
    resolve_meta_learning_model_path(station_id, class_idx) is not None
    for station_id in station_ids
    for class_idx in range(4)
)

station_result_files = [
    STATION_RESULT_TEMPLATE.format(station_id=station_id)
    for station_id in station_ids
]
if any(os.path.exists(path) for path in station_result_files):
    print("\n检测到每站结果文件。当前评估仍按极端天气类别重跑模型，不直接复用全年测试集预测。")
print("\n当前评估口径：主表固定为 Proposed / Local_Meta_Transfer / Transfer_Learning / Meta_Learning / Local_PreTraining。")
if not meta_learning_available:
    print("注意：当前目录缺少完整 Meta_Learning 产物，对应行将以 NaN 导出。")

# 输出到表格的模型名称（按论文表风格）
model_names = resolve_output_model_names()

# 创建模型
model_test = model_fore(
    input_channel_fore=dem_realc,
    output_channel_fore=[128, 96, 64, 48, 32, 16, 8],
    mode='test_task_support',
)
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
    
    proposed_model_files = {}
    local_meta_model_files = {}
    transfer_model_files = {}
    meta_model_files = {}

    for i_class in range(4):
        proposed_file = resolve_proposed_model_path(station_id, i_class)
        proposed_model_files[i_class] = proposed_file
        if proposed_file is not None:
            if proposed_file.endswith("_tuned.pth"):
                print(f"  ✓ Proposed(Class{i_class+1}) 使用 tuned: {proposed_file}")
            else:
                print(f"  ✓ Proposed(Class{i_class+1}) 使用默认: {proposed_file}")
        else:
            print(f"  ✗ Proposed(Class{i_class+1}) 模型不存在（默认/调优均缺失）")

        local_meta_file = resolve_local_meta_transfer_model_path(station_id, i_class)
        local_meta_model_files[i_class] = local_meta_file
        if local_meta_file is not None:
            print(f"  ✓ Local_Meta_Transfer(Class{i_class+1}) 使用: {local_meta_file}")
        else:
            print(f"  ✗ Local_Meta_Transfer(Class{i_class+1}) 无可用模型")

        transfer_file = resolve_transfer_learning_model_path(station_id, i_class)
        transfer_model_files[i_class] = transfer_file
        if transfer_file is not None:
            print(f"  ✓ Transfer_Learning(Class{i_class+1}) 使用: {transfer_file}")
        else:
            print(f"  ✗ Transfer_Learning(Class{i_class+1}) 无可用模型")

        meta_model_file = resolve_meta_learning_model_path(station_id, i_class)
        meta_model_files[i_class] = meta_model_file
        if meta_model_file is not None:
            print(f"  ✓ Meta_Learning(Class{i_class+1}) 使用: {meta_model_file}")
        else:
            print(f"  ✗ Meta_Learning(Class{i_class+1}) 无可用模型")

    local_pretrain_file = resolve_station_pretrain_model_path(station_id)
    if local_pretrain_file is None:
        print(f"  ✗ 场站{station_id} 的 Local_PreTraining 模型不存在")
    else:
        print(f"  ✓ Local_PreTraining 使用: {local_pretrain_file}")

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
            elif model_name == 'Local_Meta_Transfer':
                model_file = local_meta_model_files.get(eval_class)
            elif model_name == 'Transfer_Learning':
                model_file = transfer_model_files.get(eval_class)
            elif model_name == 'Meta_Learning':
                model_file = meta_model_files.get(eval_class)
            else:
                model_file = local_pretrain_file
            if model_file is None:
                all_results.append({
                    'Station': station_id,
                    'Extreme_Class': f'Extreme_Weather_Class{eval_class+1}',
                    'Model': model_name,
                    'Samples': int(num_samples),
                    'nMAE_%': np.nan,
                    'nRMSE_%': np.nan,
                    'WD_%': np.nan,
                    'R_p<0.05_%': np.nan
                })
                continue
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
wide_df = ensure_expected_model_rows(wide_df, ['58', '59', '60', 'Overall_Average'], model_names)

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

# 论文消融关系一致性检查
paper_order_issues = validate_paper_ablation_order(
    wide_df,
    weather_order,
)
if paper_order_issues:
    print("\n" + "=" * 70)
    print("WARNING: 5组主表消融排序校验未通过（Proposed <= Local_Meta_Transfer <= {Transfer_Learning, Meta_Learning, Local_PreTraining}）")
    print("=" * 70)
    for issue in paper_order_issues[:30]:
        print(" - " + issue)
    if len(paper_order_issues) > 30:
        print(f" - ... 其余 {len(paper_order_issues) - 30} 条省略")
    if STRICT_PAPER_ORDER:
        raise RuntimeError(
            "检测到与论文消融排序冲突的结果。请先重训/更新基线模型后再生成CSV，"
            "或将 STRICT_PAPER_ORDER 设为 False。"
        )

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
print(f"  - 每场站: {len(model_names)}模型 = {len(model_names)}行")
print(f"  - 3场站: {len(model_names) * 3}行")
print(f"  - Overall Average: {len(model_names)}行")
print(f"  - 总计: {len(model_names) * 4}行")

print("\n" + "="*70)
print("性能对比表格（横向展开）:")
print("="*70)
print(wide_df.to_string(index=False))

print("\n" + "="*70)
print("Overall Average（论文口径，横向展开）:")
print("="*70)
print(wide_df[wide_df['Station'] == 'Overall_Average'].to_string(index=False))
print("="*70)
