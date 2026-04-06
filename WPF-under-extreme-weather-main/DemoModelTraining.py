import os
import sys
import time
import json
import math
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


def env_str(name, default):
    value = os.getenv(name)
    return default if value is None else value.strip()


def configure_runtime_output_streams():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except AttributeError:
            continue


def progress_log(message=""):
    print(message, flush=True)


def log_stage_banner(stage_name, detail_lines=None):
    progress_log("\n" + "=" * 70)
    progress_log(stage_name)
    for detail_line in detail_lines or []:
        progress_log(f"  {detail_line}")
    progress_log("=" * 70)


def should_log_epoch(epoch_index, total_epochs, interval, warmup_epochs=10):
    current_epoch = int(epoch_index) + 1
    if current_epoch <= max(1, int(warmup_epochs)):
        return True
    if current_epoch >= int(total_epochs):
        return True
    return current_epoch % max(1, int(interval)) == 0


configure_runtime_output_streams()


# ========== [联邦新增] 联邦学习开关 ==========
USE_FEDERATION = env_flag("USE_FEDERATION", True)  # True=联邦多场站, False=单场站原方法
SEASONAL_PROTOCOL_ENABLED = env_flag("SEASONAL_PROTOCOL_ENABLED", False)
SEASONAL_PROTOCOL_METADATA_PATH = env_str(
    "SEASONAL_PROTOCOL_METADATA_PATH",
    "seasonal_protocol_data/seasonal_protocol_metadata.json"
)
YEARLY_PROTOCOL_ENABLED = env_flag("YEARLY_PROTOCOL_ENABLED", False)
YEARLY_PROTOCOL_METADATA_PATH = env_str(
    "YEARLY_PROTOCOL_METADATA_PATH",
    "three_station_yearly_protocol_data/three_station_yearly_protocol_metadata.json"
)
RUN_FEDERATED_PRETRAIN = env_flag("RUN_FEDERATED_PRETRAIN", USE_FEDERATION and not YEARLY_PROTOCOL_ENABLED)
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
PRETRAIN_LOG_INTERVAL = env_int("PRETRAIN_LOG_INTERVAL", 100)
META_LOG_INTERVAL = env_int("META_LOG_INTERVAL", 100)
FEW_SHOT_LOG_INTERVAL = env_int("FEW_SHOT_LOG_INTERVAL", 1)
PROPOSED_META_SAMPLER_MODE = env_str("PROPOSED_META_SAMPLER_MODE", "balanced").lower()
REGIME_MISSING_MODE = env_str("REGIME_MISSING_MODE", "none").lower()
# 论文消融口径：Meta-only = 去掉 pre-training，其余训练机制保持一致
META_ONLY_USE_CDRM = env_flag("META_ONLY_USE_CDRM", True)
META_ONLY_TRAIN_ALL_PARAMS = env_flag("META_ONLY_TRAIN_ALL_PARAMS", False)
META_ONLY_DISABLE_LWP = env_flag("META_ONLY_DISABLE_LWP", False)
FED_PRETRAIN_REGIME_ALPHA = env_float("FED_PRETRAIN_REGIME_ALPHA", 1.0)
FED_PRETRAIN_AGGREGATION_GAMMA = env_float("FED_PRETRAIN_AGGREGATION_GAMMA", 0.5)
PROPOSED_META_SHARED_ANCHOR_BETA = env_float("PROPOSED_META_SHARED_ANCHOR_BETA", 0.01)
PROPOSED_META_SHARED_LR_SCALE = env_float("PROPOSED_META_SHARED_LR_SCALE", 0.3)
EXTREME_WEIGHT_TAU = env_float("EXTREME_WEIGHT_TAU", 1.0)
EXTREME_WEIGHT_LAMBDA = env_float("EXTREME_WEIGHT_LAMBDA", 1.0)
EXTREME_WEIGHT_MU = env_float("EXTREME_WEIGHT_MU", 1.0)
EXTREME_WEIGHT_NU = env_float("EXTREME_WEIGHT_NU", 2.0)
ENABLE_CONVERGENCE_MONITOR = env_flag("ENABLE_CONVERGENCE_MONITOR", True)
CONVERGENCE_REPORT_PATH = env_str("CONVERGENCE_REPORT_PATH", "training_convergence_report.json")
CONVERGENCE_MIN_DELTA = env_float("CONVERGENCE_MIN_DELTA", 1e-4)
CONVERGENCE_MIN_EPOCHS = env_int("CONVERGENCE_MIN_EPOCHS", 5)
CONVERGENCE_PATIENCE_PRETRAIN = env_int("CONVERGENCE_PATIENCE_PRETRAIN", 200)
CONVERGENCE_PATIENCE_META = env_int("CONVERGENCE_PATIENCE_META", 100)
CONVERGENCE_PATIENCE_FEW_SHOT = env_int("CONVERGENCE_PATIENCE_FEW_SHOT", 5)
META_SUPPORT_SHOTS = env_int("META_SUPPORT_SHOTS", 10)
META_QUERY_SHOTS = env_int("META_QUERY_SHOTS", 10)
SUPPORTED_CONVENTIONAL_RATIOS = (1.0, 0.7, 0.5, 0.3)
CONVENTIONAL_RATIO = env_float("CONVENTIONAL_RATIO", 1.0)
CONVENTIONAL_SUBSAMPLE_BINS = 10
CONVENTIONAL_SUBSAMPLE_SEED_OFFSET = env_int("CONVENTIONAL_SUBSAMPLE_SEED_OFFSET", 0)
META_MIN_EPISODE_SAMPLES = META_SUPPORT_SHOTS + META_QUERY_SHOTS
PROPOSED_META_COVERAGE_WINDOW_FIXED = 4
DEFAULT_REGIME_MISSING_CLASS_MAP = {
    '58': (1, 2, 3, 4),
    '59': (4, 5, 6, 7),
    '60': (7, 8, 9, 10),
}


def validate_conventional_ratio(ratio):
    if not any(abs(ratio - supported_ratio) < 1e-8 for supported_ratio in SUPPORTED_CONVENTIONAL_RATIOS):
        raise ValueError(
            f"CONVENTIONAL_RATIO={ratio} 不受支持，仅支持 {SUPPORTED_CONVENTIONAL_RATIOS}"
        )
    return ratio


def validate_sampler_mode(sampler_mode):
    supported_modes = {"uniform", "balanced"}
    if sampler_mode not in supported_modes:
        raise ValueError(
            f"PROPOSED_META_SAMPLER_MODE={sampler_mode} 不受支持，仅支持 {sorted(supported_modes)}"
        )
    return sampler_mode


def validate_regime_missing_mode(regime_missing_mode):
    supported_modes = {"none", "class_dropout"}
    if regime_missing_mode not in supported_modes:
        raise ValueError(
            f"REGIME_MISSING_MODE={regime_missing_mode} 不受支持，仅支持 {sorted(supported_modes)}"
        )
    return regime_missing_mode


def validate_regime_missing_class_map(class_map, total_classes=10):
    class_presence = {class_idx: 0 for class_idx in range(1, total_classes + 1)}
    for station_id, dropped_classes in class_map.items():
        if len(dropped_classes) != 4:
            raise ValueError(f"场站 {station_id} 必须固定 drop 4 个 classes，当前为 {dropped_classes}")
        if len(set(dropped_classes)) != len(dropped_classes):
            raise ValueError(f"场站 {station_id} 的 drop classes 存在重复: {dropped_classes}")
        for class_idx in dropped_classes:
            if class_idx < 1 or class_idx > total_classes:
                raise ValueError(f"场站 {station_id} 的 class {class_idx} 超出有效范围 1..{total_classes}")
            class_presence[class_idx] += 1
    globally_missing = [class_idx for class_idx, dropped_count in class_presence.items() if dropped_count >= len(class_map)]
    if globally_missing:
        raise ValueError(f"存在在所有场站同时缺失的 classes: {globally_missing}")
    return class_map


def load_seasonal_protocol_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    client_map = {}
    for client in metadata.get("clients", []):
        client_copy = copy.deepcopy(client)
        client_copy["asset_path"] = os.path.join(
            os.path.dirname(metadata_path),
            client_copy["asset_path"]
        )
        client_copy["valid_class_index_set"] = set(client_copy.get("valid_class_indices", []))
        client_map[str(client_copy["client_id"])] = client_copy
    metadata["client_map"] = client_map
    return metadata


def load_yearly_protocol_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    station_map = {}
    extreme_class_names = metadata.get(
        "extreme_class_names",
        ["high_wind", "high_temp", "cold_wave", "frost"],
    )
    for station in metadata.get("stations", []):
        station_copy = copy.deepcopy(station)
        station_copy["asset_path"] = os.path.join(
            os.path.dirname(metadata_path),
            station_copy["asset_path"]
        )
        support_counts = station_copy.get("extreme_support_window_counts", {})
        test_counts = station_copy.get("extreme_test_window_counts", {})
        valid_class_indices = []
        for class_index, class_name in enumerate(extreme_class_names):
            if int(support_counts.get(class_name, 0)) > 0 and int(test_counts.get(class_name, 0)) > 0:
                valid_class_indices.append(class_index)
        station_copy["valid_class_indices"] = valid_class_indices
        station_copy["valid_class_index_set"] = set(valid_class_indices)
        station_map[str(station_copy["station_id"])] = station_copy
    metadata["station_map"] = station_map
    return metadata


def resolve_station_sampler_tasks_per_epoch(station_id, station_tasks, requested_tasks_per_epoch=META_TASKS_PER_EPOCH):
    if SEASONAL_PROTOCOL_ENABLED and seasonal_protocol_metadata is not None:
        station_meta = seasonal_protocol_metadata["client_map"][str(station_id)]
        sampler_task_count = int(station_meta["sampler_task_count"])
        return min(len(station_tasks), max(1, sampler_task_count))
    return resolve_local_meta_tasks_per_epoch(
        station_tasks,
        requested_tasks_per_epoch=requested_tasks_per_epoch
    )


def resolve_station_extreme_class_indices(station_id):
    if YEARLY_PROTOCOL_ENABLED and yearly_protocol_metadata is not None:
        return list(yearly_protocol_metadata["station_map"][str(station_id)]["valid_class_indices"])
    if SEASONAL_PROTOCOL_ENABLED and seasonal_protocol_metadata is not None:
        return list(seasonal_protocol_metadata["client_map"][str(station_id)]["valid_class_indices"])
    return list(range(4))


if YEARLY_PROTOCOL_ENABLED and SEASONAL_PROTOCOL_ENABLED:
    raise ValueError("YEARLY_PROTOCOL_ENABLED 与 SEASONAL_PROTOCOL_ENABLED 不能同时开启")

yearly_protocol_metadata = load_yearly_protocol_metadata(YEARLY_PROTOCOL_METADATA_PATH) if YEARLY_PROTOCOL_ENABLED else None
seasonal_protocol_metadata = load_seasonal_protocol_metadata(SEASONAL_PROTOCOL_METADATA_PATH) if SEASONAL_PROTOCOL_ENABLED else None
convergence_records = []


def initialize_convergence_record(stage_type, stage_id, total_epochs, patience, min_delta=CONVERGENCE_MIN_DELTA, min_epochs=CONVERGENCE_MIN_EPOCHS):
    return {
        "stage_type": stage_type,
        "stage_id": stage_id,
        "total_epochs": int(total_epochs),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "min_epochs": int(min_epochs),
        "converged": False,
        "convergence_epoch": None,
        "best_epoch": None,
        "best_loss": None,
        "final_loss": None,
        "_last_improved_epoch": None,
        "_last_announced_convergence_epoch": None,
    }


def update_convergence_record(record, epoch, loss_value):
    if not ENABLE_CONVERGENCE_MONITOR or record is None:
        return

    current_epoch = int(epoch) + 1
    loss_value = float(loss_value)
    record["final_loss"] = loss_value

    if record["best_loss"] is None or loss_value < (record["best_loss"] - record["min_delta"]):
        record["best_loss"] = loss_value
        record["best_epoch"] = current_epoch
        record["_last_improved_epoch"] = current_epoch
        record["converged"] = False
        record["convergence_epoch"] = None
        return

    last_improved_epoch = record["_last_improved_epoch"]
    if (
        current_epoch >= record["min_epochs"]
        and record["convergence_epoch"] is None
        and last_improved_epoch is not None
        and (current_epoch - last_improved_epoch) >= record["patience"]
    ):
        record["converged"] = True
        record["convergence_epoch"] = current_epoch
        if record.get("_last_announced_convergence_epoch") != current_epoch:
            progress_log(
                f"  收敛检测[{record['stage_type']}:{record['stage_id']}]: "
                f"首次达到收敛条件 @ epoch {current_epoch} "
                f"(best_epoch={record['best_epoch']}, best_loss={format_convergence_loss(record['best_loss'])}, "
                f"final_loss={format_convergence_loss(record['final_loss'])})"
            )
            record["_last_announced_convergence_epoch"] = current_epoch


def finalize_convergence_record(record):
    if record is None:
        return None
    finalized_record = copy.deepcopy(record)
    finalized_record.pop("_last_improved_epoch", None)
    finalized_record.pop("_last_announced_convergence_epoch", None)
    return finalized_record


def format_convergence_loss(loss_value):
    return "nan" if loss_value is None else f"{float(loss_value):.6f}"


def register_convergence_record(record):
    if not ENABLE_CONVERGENCE_MONITOR or record is None:
        return None
    finalized_record = finalize_convergence_record(record)
    convergence_records.append(finalized_record)
    if finalized_record["converged"]:
        progress_log(
            f"  收敛检测[{finalized_record['stage_type']}:{finalized_record['stage_id']}]: "
            f"已收敛 @ epoch {finalized_record['convergence_epoch']} "
            f"(best_epoch={finalized_record['best_epoch']}, best_loss={format_convergence_loss(finalized_record['best_loss'])}, "
            f"final_loss={format_convergence_loss(finalized_record['final_loss'])})"
        )
    else:
        progress_log(
            f"  收敛检测[{finalized_record['stage_type']}:{finalized_record['stage_id']}]: "
            f"未收敛 (best_epoch={finalized_record['best_epoch']}, "
            f"best_loss={format_convergence_loss(finalized_record['best_loss'])}, "
            f"final_loss={format_convergence_loss(finalized_record['final_loss'])})"
        )
    return finalized_record


def export_convergence_report(report_path, records, run_config):
    if not ENABLE_CONVERGENCE_MONITOR:
        return
    report_payload = {
        "generated_at_unix": float(time.time()),
        "run_config": run_config,
        "records": records,
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report_payload, report_file, indent=2, ensure_ascii=False)
    progress_log(f"✓ 收敛检测报告已保存: {report_path}")


def make_station_rng(station_id, offset=0, subsample_seed_offset=0):
    return random.Random(1029 + int(station_id) * 1000 + offset + subsample_seed_offset * 100000)


def subsample_time_binned_indices(total_samples, ratio, station_id, num_bins=CONVENTIONAL_SUBSAMPLE_BINS, subsample_seed_offset=0):
    if ratio >= 1.0 or total_samples <= 0:
        return np.arange(total_samples, dtype=int)

    rng = make_station_rng(station_id, offset=total_samples, subsample_seed_offset=subsample_seed_offset)
    selected_indices = []
    bin_edges = np.linspace(0, total_samples, num_bins + 1, dtype=int)

    for i_bin in range(num_bins):
        start = int(bin_edges[i_bin])
        end = int(bin_edges[i_bin + 1])
        if end <= start:
            continue
        bin_indices = list(range(start, end))
        keep_count = int(np.ceil(len(bin_indices) * ratio))
        keep_count = min(len(bin_indices), max(1, keep_count))
        if keep_count == len(bin_indices):
            selected_indices.extend(bin_indices)
        else:
            selected_indices.extend(sorted(rng.sample(bin_indices, keep_count)))

    return np.array(sorted(selected_indices), dtype=int)


def subsample_pretrain_conventional_data(clients_train_data, ratio, subsample_seed_offset=0):
    if ratio >= 1.0:
        return clients_train_data

    reduced_clients_train_data = {}
    for station_id, train_data in clients_train_data.items():
        sample_axis = 1 if train_data['input'].ndim == 3 and train_data['input'].shape[0] == 1 else 0
        total_samples = train_data['input'].shape[sample_axis]
        selected_indices = subsample_time_binned_indices(
            total_samples,
            ratio,
            station_id,
            subsample_seed_offset=subsample_seed_offset
        )
        if sample_axis == 1:
            reduced_input = train_data['input'][:, selected_indices, :]
            reduced_target = train_data['target'][:, selected_indices, :]
        else:
            reduced_input = train_data['input'][selected_indices, :, :]
            reduced_target = train_data['target'][selected_indices, :, :]
        reduced_clients_train_data[station_id] = {
            'input': reduced_input,
            'target': reduced_target
        }
        progress_log(
            f"    场站 {station_id} conventional pretrain 缩减: "
            f"{total_samples} -> {reduced_clients_train_data[station_id]['input'].shape[sample_axis]} "
            f"(ratio={ratio})"
        )
    return reduced_clients_train_data


def subsample_meta_conventional_data(all_stations_full_data, ratio, subsample_seed_offset=0):
    if ratio >= 1.0:
        return all_stations_full_data

    reduced_all_stations_full_data = copy.deepcopy(all_stations_full_data)
    for station_id, station_payload in reduced_all_stations_full_data.items():
        p_conven_class_st = station_payload['p_conven_class']
        nwp_conven_class_st = station_payload['nwp_conven_class']
        total_classes = np.size(p_conven_class_st, axis=1)

        for i_class in range(total_classes):
            p_data = p_conven_class_st[0, i_class]
            num_samples = p_data.shape[0] // len_realp
            keep_samples = min(num_samples, max(META_MIN_EPISODE_SAMPLES, int(np.ceil(num_samples * ratio))))
            rng = make_station_rng(station_id, offset=100 + i_class, subsample_seed_offset=subsample_seed_offset)
            selected_segments = sorted(rng.sample(range(num_samples), keep_samples))

            p_segments = p_data[:num_samples * len_realp].reshape(num_samples, len_realp)
            p_conven_class_st[0, i_class] = p_segments[selected_segments, :].reshape(-1, 1)

            for i_nwp in range(np.size(nwp_conven_class_st, axis=1)):
                nwp_data = nwp_conven_class_st[0, i_nwp][0, i_class]
                nwp_segments = nwp_data[:num_samples * len_realp].reshape(num_samples, len_realp)
                nwp_conven_class_st[0, i_nwp][0, i_class] = nwp_segments[selected_segments, :].reshape(-1, 1)

        min_class_segments = min(
            station_payload['p_conven_class'][0, i_class].shape[0] // len_realp
            for i_class in range(total_classes)
        )
        progress_log(
            f"    场站 {station_id} conventional meta 缩减完成: "
            f"最小类别样本段数={min_class_segments} (ratio={ratio})"
        )
    return reduced_all_stations_full_data


def build_retained_class_indices(total_classes, dropped_classes_one_based):
    dropped_class_indices = {class_idx - 1 for class_idx in dropped_classes_one_based}
    return [class_idx for class_idx in range(total_classes) if class_idx not in dropped_class_indices]


def apply_regime_missing_to_meta_data(all_stations_full_data, regime_missing_mode):
    if regime_missing_mode != "class_dropout":
        return all_stations_full_data

    reduced_all_stations_full_data = copy.deepcopy(all_stations_full_data)
    for station_id, station_payload in reduced_all_stations_full_data.items():
        dropped_classes = DEFAULT_REGIME_MISSING_CLASS_MAP.get(station_id, ())
        p_conven_class_st = station_payload['p_conven_class']
        nwp_conven_class_st = station_payload['nwp_conven_class']
        total_classes = np.size(p_conven_class_st, axis=1)
        retained_class_indices = build_retained_class_indices(total_classes, dropped_classes)

        p_conven_class_new = np.empty([1, len(retained_class_indices)], dtype=object)
        for new_class_idx, old_class_idx in enumerate(retained_class_indices):
            p_conven_class_new[0, new_class_idx] = p_conven_class_st[0, old_class_idx]

        nwp_conven_class_new = np.empty([1, np.size(nwp_conven_class_st, axis=1)], dtype=object)
        for i_nwp in range(np.size(nwp_conven_class_st, axis=1)):
            nwp_conven_class_new[0, i_nwp] = np.empty([1, len(retained_class_indices)], dtype=object)
            for new_class_idx, old_class_idx in enumerate(retained_class_indices):
                nwp_conven_class_new[0, i_nwp][0, new_class_idx] = nwp_conven_class_st[0, i_nwp][0, old_class_idx]

        station_payload['p_conven_class'] = p_conven_class_new
        station_payload['nwp_conven_class'] = nwp_conven_class_new
        station_payload['retained_class_ids'] = [class_idx + 1 for class_idx in retained_class_indices]
        progress_log(
            f"    场站 {station_id} regime-missing: drop={list(dropped_classes)}, "
            f"retain={station_payload['retained_class_ids']}"
        )
    return reduced_all_stations_full_data


def apply_regime_missing_to_pretrain_data(clients_train_data, all_stations_full_data, regime_missing_mode):
    if regime_missing_mode != "class_dropout":
        return clients_train_data

    reduced_clients_train_data = {}
    for station_id, station_payload in all_stations_full_data.items():
        p_conven_class_st = station_payload['p_conven_class']
        nwp_conven_class_st = station_payload['nwp_conven_class']
        class_inputs = []
        class_targets = []

        for i_class in range(np.size(p_conven_class_st, axis=1)):
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
            class_inputs.append(nwp_conven_class_1)
            class_targets.append(p_conven_class_1)

        reduced_clients_train_data[station_id] = {
            'input': np.concatenate(class_inputs, axis=0),
            'target': np.concatenate(class_targets, axis=0)
        }
        progress_log(
            f"    场站 {station_id} pretrain pool 经 regime-missing 重建: "
            f"{reduced_clients_train_data[station_id]['input'].shape}"
        )
    return reduced_clients_train_data

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
CONVENTIONAL_RATIO = validate_conventional_ratio(CONVENTIONAL_RATIO)
PROPOSED_META_SAMPLER_MODE = validate_sampler_mode(PROPOSED_META_SAMPLER_MODE)
REGIME_MISSING_MODE = validate_regime_missing_mode(REGIME_MISSING_MODE)
validate_regime_missing_class_map(DEFAULT_REGIME_MISSING_CLASS_MAP)
if CONVENTIONAL_RATIO < 1.0:
    log_stage_banner(
        "运行配置: conventional-ratio 必要性实验",
        [
            f"ratio={CONVENTIONAL_RATIO}",
            f"支持比例={SUPPORTED_CONVENTIONAL_RATIOS}",
            f"时间分箱数={CONVENTIONAL_SUBSAMPLE_BINS}",
            f"subsample_seed_offset={CONVENTIONAL_SUBSAMPLE_SEED_OFFSET}",
        ],
    )
else:
    log_stage_banner(
        "运行配置",
        [
            f"sampler_mode={PROPOSED_META_SAMPLER_MODE}",
            f"regime_missing_mode={REGIME_MISSING_MODE}",
            f"pretrain_log_interval={PRETRAIN_LOG_INTERVAL}",
            f"meta_log_interval={META_LOG_INTERVAL}",
            f"few_shot_log_interval={FEW_SHOT_LOG_INTERVAL}",
        ],
    )
progress_log(f"Phase 2 Proposed sampler mode: {PROPOSED_META_SAMPLER_MODE}")
progress_log(f"Regime-missing mode: {REGIME_MISSING_MODE}")
if YEARLY_PROTOCOL_ENABLED:
    progress_log(f"Yearly protocol enabled: {YEARLY_PROTOCOL_METADATA_PATH}")
if SEASONAL_PROTOCOL_ENABLED:
    progress_log(f"Seasonal protocol enabled: {SEASONAL_PROTOCOL_METADATA_PATH}")
    progress_log(f"META_SUPPORT_SHOTS={META_SUPPORT_SHOTS}, META_QUERY_SHOTS={META_QUERY_SHOTS}")
if YEARLY_PROTOCOL_ENABLED:
    progress_log(f"META_SUPPORT_SHOTS={META_SUPPORT_SHOTS}, META_QUERY_SHOTS={META_QUERY_SHOTS}")

# ========== [联邦修改] 多场站数据加载 ==========
if YEARLY_PROTOCOL_ENABLED:
    log_stage_banner("数据协议: yearly extreme protocol", ["加载3个yearly实验场站"])
    station_ids = [str(station["station_id"]) for station in yearly_protocol_metadata["stations"]]
elif SEASONAL_PROTOCOL_ENABLED:
    log_stage_banner("数据协议: seasonal scarcity protocol", ["加载6个实验客户端"])
    station_ids = [str(client["client_id"]) for client in seasonal_protocol_metadata["clients"]]
elif USE_FEDERATION:
    log_stage_banner("数据协议: federated training", ["加载3个场站数据（58/59/60）"])
    station_ids = ['58', '59', '60']  # [联邦新增] 3个场站作为客户端
else:
    log_stage_banner("数据协议: standalone training", ["加载场站58数据（原方法）"])
    station_ids = ['58']  # [原代码] 单场站

# [联邦修改] 循环加载所有场站数据
station_data = {}
for station_id in station_ids:
    if YEARLY_PROTOCOL_ENABLED:
        dataFile = yearly_protocol_metadata["station_map"][str(station_id)]["asset_path"].replace(".mat", "")
    elif SEASONAL_PROTOCOL_ENABLED:
        dataFile = seasonal_protocol_metadata["client_map"][str(station_id)]["asset_path"].replace(".mat", "")
    else:
        dataFile = f'{station_id}wf_4_train'
    progress_log(f"  加载 {dataFile}.mat...")
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
        'nwp_extre_class4_00': wf_1['nwp_extre_class4_'],
        'p_test_00': wf_1['p_test'] if 'p_test' in wf_1 else None,
        'nwp_test_00': wf_1['nwp_test'] if 'nwp_test' in wf_1 else None,
        'p_test_extre_class1_00': wf_1['p_test_extre_class1'] if 'p_test_extre_class1' in wf_1 else None,
        'p_test_extre_class2_00': wf_1['p_test_extre_class2'] if 'p_test_extre_class2' in wf_1 else None,
        'p_test_extre_class3_00': wf_1['p_test_extre_class3'] if 'p_test_extre_class3' in wf_1 else None,
        'p_test_extre_class4_00': wf_1['p_test_extre_class4'] if 'p_test_extre_class4' in wf_1 else None,
        'nwp_test_extre_class1_00': wf_1['nwp_test_extre_class1_'] if 'nwp_test_extre_class1_' in wf_1 else None,
        'nwp_test_extre_class2_00': wf_1['nwp_test_extre_class2_'] if 'nwp_test_extre_class2_' in wf_1 else None,
        'nwp_test_extre_class3_00': wf_1['nwp_test_extre_class3_'] if 'nwp_test_extre_class3_' in wf_1 else None,
        'nwp_test_extre_class4_00': wf_1['nwp_test_extre_class4_'] if 'nwp_test_extre_class4_' in wf_1 else None,
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
PRETRAIN_MODEL_PATH = "model_fore_pre_federated.pth" if RUN_FEDERATED_PRETRAIN else "model_fore_pre.pth"
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


EXTREME_FEDAVG_PLACEHOLDER_MODEL_PATH = "model_fore_train_task_query_extreme_fedavg_placeholder.pth"
PROPOSED_A_PLACEHOLDER_MODEL_PATH = "model_fore_train_task_query_proposed_a_placeholder.pth"
EXTREME_SCENARIO_DESCRIPTOR_DIM = len([0, 1, 2, 3, 4]) * 4 + 4


def resolve_yearly_extreme_base_model_path(station_id, baseline_name):
    if baseline_name == "LMT-new":
        return get_local_meta_model_path(station_id)
    if baseline_name == "Extreme-FedAvg":
        return get_local_meta_model_path(station_id)
    if baseline_name == "Proposed-A":
        return get_local_meta_model_path(station_id)
    raise ValueError(f"未知的全年极端基线: {baseline_name}")


def get_yearly_extreme_baseline_specs(station_id):
    return [
        {
            "label": "LMT-new",
            "slug": "lmt_new",
            "base_model_path": resolve_yearly_extreme_base_model_path(station_id, "LMT-new"),
            "save_path_builder": lambda class_index, sid=station_id: f"./model_fore_station{sid}_extreme{class_index}_lmt_new.pth",
            "result_key_builder": lambda class_index, sid=station_id: f"lmt_new_{sid}_class{class_index}",
        },
        {
            "label": "Extreme-FedAvg",
            "slug": "extreme_fedavg",
            "base_model_path": resolve_yearly_extreme_base_model_path(station_id, "Extreme-FedAvg"),
            "save_path_builder": lambda class_index, sid=station_id: f"./model_fore_station{sid}_extreme{class_index}_extreme_fedavg.pth",
            "result_key_builder": lambda class_index, sid=station_id: f"extreme_fedavg_{sid}_class{class_index}",
        },
        {
            "label": "Proposed-A",
            "slug": "proposed_a",
            "base_model_path": resolve_yearly_extreme_base_model_path(station_id, "Proposed-A"),
            "save_path_builder": lambda class_index, sid=station_id: f"./model_fore_station{sid}_extreme{class_index}.pth",
            "result_key_builder": lambda class_index, sid=station_id: f"proposed_a_{sid}_class{class_index}",
        },
    ]


def build_extreme_scenario_descriptor(nwp_window, power_window):
    nwp_array = np.asarray(nwp_window, dtype=np.float32)
    power_array = np.asarray(power_window, dtype=np.float32).reshape(-1)
    if nwp_array.ndim != 2:
        raise ValueError(f"nwp_window 应为二维窗口，当前 shape={nwp_array.shape}")
    if power_array.size == 0:
        raise ValueError("power_window 不能为空")

    mu_x = np.mean(nwp_array, axis=0)
    std_x = np.std(nwp_array, axis=0)
    min_x = np.min(nwp_array, axis=0)
    max_x = np.max(nwp_array, axis=0)
    mu_y = float(np.mean(power_array))
    std_y = float(np.std(power_array))
    ramp_y = float(np.mean(np.abs(np.diff(power_array)))) if power_array.size > 1 else 0.0
    drop_y = float(np.max(power_array) - np.min(power_array))

    descriptor = np.concatenate(
        [
            mu_x,
            std_x,
            min_x,
            max_x,
            np.array([mu_y, std_y, ramp_y, drop_y], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return descriptor.reshape(-1)


def normalize_prototype_features_in_client(descriptor_matrix):
    descriptor_array = np.asarray(descriptor_matrix, dtype=np.float32)
    if descriptor_array.ndim == 1:
        descriptor_array = descriptor_array.reshape(1, -1)
    if descriptor_array.shape[0] == 0:
        return descriptor_array
    descriptor_mean = np.mean(descriptor_array, axis=0, keepdims=True)
    descriptor_std = np.std(descriptor_array, axis=0, keepdims=True)
    descriptor_std = np.where(descriptor_std < 1e-8, 1.0, descriptor_std)
    return (descriptor_array - descriptor_mean) / descriptor_std


def compute_extreme_class_prototype(nwp_windows, power_windows):
    nwp_array = np.asarray(nwp_windows, dtype=np.float32)
    power_array = np.asarray(power_windows, dtype=np.float32)
    if nwp_array.ndim != 3:
        raise ValueError(f"nwp_windows 应为三维张量，当前 shape={nwp_array.shape}")
    if power_array.ndim == 2:
        power_array = power_array[:, :, np.newaxis]
    if power_array.ndim != 3:
        raise ValueError(f"power_windows 应为三维张量，当前 shape={power_array.shape}")
    if nwp_array.shape[0] == 0:
        return np.zeros((EXTREME_SCENARIO_DESCRIPTOR_DIM,), dtype=np.float32)

    descriptor_rows = []
    for sample_index in range(nwp_array.shape[0]):
        descriptor_rows.append(
            build_extreme_scenario_descriptor(
                nwp_array[sample_index],
                power_array[sample_index],
            )
        )
    descriptor_matrix = np.stack(descriptor_rows, axis=0).astype(np.float32)
    descriptor_matrix = normalize_prototype_features_in_client(descriptor_matrix)
    return np.mean(descriptor_matrix, axis=0).astype(np.float32)


def compute_sample_count_reliability(sample_count):
    return float(math.log1p(max(0, int(sample_count))))


def compute_query_reliability(query_loss, tau=EXTREME_WEIGHT_TAU):
    if query_loss is None:
        return 1.0
    return float(math.exp(-float(tau) * float(query_loss)))


def compute_group_similarity_scale(prototype_list):
    if len(prototype_list) <= 1:
        return 1.0
    pairwise_distances = []
    for source_index in range(len(prototype_list)):
        for target_index in range(source_index + 1, len(prototype_list)):
            distance = np.linalg.norm(prototype_list[source_index] - prototype_list[target_index])
            pairwise_distances.append(float(distance))
    if not pairwise_distances:
        return 1.0
    return max(1.0, float(np.mean(pairwise_distances)))


def compute_scenario_similarity(source_prototype, target_prototype, sigma_c):
    source_vector = np.asarray(source_prototype, dtype=np.float32).reshape(-1)
    target_vector = np.asarray(target_prototype, dtype=np.float32).reshape(-1)
    distance_sq = float(np.sum((source_vector - target_vector) ** 2))
    sigma_sq = max(float(sigma_c) ** 2, 1e-8)
    return float(math.exp(-distance_sq / sigma_sq))


def compute_reliability_aware_weights(
    target_station_id,
    extreme_class_index,
    station_payloads,
    query_loss_by_station=None,
    tau=EXTREME_WEIGHT_TAU,
    lambda_weight=EXTREME_WEIGHT_LAMBDA,
    mu=EXTREME_WEIGHT_MU,
    nu=EXTREME_WEIGHT_NU,
):
    if query_loss_by_station is None:
        query_loss_by_station = {}

    eligible_station_ids = []
    prototype_list = []
    for station_id, payload in station_payloads.items():
        if extreme_class_index not in payload.get("valid_class_indices", []):
            continue
        eligible_station_ids.append(station_id)
        prototype_list.append(np.asarray(payload["extreme_class_prototypes"][extreme_class_index], dtype=np.float32))

    if not eligible_station_ids:
        return {}

    target_prototype = np.asarray(
        station_payloads[target_station_id]["extreme_class_prototypes"][extreme_class_index],
        dtype=np.float32,
    )
    sigma_c = compute_group_similarity_scale(prototype_list)

    weight_payload = {}
    total_score = 0.0
    for station_id in eligible_station_ids:
        station_payload = station_payloads[station_id]
        support_hour_count = int(station_payload["p_extre"][0, extreme_class_index].shape[0])
        support_window_count = support_hour_count // len_realp
        source_prototype = np.asarray(
            station_payload["extreme_class_prototypes"][extreme_class_index],
            dtype=np.float32,
        )

        m_k_c = compute_sample_count_reliability(support_window_count)
        q_k_c = compute_query_reliability(query_loss_by_station.get(station_id), tau=tau)
        sim_k_to_s_c = compute_scenario_similarity(source_prototype, target_prototype, sigma_c)
        a_k_to_s_c = float((m_k_c ** lambda_weight) * (q_k_c ** mu) * (sim_k_to_s_c ** nu))
        total_score += a_k_to_s_c
        weight_payload[station_id] = {
            "m_k_c": m_k_c,
            "q_k_c": q_k_c,
            "sim_k_to_s_c": sim_k_to_s_c,
            "a_k_to_s_c": a_k_to_s_c,
            "alpha_k_to_s_c": 0.0,
        }

    if total_score <= 0:
        uniform_alpha = 1.0 / len(weight_payload)
        for station_id in weight_payload:
            weight_payload[station_id]["alpha_k_to_s_c"] = uniform_alpha
        return weight_payload

    for station_id in weight_payload:
        weight_payload[station_id]["alpha_k_to_s_c"] = weight_payload[station_id]["a_k_to_s_c"] / total_score
    return weight_payload


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
log_stage_banner("阶段: data_preparation", ["准备所有场站的完整数据（元训练+极端天气+测试集）"])

# 准备所有客户端的常规天气数据（已在预训练中准备）
# ========== [联邦新增] 准备所有客户端的常规天气数据 ==========
if USE_FEDERATION:
    progress_log("\n准备联邦客户端数据（常规天气）...")
    clients_train_data = {}
    
    for station_id in station_ids:
        progress_log(f"  处理场站 {station_id}...")
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
        progress_log(f"    shape: {nwp_conven_1_st.shape} → {p_conven_1_st.shape}")
    
    clients_train_data = subsample_pretrain_conventional_data(
        clients_train_data,
        CONVENTIONAL_RATIO,
        subsample_seed_offset=CONVENTIONAL_SUBSAMPLE_SEED_OFFSET
    )
    progress_log(f"  总数据量: {sum([clients_train_data[s]['input'].shape[1] for s in station_ids])} 样本")
# ========== [联邦新增] 结束 ==========

# ========== [联邦修改] 准备所有场站的元训练、极端天气和测试数据 ==========
progress_log("\n准备所有场站的元训练和测试数据（3场站一视同仁）...")
all_stations_full_data = {}

for station_id in station_ids:
    progress_log(f"\n  场站 {station_id}:")
    
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
    
    # 测试集
    if (SEASONAL_PROTOCOL_ENABLED or YEARLY_PROTOCOL_ENABLED) and station_data[station_id]['p_test_00'] is not None:
        p_test_st = station_data[station_id]['p_test_00']
        nwp_test_st = station_data[station_id]['nwp_test_00']
        p_test_st_ = p_test_st
        nwp_test_st_ = np.empty([1,5],dtype=object)
        for i in range(np.size(nwp_test_st, axis=1)):
            nwp_test_st_[0, i] = nwp_test_st[:, i].reshape(-1, 1) / np.max(abs(P_nwp_st[:, i]), axis=0)
        for i_nwp in range(np.size(nwp_test_st_, axis=1)):
            if i_nwp == 0:
                test_input_c_st = nwp_test_st_[0, i_nwp].transpose(1, 0)
                test_input_c_st = test_input_c_st[:, :, np.newaxis]
            else:
                nwp_test_0 = nwp_test_st_[0, i_nwp].transpose(1, 0)
                test_input_c_st = np.concatenate((test_input_c_st, nwp_test_0[:, :, np.newaxis]), axis=2)
        test_target_p_st = p_test_st_.transpose(1, 0)
        test_target_p_st = test_target_p_st[:, :, np.newaxis]
        test_target_p_st = test_target_p_st.reshape(-1, len_realp, dem_realp)
        test_input_c_st = test_input_c_st.reshape(-1, len_realp, dem_realc)
    else:
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
    p_test_extre_st = np.empty([1,4],dtype=object)
    nwp_test_extre_st = np.empty([1,5],dtype=object)
    
    for i_class in range(4):
        p_extre_st[0,i_class] = station_data[station_id][f'p_extre_class{i_class+1}_00']
        test_power_class = station_data[station_id][f'p_test_extre_class{i_class+1}_00']
        p_test_extre_st[0, i_class] = test_power_class if test_power_class is not None else np.empty((0, 1))
    
    for i_nwp in range(5):
        nwp_extre_st[0,i_nwp] = np.empty([1,4],dtype=object)
        nwp_test_extre_st[0, i_nwp] = np.empty([1,4],dtype=object)
        for i_class in range(4):
            nwp_extre_st[0, i_nwp][0, i_class] = station_data[station_id][f'nwp_extre_class{i_class+1}_00'][0, i_nwp]
            test_nwp_class = station_data[station_id][f'nwp_test_extre_class{i_class+1}_00']
            if test_nwp_class is None:
                nwp_test_extre_st[0, i_nwp][0, i_class] = np.empty((0, 1))
            else:
                nwp_test_extre_st[0, i_nwp][0, i_class] = test_nwp_class[0, i_nwp]
    
    for i in range(np.size(nwp_extre_st,axis=1)):
        nwp_extre_st[0,i] = nwp_extre_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)
        nwp_test_extre_st[0,i] = nwp_test_extre_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)

    extreme_class_prototypes = {}
    for i_class in range(4):
        class_nwp_windows = []
        for i_nwp in range(np.size(nwp_extre_st, axis=1)):
            class_nwp_feature = nwp_extre_st[0, i_nwp][0, i_class]
            class_window_count = class_nwp_feature.shape[0] // len_realp
            feature_windows = class_nwp_feature[:class_window_count * len_realp].reshape(class_window_count, len_realp, 1)
            class_nwp_windows.append(feature_windows)
        if class_nwp_windows:
            class_nwp_tensor = np.concatenate(class_nwp_windows, axis=2)
        else:
            class_nwp_tensor = np.empty((0, len_realp, dem_realc), dtype=np.float32)
        class_power_data = p_extre_st[0, i_class]
        class_window_count = class_power_data.shape[0] // len_realp
        class_power_tensor = class_power_data[:class_window_count * len_realp].reshape(class_window_count, len_realp, 1)
        extreme_class_prototypes[i_class] = compute_extreme_class_prototype(
            class_nwp_tensor,
            class_power_tensor,
        )
    
    # 存储该场站的完整数据
    all_stations_full_data[station_id] = {
        'P_nwp': P_nwp_st,
        'test_input': test_input_c_st,
        'test_target': test_target_p_st,
        'p_conven_class': p_conven_class_st,
        'nwp_conven_class': nwp_conven_class_st,
        'p_extre': p_extre_st,
        'nwp_extre': nwp_extre_st,
        'p_test_extre': p_test_extre_st,
        'nwp_test_extre': nwp_test_extre_st,
        'extreme_class_prototypes': extreme_class_prototypes,
        'valid_class_indices': resolve_station_extreme_class_indices(station_id),
        'sampler_task_count': seasonal_protocol_metadata["client_map"][str(station_id)]["sampler_task_count"] if SEASONAL_PROTOCOL_ENABLED else None
    }
    progress_log(f"    测试集2023: {test_target_p_st.shape}")
    progress_log(f"    聚类类别: {np.size(p_conven_class_st, axis=1)}类")
    progress_log(f"    极端天气: 4类")

progress_log(f"\n✓ 所有场站数据准备完成！")

all_stations_full_data = subsample_meta_conventional_data(
    all_stations_full_data,
    CONVENTIONAL_RATIO,
    subsample_seed_offset=CONVENTIONAL_SUBSAMPLE_SEED_OFFSET
)
all_stations_full_data = apply_regime_missing_to_meta_data(
    all_stations_full_data,
    REGIME_MISSING_MODE
)
if USE_FEDERATION:
    clients_train_data = apply_regime_missing_to_pretrain_data(
        clients_train_data,
        all_stations_full_data,
        REGIME_MISSING_MODE
    )

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
    log_stage_banner(
        "阶段: local_pretrain",
        [
            f"station_id={station_id}",
            f"epochs={epoch1_pre}",
            f"log_interval={PRETRAIN_LOG_INTERVAL}",
        ],
    )

    local_model = model_fore(
        input_channel_fore=dem_realc,
        output_channel_fore=[128, 96, 64, 48, 32, 16, 8],
        mode='pre'
    ).to(device)
    local_optimizer = torch.optim.Adam(local_model.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999))
    convergence_record = initialize_convergence_record(
        stage_type="local_pretrain",
        stage_id=f"station{station_id}",
        total_epochs=epoch1_pre,
        patience=CONVERGENCE_PATIENCE_PRETRAIN,
    )

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
        update_convergence_record(convergence_record, i, loss_mse.item())

        if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
            progress_log(
                f"  [local_pretrain:station{station_id}] "
                f"[Epoch {i + 1}/{epoch1_pre}] [loss_mse: {loss_mse.item():.6f}]"
            )
            writer1.add_scalar(f"loss_penalty_pre_local_station{station_id}", loss_penalty.item(), i)
            writer2.add_scalar(f"loss_mse_pre_local_station{station_id}", loss_mse.item(), i)

    local_model.eval()
    local_state = clone_state_dict(local_model.state_dict())
    torch.save(local_state, save_path)
    register_convergence_record(convergence_record)
    progress_log(f"✓ 场站 {station_id} 本地 conventional pretrain 完成: {save_path}")
    return local_state


def server_aggregate_client_states(client_updates):
    weighted_states = [
        (client_update['state_dict'], client_update['aggregation_weight'])
        for client_update in client_updates
    ]
    return average_state_dicts(weighted_states)


def build_station_meta_tasks(station_id):
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
            'class_index': i_class,
            'class_size': num_samples,
            'nwp': nwp_conven_class_1,
            'p': p_conven_class_1,
        })

    return station_tasks


def resolve_local_meta_tasks_per_epoch(station_tasks, requested_tasks_per_epoch=META_TASKS_PER_EPOCH):
    if REGIME_MISSING_MODE == "class_dropout":
        return min(len(station_tasks), max(1, len(station_tasks) // 2))
    return min(requested_tasks_per_epoch, len(station_tasks))


def sample_task_indices_weighted_without_replacement(candidate_indices, candidate_weights, sample_count):
    # weighted random without replacement
    selected_indices = []
    remaining_indices = list(candidate_indices)
    remaining_weights = [float(max(weight, 1e-8)) for weight in candidate_weights]
    sample_count = min(sample_count, len(remaining_indices))

    for _ in range(sample_count):
        weight_sum = sum(remaining_weights)
        threshold = random.random() * weight_sum
        cumulative_weight = 0.0
        selected_position = len(remaining_indices) - 1
        for i_weight, weight in enumerate(remaining_weights):
            cumulative_weight += weight
            if cumulative_weight >= threshold:
                selected_position = i_weight
                break
        selected_indices.append(remaining_indices.pop(selected_position))
        remaining_weights.pop(selected_position)

    return selected_indices


def build_meta_batch_from_tasks(selected_tasks):
    for i_task, task in enumerate(selected_tasks):
        index_shot = random.sample(range(0, np.size(task['nwp'], axis=0)), META_SUPPORT_SHOTS + META_QUERY_SHOTS)
        train_input_support_ = task['nwp'][index_shot[0:META_SUPPORT_SHOTS], :, :]
        train_input_query_ = task['nwp'][index_shot[META_SUPPORT_SHOTS:META_SUPPORT_SHOTS + META_QUERY_SHOTS], :, :]
        train_target_support_ = task['p'][index_shot[0:META_SUPPORT_SHOTS], :, :]
        train_target_query_ = task['p'][index_shot[META_SUPPORT_SHOTS:META_SUPPORT_SHOTS + META_QUERY_SHOTS], :, :]
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


def sample_station_meta_batch_uniform(station_id, tasks_per_epoch=META_TASKS_PER_EPOCH):
    station_tasks = build_station_meta_tasks(station_id)
    tasks_per_epoch = resolve_local_meta_tasks_per_epoch(station_tasks, requested_tasks_per_epoch=tasks_per_epoch)
    selected_tasks = random.sample(station_tasks, tasks_per_epoch)
    return build_meta_batch_from_tasks(selected_tasks), [task['class_index'] for task in selected_tasks]


def sample_station_meta_batch_balanced(
    station_id,
    recent_selected_classes,
    tasks_per_epoch=META_TASKS_PER_EPOCH,
    coverage_window=PROPOSED_META_COVERAGE_WINDOW_FIXED
):
    station_tasks = build_station_meta_tasks(station_id)
    tasks_per_epoch = resolve_local_meta_tasks_per_epoch(station_tasks, requested_tasks_per_epoch=tasks_per_epoch)
    mean_class_size = np.mean([task['class_size'] for task in station_tasks])
    recent_window = recent_selected_classes[-coverage_window:] if coverage_window > 0 else []

    candidate_indices = []
    candidate_weights = []
    for task in station_tasks:
        exposure_c = sum(task['class_index'] in selected_class_set for selected_class_set in recent_window)
        coverage_bonus = 1.0 / (1.0 + exposure_c)
        size_bonus = np.sqrt(mean_class_size / max(task['class_size'], 1))
        candidate_indices.append(task['class_index'])
        candidate_weights.append(float(size_bonus * coverage_bonus))

    selected_class_indices = sample_task_indices_weighted_without_replacement(
        candidate_indices,
        candidate_weights,
        tasks_per_epoch
    )
    selected_tasks = [station_tasks[class_index] for class_index in selected_class_indices]
    return build_meta_batch_from_tasks(selected_tasks), selected_class_indices


## pre-train
if RUN_FEDERATED_PRETRAIN:
    log_stage_banner(
        "阶段: federated_pretrain",
        [
            f"clients={task_num}",
            f"stations={', '.join(station_ids)}",
            f"epochs={PRETRAIN_EPOCHS}",
            f"log_interval={PRETRAIN_LOG_INTERVAL}",
        ],
    )
elif YEARLY_PROTOCOL_ENABLED:
    log_stage_banner(
        "阶段: federated_pretrain_skipped",
        [
            "reason=yearly clean ablation uses common local_pretrain + common local_meta backbone",
            f"stations={', '.join(station_ids)}",
        ],
    )
else:
    log_stage_banner(
        "阶段: standalone_pretrain",
        [
            f"station_id={station_ids[0]}",
            f"epochs={PRETRAIN_EPOCHS}",
            f"log_interval={PRETRAIN_LOG_INTERVAL}",
        ],
    )

total_train_step=0
total_test_step=0
epoch1_pre = PRETRAIN_EPOCHS
writer1=SummaryWriter("./logs_train/loss1")
writer2=SummaryWriter("./logs_train/loss2")
start_time=time.time()

if RUN_FEDERATED_PRETRAIN:
    global_pretrain_state = clone_state_dict(model_fore_pre.state_dict())
    pretrain_convergence_record = initialize_convergence_record(
        stage_type="federated_pretrain",
        stage_id="global",
        total_epochs=epoch1_pre,
        patience=CONVERGENCE_PATIENCE_PRETRAIN,
    )
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
        update_convergence_record(pretrain_convergence_record, i, loss2_display)

        if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
            end_time = time.time()
            progress_log(
                f"  [federated_pretrain] [Epoch {i + 1}/{epoch1_pre}] "
                f"[loss_mse: {loss2_display:.6f}] [elapsed_sec: {end_time - start_time:.6f}]"
            )
            writer1.add_scalar("loss_mse_pre", loss1_display, i)
            writer2.add_scalar("loss_mse_pre", loss2_display, i)
elif YEARLY_PROTOCOL_ENABLED:
    pretrain_convergence_record = None
    progress_log("跳过联邦预训练；yearly clean ablation 不使用 federated pretrain 初始化。")
else:
    pretrain_convergence_record = initialize_convergence_record(
        stage_type="standalone_pretrain",
        stage_id="single_station",
        total_epochs=epoch1_pre,
        patience=CONVERGENCE_PATIENCE_PRETRAIN,
    )
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
        update_convergence_record(pretrain_convergence_record, i, loss2_display)

        if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
            end_time = time.time()
            progress_log(
                f"  [standalone_pretrain] [Epoch {i + 1}/{epoch1_pre}] "
                f"[loss_mse: {loss2_display:.6f}] [elapsed_sec: {end_time - start_time:.6f}]"
            )
            writer1.add_scalar("loss_mse_pre", loss1_display, i)
            writer2.add_scalar("loss_mse_pre", loss2_display, i)

if pretrain_convergence_record is not None:
    model_fore_pre.eval()
    torch.save(model_fore_pre.state_dict(), PRETRAIN_MODEL_PATH)
    register_convergence_record(pretrain_convergence_record)
    if RUN_FEDERATED_PRETRAIN:
        progress_log(f"\n✓ 联邦预训练完成: {PRETRAIN_MODEL_PATH}")
    else:
        progress_log(f"\n✓ 预训练完成: {PRETRAIN_MODEL_PATH}")


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
    shared_lr_scale=1.0,
    sampler_mode="uniform"
):
    """
    单场站本地元训练过程：
    - proposed: init_state_dict 为 federated pre-train 权重
    - meta_only: init_state_dict 为随机初始化
    """
    log_stage_banner(
        "阶段: local_meta",
        [
            f"station_id={station_id}",
            f"meta_tag={meta_tag}",
            f"epochs={epoch_train_task}",
            f"log_interval={META_LOG_INTERVAL}",
            f"use_cdrm={use_cdrm}",
            f"train_all_params={train_all_params}",
            f"disable_lwp={disable_lwp}",
            f"shared_anchor_beta={shared_anchor_beta}",
            f"shared_lr_scale={shared_lr_scale}",
            f"sampler_mode={sampler_mode}",
        ],
    )
    total_task_pool = np.size(all_stations_full_data[station_id]['p_conven_class'], axis=1)
    local_tasks_per_epoch = resolve_station_sampler_tasks_per_epoch(
        station_id,
        [None] * total_task_pool,
        requested_tasks_per_epoch=META_TASKS_PER_EPOCH
    )
    progress_log(f"  tasks_per_epoch={local_tasks_per_epoch}, station_task_pool={total_task_pool}")

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
    recent_selected_classes = []
    convergence_record = initialize_convergence_record(
        stage_type="local_meta",
        stage_id=meta_tag,
        total_epochs=epoch_train_task,
        patience=CONVERGENCE_PATIENCE_META,
    )

    for i_t in range(epoch_train_task):
        if sampler_mode == "balanced":
            (
                (Train_target_support, Train_input_support, Train_target_query, Train_input_query),
                selected_class_indices
            ) = sample_station_meta_batch_balanced(
                station_id,
                recent_selected_classes,
                tasks_per_epoch=local_tasks_per_epoch
            )
            recent_selected_classes.append(set(selected_class_indices))
        else:
            (
                (Train_target_support, Train_input_support, Train_target_query, Train_input_query),
                _
            ) = sample_station_meta_batch_uniform(
                station_id,
                tasks_per_epoch=local_tasks_per_epoch
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
        update_convergence_record(convergence_record, i_t, loss2_q.item())
        if should_log_epoch(i_t, epoch_train_task, interval=META_LOG_INTERVAL):
            progress_log(
                f"  [{meta_tag}] [Epoch {i_t + 1}/{epoch_train_task}] "
                f"[support_mse: {loss2.item():.6f}] [query_mse: {loss2_q.item():.6f}] "
                f"[support_anchor: {anchor_loss_support.item():.6f}] [query_anchor: {anchor_loss_q.item():.6f}]"
            )

    register_convergence_record(convergence_record)
    progress_log(f"✓ 场站 {station_id} 元训练完成: {query_model_path}")


if YEARLY_PROTOCOL_ENABLED:
    progress_log("跳过 proposed_station local meta；yearly clean ablation 统一使用 local_meta 作为 common midstream。")
else:
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
            shared_lr_scale=PROPOSED_META_SHARED_LR_SCALE,
            sampler_mode=PROPOSED_META_SAMPLER_MODE
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
        shared_lr_scale=1.0,
        sampler_mode="uniform"
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
            shared_lr_scale=1.0,
            sampler_mode="uniform"
        )


## test_task_support
few_shot_class_total = sum(len(resolve_station_extreme_class_indices(station_id)) for station_id in station_ids)
if YEARLY_PROTOCOL_ENABLED:
    few_shot_model_count = few_shot_class_total * len(get_yearly_extreme_baseline_specs(station_ids[0]))
else:
    few_shot_model_count = few_shot_class_total * (4 if TRAIN_META_ONLY_BASELINE else 3)
log_stage_banner(
    "阶段: few_shot",
    [
        f"few_shot_model_count={few_shot_model_count}",
        f"epochs={FEW_SHOT_EPOCHS}",
        f"log_interval={FEW_SHOT_LOG_INTERVAL}",
    ],
)

# ========== [联邦修改] 为所有场站的所有极端天气类别训练个性化模型 ==========
all_personalized_models = {}  # 存储所有个性化模型
yearly_extreme_weight_cache = {}

def build_station_extreme_support_tensors(station_id, class_index):
    nwp_extre_st = all_stations_full_data[station_id]['nwp_extre']
    p_extre_st = all_stations_full_data[station_id]['p_extre']

    feature_windows = []
    for i_nwp in range(np.size(nwp_extre_st, axis=1)):
        nwp_data = nwp_extre_st[0, i_nwp][0, class_index]
        num_samples = nwp_data.shape[0] // len_realp
        nwp_reshaped = nwp_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)
        feature_windows.append(nwp_reshaped)

    if feature_windows:
        nwp_extre_class = np.concatenate(feature_windows, axis=2)
    else:
        nwp_extre_class = np.empty((0, len_realp, dem_realc), dtype=np.float32)

    p_data = p_extre_st[0, class_index]
    num_samples = p_data.shape[0] // len_realp
    p_extre_class = p_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)
    return (
        torch.tensor(nwp_extre_class, dtype=torch.float32),
        torch.tensor(p_extre_class, dtype=torch.float32),
        int(num_samples),
    )


def run_extreme_client_few_shot_update(base_model_path, log_tag, model_label, test_input_tensor, test_target_tensor):
    model_fore_test_task_support.load_state_dict(torch.load(base_model_path))
    optimizer = torch.optim.Adam(
        model_fore_test_task_support.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999)
    )
    convergence_record = initialize_convergence_record(
        stage_type="few_shot",
        stage_id=f"{model_label}:{log_tag}",
        total_epochs=FEW_SHOT_EPOCHS,
        patience=CONVERGENCE_PATIENCE_FEW_SHOT,
    )

    test_input_device = test_input_tensor.to(device)
    test_target_device = test_target_tensor.to(device)
    final_loss = np.nan

    for i in range(FEW_SHOT_EPOCHS):
        model_fore_test_task_support.train()
        test_outputs_support = model_fore_test_task_support(test_input_device)
        loss2 = loss_fn_1(test_outputs_support, test_target_device)
        loss_en = loss2
        optimizer.zero_grad()
        loss_en.backward()
        optimizer.step()
        final_loss = float(loss2.item())
        update_convergence_record(convergence_record, i, loss2.item())

        if should_log_epoch(i, FEW_SHOT_EPOCHS, interval=FEW_SHOT_LOG_INTERVAL):
            progress_log(
                f"      [{model_label}] [Epoch {i+1}/{FEW_SHOT_EPOCHS}] "
                f"[loss_mse: {loss2.item():.6f}]"
            )
            writer1.add_scalar(f"loss_penalty_{log_tag}", 0.0, i)
            writer2.add_scalar(f"loss_mse_{log_tag}", loss2.item(), i)

    model_fore_test_task_support.eval()
    register_convergence_record(convergence_record)
    tuned_state = clone_state_dict(model_fore_test_task_support.state_dict())
    return {
        "state_dict": tuned_state,
        "final_loss": final_loss,
        "best_loss": convergence_record["best_loss"],
    }


def run_few_shot_adaptation(base_model_path, save_path, log_tag, model_label, test_input_tensor, test_target_tensor):
    """针对某个初始化模型执行一次 few-shot 适应并保存。"""
    update_payload = run_extreme_client_few_shot_update(
        base_model_path=base_model_path,
        log_tag=log_tag,
        model_label=model_label,
        test_input_tensor=test_input_tensor,
        test_target_tensor=test_target_tensor,
    )
    torch.save(update_payload["state_dict"], save_path)
    progress_log(f"    ✓ 保存({model_label}): {save_path}")
    return update_payload


def aggregate_extreme_client_states(
    client_update_payloads,
    aggregation_mode,
    target_station_id,
    extreme_class_index,
    station_payloads,
):
    if not client_update_payloads:
        raise ValueError("client_update_payloads 不能为空")
    if aggregation_mode == "uniform":
        uniform_weight = 1.0 / len(client_update_payloads)
        weight_payload = {
            payload["station_id"]: {"alpha_k_to_s_c": uniform_weight}
            for payload in client_update_payloads
        }
        weighted_states = [
            (payload["state_dict"], uniform_weight)
            for payload in client_update_payloads
        ]
        return average_state_dicts(weighted_states), weight_payload
    if aggregation_mode == "reliability_aware":
        query_loss_by_station = {
            payload["station_id"]: payload["final_loss"]
            for payload in client_update_payloads
        }
        weight_payload = compute_reliability_aware_weights(
            target_station_id=target_station_id,
            extreme_class_index=extreme_class_index,
            station_payloads=station_payloads,
            query_loss_by_station=query_loss_by_station,
        )
        weighted_states = [
            (
                payload["state_dict"],
                weight_payload[payload["station_id"]]["alpha_k_to_s_c"],
            )
            for payload in client_update_payloads
        ]
        return average_state_dicts(weighted_states), weight_payload
    raise ValueError(f"未知的 aggregation_mode: {aggregation_mode}")

for station_id in station_ids:
    progress_log(f"\n{'='*70}")
    progress_log(f"场站 {station_id} 的Few-shot适应")
    progress_log(f"{'='*70}")
    
    for i_class in resolve_station_extreme_class_indices(station_id):
        progress_log(f"\n  极端天气类别 {i_class+1}:")
        Test_input_support, Test_target_support, num_samples = build_station_extreme_support_tensors(station_id, i_class)

        progress_log(f"    样本数: {num_samples}")
        progress_log(
            f"    训练轮数: {FEW_SHOT_EPOCHS}, "
            f"few-shot loss={'CDRM+MSE' if FEW_SHOT_USE_CDRM else 'MSE'}"
        )

        if YEARLY_PROTOCOL_ENABLED:
            baseline_specs = get_yearly_extreme_baseline_specs(station_id)
            for baseline_spec in baseline_specs:
                baseline_model_name = baseline_spec["save_path_builder"](i_class)
                if baseline_spec["label"] == "LMT-new":
                    run_few_shot_adaptation(
                        base_model_path=baseline_spec["base_model_path"],
                        save_path=baseline_model_name,
                        log_tag=f"{baseline_spec['slug']}_station{station_id}_class{i_class}",
                        model_label=baseline_spec["label"],
                        test_input_tensor=Test_input_support,
                        test_target_tensor=Test_target_support
                    )
                else:
                    if baseline_spec["label"] == "Extreme-FedAvg":
                        aggregation_mode="uniform"
                    else:
                        aggregation_mode="reliability_aware"

                    eligible_station_ids = [
                        source_station_id
                        for source_station_id in station_ids
                        if i_class in resolve_station_extreme_class_indices(source_station_id)
                    ]
                    client_update_payloads = []
                    for source_station_id in eligible_station_ids:
                        source_input_support, source_target_support, source_num_samples = build_station_extreme_support_tensors(
                            source_station_id,
                            i_class,
                        )
                        if source_num_samples == 0:
                            continue
                        progress_log(
                            f"    [{baseline_spec['label']}] source_station={source_station_id} -> "
                            f"target_station={station_id}, samples={source_num_samples}"
                        )
                        client_update_payloads.append(
                            {
                                "station_id": source_station_id,
                                **run_extreme_client_few_shot_update(
                                    base_model_path=get_local_meta_model_path(source_station_id),
                                    log_tag=(
                                        f"{baseline_spec['slug']}_target{station_id}_"
                                        f"source{source_station_id}_class{i_class}"
                                    ),
                                    model_label=f"{baseline_spec['label']}:source{source_station_id}",
                                    test_input_tensor=source_input_support,
                                    test_target_tensor=source_target_support,
                                ),
                            }
                        )

                    aggregated_state, weight_payload = aggregate_extreme_client_states(
                        client_update_payloads=client_update_payloads,
                        aggregation_mode=aggregation_mode,
                        target_station_id=station_id,
                        extreme_class_index=i_class,
                        station_payloads=all_stations_full_data,
                    )
                    yearly_extreme_weight_cache[(baseline_spec["label"], station_id, i_class)] = weight_payload
                    progress_log(
                        f"    [{baseline_spec['label']}] 聚合权重: "
                        f"{json.dumps({sid: round(payload['alpha_k_to_s_c'], 4) for sid, payload in weight_payload.items()}, ensure_ascii=False)}"
                    )
                    torch.save(aggregated_state, baseline_model_name)
                    progress_log(f"    ✓ 保存({baseline_spec['label']}): {baseline_model_name}")
                all_personalized_models[baseline_spec["result_key_builder"](i_class)] = baseline_model_name
        else:
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
progress_log(f"\n✓ Few-shot训练完成，生成了 {len(all_personalized_models)} 个个性化模型")
# ========== [联邦修改] 保存所有场站的测试结果 ==========
log_stage_banner("阶段: eval", ["生成所有场站的测试预测结果"])

all_test_results = {}  # 存储所有场站的测试结果

for station_id in station_ids:
    progress_log(f"\n场站 {station_id} 预测:")
    
    # 获取该场站的测试数据
    Test_input_c_st = torch.tensor(all_stations_full_data[station_id]['test_input'], dtype=torch.float32)
    Test_target_p_st = all_stations_full_data[station_id]['test_target']
    
    all_test_results[station_id] = {
        'test_input': Test_input_c_st,
        'test_target': Test_target_p_st,
        'predictions': {},
        'valid_class_indices': resolve_station_extreme_class_indices(station_id),
    }
    
    # 预测：该场站的有效极端天气模型
    for i_class in resolve_station_extreme_class_indices(station_id):
        model_name = f"model_fore_station{station_id}_extreme{i_class}.pth"
        model_fore_test_task_query.load_state_dict(torch.load(model_name))
        
        with torch.no_grad():
            Test_input_device = Test_input_c_st.to(device)
            Test_output = model_fore_test_task_query(Test_input_device)
            test_output = Test_output.to(device0)
            test_output_np = np.array(test_output.reshape(-1,dem_realp))
            all_test_results[station_id]['predictions'][f'extreme_{i_class}'] = test_output_np
        
        progress_log(f"  ✓ 极端类别{i_class+1}")
    
    # 预测：元学习模型（辅助口径）
    if YEARLY_PROTOCOL_ENABLED:
        meta_model_path = get_local_meta_model_path(station_id)
    else:
        meta_model_path = get_meta_only_model_path(station_id) if TRAIN_META_ONLY_BASELINE else get_proposed_meta_model_path(station_id)
    model_fore_test_task_query.load_state_dict(torch.load(meta_model_path))
    with torch.no_grad():
        Test_input_device = Test_input_c_st.to(device)
        Test_output = model_fore_test_task_query(Test_input_device)
        test_output = Test_output.to(device0)
        test_output_np = np.array(test_output.reshape(-1,dem_realp))
        all_test_results[station_id]['predictions']['meta'] = test_output_np
    progress_log(f"  ✓ 元学习模型")
    
    # 预测：本地预训练模型
    model_fore_test_task_query.load_state_dict(torch.load(get_local_pretrain_model_path(station_id)))
    with torch.no_grad():
        Test_input_device = Test_input_c_st.to(device)
        Test_output = model_fore_test_task_query(Test_input_device)
        test_output = Test_output.to(device0)
        test_output_np = np.array(test_output.reshape(-1,dem_realp))
        all_test_results[station_id]['predictions']['local_pre'] = test_output_np
    progress_log(f"  ✓ 本地预训练模型")

    if RUN_FEDERATED_PRETRAIN:
        # 预测：联邦预训练模型（辅助口径，非主表）
        model_fore_test_task_query.load_state_dict(torch.load(PRETRAIN_MODEL_PATH))
        with torch.no_grad():
            Test_input_device = Test_input_c_st.to(device)
            Test_output = model_fore_test_task_query(Test_input_device)
            test_output = Test_output.to(device0)
            test_output_np = np.array(test_output.reshape(-1,dem_realp))
            all_test_results[station_id]['predictions']['fed_pre'] = test_output_np
        progress_log(f"  ✓ 联邦预训练模型")

# 保存所有结果
progress_log("\n保存所有场站测试结果...")
scio.savemat('all_stations_test_results.mat', {'all_test_results': all_test_results, 'Cap': Cap})
progress_log("✓ 已保存: all_stations_test_results.mat")

export_convergence_report(
    CONVERGENCE_REPORT_PATH,
    convergence_records,
    run_config={
        "use_federation": USE_FEDERATION,
        "run_federated_pretrain": RUN_FEDERATED_PRETRAIN,
        "yearly_protocol_enabled": YEARLY_PROTOCOL_ENABLED,
        "yearly_protocol_metadata_path": YEARLY_PROTOCOL_METADATA_PATH,
        "seasonal_protocol_enabled": SEASONAL_PROTOCOL_ENABLED,
        "seasonal_protocol_metadata_path": SEASONAL_PROTOCOL_METADATA_PATH,
        "station_ids": station_ids,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "proposed_meta_epochs": PROPOSED_META_EPOCHS,
        "meta_only_meta_epochs": META_ONLY_META_EPOCHS,
        "few_shot_epochs": FEW_SHOT_EPOCHS,
        "meta_support_shots": META_SUPPORT_SHOTS,
        "meta_query_shots": META_QUERY_SHOTS,
        "proposed_meta_sampler_mode": PROPOSED_META_SAMPLER_MODE,
        "conventional_ratio": CONVENTIONAL_RATIO,
        "regime_missing_mode": REGIME_MISSING_MODE,
        "enable_convergence_monitor": ENABLE_CONVERGENCE_MONITOR,
    }
)

progress_log("\n" + "="*70)
progress_log("✓✓✓ 训练和测试全部完成！")
if TRAIN_META_ONLY_BASELINE:
    progress_log(f"生成的模型: {len(all_personalized_models)}个个性化模型（Proposed+Local_Meta_Transfer+Transfer_Learning+Meta-only） + 3类元模型 + 联邦/本地预训练模型")
else:
    progress_log(f"生成的模型: {len(all_personalized_models)}个个性化模型（Proposed+Local_Meta_Transfer+Transfer_Learning） + 2类元模型 + 联邦/本地预训练模型")
progress_log("="*70)

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
