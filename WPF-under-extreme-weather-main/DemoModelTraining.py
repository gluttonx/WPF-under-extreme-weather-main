import os
import time
import math
import json
import numpy as np
import torch.nn as nn
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
import random
import model
from torch.nn.utils import weight_norm

# ========== [多场站运行开关] ==========
USE_FEDERATION = True  # True=多场站, False=单场站原方法
# 说明：设为False时完全退化为原始单场站元学习方法

# ========== [基线重置开关] ==========
USE_PSEUDO_FED = False  # False=去掉 shared pretrain/shared meta，恢复 local LMT 基线

# ========== 论文口径关键开关 ==========
TRAIN_META_ONLY_BASELINE = os.getenv("TRAIN_META_ONLY_BASELINE", "0") != "0"
SKIP_LOCAL_PRETRAIN = os.getenv("SKIP_LOCAL_PRETRAIN", "0") != "0"
SKIP_LOCAL_META = os.getenv("SKIP_LOCAL_META", "0") != "0"
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", ".")


def resolve_artifact_path(filename):
    if os.path.isabs(filename):
        return filename
    if ARTIFACT_DIR in ("", "."):
        return filename
    return os.path.join(ARTIFACT_DIR, filename)


MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", resolve_artifact_path("models") if ARTIFACT_DIR not in ("", ".") else ".")
LOGS_TRAIN_DIR = os.getenv("LOGS_TRAIN_DIR", resolve_artifact_path("logs_train"))
ALL_STATIONS_TEST_RESULTS_PATH = os.getenv(
    "ALL_STATIONS_TEST_RESULTS_PATH",
    resolve_artifact_path("all_stations_test_results.mat"),
)
FEW_SHOT_EPOCHS = int(os.getenv("FEW_SHOT_EPOCHS", "50"))             # 论文口径：每个极端天气 fine-tune 50 epochs
FEW_SHOT_USE_CDRM = False
FEW_SHOT_CDRM_WEIGHT = 5.0
# 联邦场景下按论文口径保持每轮总任务数 k*=5
META_TASKS_PER_EPOCH = 5
PRETRAIN_EPOCHS = int(os.getenv("PRETRAIN_EPOCHS", "35000"))
PROPOSED_META_EPOCHS = int(os.getenv("PROPOSED_META_EPOCHS", "30000"))
META_ONLY_META_EPOCHS = int(os.getenv("META_ONLY_META_EPOCHS", "30000"))
PRETRAIN_LOG_INTERVAL = int(os.getenv("PRETRAIN_LOG_INTERVAL", "100"))
META_LOG_INTERVAL = int(os.getenv("META_LOG_INTERVAL", "100"))
FEW_SHOT_LOG_INTERVAL = int(os.getenv("FEW_SHOT_LOG_INTERVAL", "1"))
ENABLE_CONVERGENCE_MONITOR = os.getenv("ENABLE_CONVERGENCE_MONITOR", "1") != "0"
CONVERGENCE_REPORT_PATH = os.getenv(
    "CONVERGENCE_REPORT_PATH",
    resolve_artifact_path("training_convergence_report.json"),
)
CONVERGENCE_MIN_DELTA = float(os.getenv("CONVERGENCE_MIN_DELTA", "1e-4"))
CONVERGENCE_MIN_EPOCHS = int(os.getenv("CONVERGENCE_MIN_EPOCHS", "5"))
CONVERGENCE_PATIENCE_PRETRAIN = int(os.getenv("CONVERGENCE_PATIENCE_PRETRAIN", "200"))
CONVERGENCE_PATIENCE_META = int(os.getenv("CONVERGENCE_PATIENCE_META", "100"))
CONVERGENCE_PATIENCE_FEW_SHOT = int(os.getenv("CONVERGENCE_PATIENCE_FEW_SHOT", "5"))
# 论文消融口径：Meta-only = 去掉 pre-training，其余训练机制保持一致
META_ONLY_USE_CDRM = True
META_ONLY_TRAIN_ALL_PARAMS = False
META_ONLY_DISABLE_LWP = False
EXTREME_ADAPT_RATIO = float(os.getenv("EXTREME_ADAPT_RATIO", "0.75"))
EXTREME_MIN_VAL_HORIZON = int(os.getenv("EXTREME_MIN_VAL_HORIZON", "4"))
EXTREME_SOURCE_QUALITY_KEEP_RATIO = float(os.getenv("EXTREME_SOURCE_QUALITY_KEEP_RATIO", "0.8"))
EXTREME_MIN_EFFECTIVE_WINDOWS = int(os.getenv("EXTREME_MIN_EFFECTIVE_WINDOWS", "1"))
EXTREME_SOURCE_BORROW_BUDGET_GAMMA = float(os.getenv("EXTREME_SOURCE_BORROW_BUDGET_GAMMA", "1.0"))
EXTREME_USEFULNESS_ADAPT_STEPS = int(os.getenv("EXTREME_USEFULNESS_ADAPT_STEPS", "1"))
EXTREME_WEIGHT_LAMBDA = float(os.getenv("EXTREME_WEIGHT_LAMBDA", "1.0"))
EXTREME_WEIGHT_MU = float(os.getenv("EXTREME_WEIGHT_MU", "1.0"))
EXTREME_WEIGHT_NU = float(os.getenv("EXTREME_WEIGHT_NU", "1.0"))
EXTREME_WEIGHT_BETA_SELF = float(os.getenv("EXTREME_WEIGHT_BETA_SELF", "0.5"))
EXTREME_WEIGHT_TAU_Q = float(os.getenv("EXTREME_WEIGHT_TAU_Q", "1.0"))
EXTREME_WEIGHT_TAU_T = float(os.getenv("EXTREME_WEIGHT_TAU_T", "1.0"))
EXTREME_TARGET_REFINEMENT_EPOCHS = int(
    os.getenv("EXTREME_TARGET_REFINEMENT_EPOCHS", str(max(1, FEW_SHOT_EPOCHS // 2)))
)
EXTREME_TARGET_ADAPT_MAX_WINDOWS = int(os.getenv("EXTREME_TARGET_ADAPT_MAX_WINDOWS", "0"))

# ========== [数据协议] 支持 1h/12点 legacy 与 2h/6点 yearly protocol ==========
YEARLY_PROTOCOL_ENABLED = os.getenv("YEARLY_PROTOCOL_ENABLED", "0") != "0"
SEASONAL_PROTOCOL_ENABLED = os.getenv("SEASONAL_PROTOCOL_ENABLED", "0") != "0"
DEFAULT_YEARLY_PROTOCOL_METADATA_PATH = "three_station_yearly_protocol_data/three_station_yearly_protocol_metadata.json"
YEARLY_PROTOCOL_METADATA_PATH = os.getenv("YEARLY_PROTOCOL_METADATA_PATH", DEFAULT_YEARLY_PROTOCOL_METADATA_PATH)


def load_yearly_protocol_metadata(metadata_path):
    if not metadata_path or not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


PROTOCOL_METADATA_PATH = os.getenv(
    "PROTOCOL_METADATA_PATH",
    YEARLY_PROTOCOL_METADATA_PATH if YEARLY_PROTOCOL_ENABLED else "",
)
yearly_protocol_metadata = load_yearly_protocol_metadata(PROTOCOL_METADATA_PATH)
PROTOCOL_NAME = os.getenv(
    "PROTOCOL_NAME",
    yearly_protocol_metadata.get("protocol_name", "legacy_1h_12point"),
)
PROTOCOL_DATA_DIR = os.getenv(
    "PROTOCOL_DATA_DIR",
    yearly_protocol_metadata.get("protocol_data_dir", ""),
)
LEN_REALP = int(os.getenv("LEN_REALP", str(yearly_protocol_metadata.get("len_realp", 12))))
POINTS_PER_DAY = int(os.getenv("POINTS_PER_DAY", str(yearly_protocol_metadata.get("points_per_day", 24))))
SAMPLE_INTERVAL_HOURS = int(
    os.getenv(
        "SAMPLE_INTERVAL_HOURS",
        str(yearly_protocol_metadata.get("sample_interval_hours", max(1, 24 // max(1, POINTS_PER_DAY)))),
    )
)
DOWNSAMPLE_OFFSET = int(os.getenv("DOWNSAMPLE_OFFSET", str(yearly_protocol_metadata.get("downsample_offset", 0))))
WINDOW_SPAN_HOURS = int(
    os.getenv(
        "WINDOW_SPAN_HOURS",
        str(yearly_protocol_metadata.get("window_span_hours", SAMPLE_INTERVAL_HOURS * LEN_REALP)),
    )
)
convergence_records = []


def resolve_station_mat_path(station_id):
    filename = f"{station_id}wf_4_train.mat"
    if PROTOCOL_DATA_DIR:
        candidate = os.path.join(PROTOCOL_DATA_DIR, filename)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Protocol mat missing: {candidate}")
        return candidate
    return filename


def resolve_station_ids():
    metadata_stations = yearly_protocol_metadata.get("stations", [])
    if USE_FEDERATION and metadata_stations:
        return [str(station["station_id"]) for station in metadata_stations]
    if USE_FEDERATION:
        return ['58', '59', '60']
    return ['58']


def resolve_model_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(MODEL_OUTPUT_DIR, filename)


def print_protocol_banner():
    progress_log("=" * 70)
    progress_log("数据协议配置")
    progress_log("=" * 70)
    progress_log(f"  protocol_name: {PROTOCOL_NAME}")
    progress_log(f"  protocol_data_dir: {PROTOCOL_DATA_DIR or '(legacy root mats)'}")
    progress_log(f"  protocol_metadata_path: {PROTOCOL_METADATA_PATH or '(none)'}")
    progress_log(f"  yearly_protocol_enabled: {YEARLY_PROTOCOL_ENABLED}")
    progress_log(f"  seasonal_protocol_enabled: {SEASONAL_PROTOCOL_ENABLED}")
    progress_log(f"  sample_interval_hours: {SAMPLE_INTERVAL_HOURS}")
    progress_log(f"  downsample_offset: {DOWNSAMPLE_OFFSET}")
    progress_log(f"  len_realp: {LEN_REALP}")
    progress_log(f"  points_per_day: {POINTS_PER_DAY}")
    progress_log(f"  window_span_hours: {WINDOW_SPAN_HOURS}")
    progress_log(f"  artifact_dir: {ARTIFACT_DIR}")
    progress_log(f"  model_output_dir: {MODEL_OUTPUT_DIR}")
    progress_log(f"  logs_train_dir: {LOGS_TRAIN_DIR}")


def progress_log(message=""):
    print(message, flush=True)


def should_log_epoch(epoch_index, total_epochs, interval, warmup_epochs=10):
    current_epoch = int(epoch_index) + 1
    if current_epoch <= max(1, int(warmup_epochs)):
        return True
    if current_epoch >= int(total_epochs):
        return True
    return current_epoch % max(1, int(interval)) == 0


def initialize_convergence_record(stage_type, stage_id, total_epochs, patience):
    return {
        "stage_type": stage_type,
        "stage_id": stage_id,
        "total_epochs": int(total_epochs),
        "patience": int(patience),
        "min_delta": float(CONVERGENCE_MIN_DELTA),
        "min_epochs": int(CONVERGENCE_MIN_EPOCHS),
        "converged": False,
        "convergence_epoch": None,
        "best_epoch": None,
        "best_loss": None,
        "final_loss": None,
        "_last_improved_epoch": None,
        "_last_announced_convergence_epoch": None,
    }


def format_convergence_loss(loss_value):
    return "nan" if loss_value is None else f"{float(loss_value):.6f}"


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
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    report_payload = {
        "generated_at_unix": float(time.time()),
        "run_config": run_config,
        "records": records,
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report_payload, report_file, indent=2, ensure_ascii=False)
    progress_log(f"✓ 收敛检测报告已保存: {report_path}")

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
os.makedirs(ARTIFACT_DIR, exist_ok=True) if ARTIFACT_DIR not in ("", ".") else None
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_TRAIN_DIR, exist_ok=True)
print_protocol_banner()

# ========== [联邦修改] 多场站数据加载 ==========
station_ids = resolve_station_ids()
if USE_FEDERATION:
    print("="*70)
    print(f"联邦模式：加载{len(station_ids)}个场站数据（{', '.join(station_ids)}）")
    print("="*70)
else:
    print("="*70)
    print(f"单场站模式：加载场站{station_ids[0]}数据（原方法）")
    print("="*70)

# [联邦修改] 循环加载所有场站数据
station_data = {}
for station_id in station_ids:
    print(f"  加载 {resolve_station_mat_path(station_id)}...")
    wf_1 = scio.loadmat(resolve_station_mat_path(station_id))
    
    # [原代码保留] 数据提取逻辑完全不变
    station_data[station_id] = {
        'p': wf_1['p_1h'],
        'p_conven_00': wf_1['p_conven'],
        'p_conven_class_00': wf_1['p_conven_class'],
        'p_extre_class1_00': wf_1['p_extre_class1'],
        'p_extre_class2_00': wf_1['p_extre_class2'],
        'p_extre_class3_00': wf_1['p_extre_class3'],
        'p_extre_class4_00': wf_1['p_extre_class4'],
        'p_test_00': wf_1.get('p_test'),
        'nwp': wf_1['nwp_1h'],
        'nwp_test_00': wf_1.get('nwp_test'),
        'nwp_conven_00': wf_1['nwp_conven_'],
        'nwp_conven_class_00': wf_1['nwp_conven_class_'],
        'nwp_extre_class1_00': wf_1['nwp_extre_class1_'],
        'nwp_extre_class2_00': wf_1['nwp_extre_class2_'],
        'nwp_extre_class3_00': wf_1['nwp_extre_class3_'],
        'nwp_extre_class4_00': wf_1['nwp_extre_class4_'],
        'p_test_extre_class1_00': wf_1.get('p_test_extre_class1', wf_1['p_extre_class1']),
        'p_test_extre_class2_00': wf_1.get('p_test_extre_class2', wf_1['p_extre_class2']),
        'p_test_extre_class3_00': wf_1.get('p_test_extre_class3', wf_1['p_extre_class3']),
        'p_test_extre_class4_00': wf_1.get('p_test_extre_class4', wf_1['p_extre_class4']),
        'nwp_test_extre_class1_00': wf_1.get('nwp_test_extre_class1_', wf_1['nwp_extre_class1_']),
        'nwp_test_extre_class2_00': wf_1.get('nwp_test_extre_class2_', wf_1['nwp_extre_class2_']),
        'nwp_test_extre_class3_00': wf_1.get('nwp_test_extre_class3_', wf_1['nwp_extre_class3_']),
        'nwp_test_extre_class4_00': wf_1.get('nwp_test_extre_class4_', wf_1['nwp_extre_class4_'])
    }

# ========== [联邦修改] 删除"主场站"概念，所有场站一视同仁 ==========
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

# 模型文件路径（兼容保留）
PRETRAIN_MODEL_PATH = resolve_model_path("model_fore_pre_federated.pth" if USE_FEDERATION else "model_fore_pre.pth")
PROPOSED_SUPPORT_MODEL_PATH = resolve_model_path("model_fore_train_task_support_proposed.pth")
PROPOSED_META_MODEL_PATH = resolve_model_path("model_fore_train_task_query_proposed.pth")
META_ONLY_SUPPORT_MODEL_PATH = resolve_model_path("model_fore_train_task_support_meta_only.pth")
META_ONLY_MODEL_PATH = resolve_model_path("model_fore_train_task_query_meta_only.pth")


def get_local_pretrain_model_path(station_id):
    return resolve_model_path(f"model_fore_pre_station{station_id}_local.pth")


def get_local_meta_support_model_path(station_id):
    return resolve_model_path(f"model_fore_train_task_support_local_meta_station{station_id}.pth")


def get_local_meta_model_path(station_id):
    return resolve_model_path(f"model_fore_train_task_query_local_meta_station{station_id}.pth")


def get_local_meta_only_support_model_path(station_id):
    return resolve_model_path(f"model_fore_train_task_support_meta_only_station{station_id}.pth")


def get_local_meta_only_model_path(station_id):
    return resolve_model_path(f"model_fore_train_task_query_meta_only_station{station_id}.pth")


def get_lmt_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}.pth")


def get_meta_only_extreme_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_meta_only.pth")


def get_extreme_fedavg_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_extreme_fedavg.pth")


def get_proposed_a_model_path(station_id, class_idx):
    return resolve_model_path(f"model_fore_station{station_id}_extreme{class_idx}_proposed_a.pth")


# Define Parameters
dem_realp=1
len_realp = LEN_REALP
Cap=50  # 总装机容量 (MW)
m=365
d = POINTS_PER_DAY
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
print(f"\n准备所有场站的元训练和测试数据（{len(station_ids)}场站一视同仁）...")
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
    
    # 测试集（2023年）：yearly protocol 直接使用 builder 输出的 p_test/nwp_test。
    if (
        (SEASONAL_PROTOCOL_ENABLED or YEARLY_PROTOCOL_ENABLED)
        and station_data[station_id]['p_test_00'] is not None
        and station_data[station_id]['nwp_test_00'] is not None
    ):
        test_target_p_st = station_data[station_id]['p_test_00']
        test_target_p_st = test_target_p_st.reshape(-1,len_realp,dem_realp)
        test_nwp_st = station_data[station_id]['nwp_test_00']
        test_input_c_st = (test_nwp_st/np.max(abs(P_nwp_st),axis=0)).reshape(-1,len_realp,dem_realc)
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
        p_test_extre_st[0,i_class] = station_data[station_id][f'p_test_extre_class{i_class+1}_00']
    
    for i_nwp in range(5):
        nwp_extre_st[0,i_nwp] = np.empty([1,4],dtype=object)
        nwp_test_extre_st[0,i_nwp] = np.empty([1,4],dtype=object)
        for i_class in range(4):
            nwp_extre_st[0, i_nwp][0, i_class] = station_data[station_id][f'nwp_extre_class{i_class+1}_00'][0, i_nwp]
            nwp_test_extre_st[0, i_nwp][0, i_class] = station_data[station_id][f'nwp_test_extre_class{i_class+1}_00'][0, i_nwp]
    
    for i in range(np.size(nwp_extre_st,axis=1)):
        nwp_extre_st[0,i] = nwp_extre_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)
        nwp_test_extre_st[0,i] = nwp_test_extre_st[0,i]/np.max(abs(P_nwp_st[:,i]),axis=0)

    if (SEASONAL_PROTOCOL_ENABLED or YEARLY_PROTOCOL_ENABLED):
        p_test_extre_source = p_test_extre_st
        nwp_test_extre_source = nwp_test_extre_st
    else:
        p_test_extre_source = p_extre_st
        nwp_test_extre_source = nwp_extre_st
    
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
        'nwp_test_extre': nwp_test_extre_st
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
pretrain_random_init_state = copy.deepcopy(model_fore_pre.state_dict())
meta_only_random_init_state = copy.deepcopy(model_fore_train_task_query.state_dict())


## Define loss
loss_fn_1=nn.MSELoss()
def penalty(logits, y):
    scale = torch.tensor(1., device=device).requires_grad_()
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
if USE_FEDERATION and USE_PSEUDO_FED:
    print("#########################################################################——————————联邦预训练（Federation Pre-train）——————————############################################################")
    print(f"客户端数量: {task_num}, 场站: {', '.join(station_ids)}")
elif USE_FEDERATION:
    print("#########################################################################——————————本地预训练（Local Pre-train）——————————############################################################")
    print(f"场站数量: {task_num}, 场站: {', '.join(station_ids)}")
else:
    print("#########################################################################——————————预训练（Pre-train）——————————############################################################")

total_train_step = 0
total_test_step = 0
epoch1_pre = PRETRAIN_EPOCHS
writer1 = SummaryWriter(os.path.join(LOGS_TRAIN_DIR, "loss1"))
writer2 = SummaryWriter(os.path.join(LOGS_TRAIN_DIR, "loss2"))


def get_pretrain_penalty_weight(epoch_idx):
    if epoch_idx < 10000:
        return 0
    if epoch_idx < 20000:
        return 1
    if epoch_idx < 30000:
        return 5
    return 10


def run_single_pretrain_epoch(train_input, train_target, optimizer, penalty_weight):
    model_fore_pre.train()
    train_input = train_input.to(device)
    train_target = train_target.to(device)
    train_outputs_pre = model_fore_pre(train_input)
    loss1 = penalty(train_outputs_pre, train_target)
    loss2 = loss_fn_1(train_outputs_pre, train_target)
    loss_en = penalty_weight * loss1 + loss2
    optimizer.zero_grad()
    loss_en.backward()
    optimizer.step()
    return loss1.item(), loss2.item()


def run_federated_pretraining():
    start_time = time.time()
    pretrain_convergence_record = initialize_convergence_record(
        stage_type="federated_pretrain",
        stage_id="all_stations",
        total_epochs=epoch1_pre,
        patience=CONVERGENCE_PATIENCE_PRETRAIN,
    )
    for i in range(epoch1_pre):
        penalty_weight = get_pretrain_penalty_weight(i)
        model_fore_pre.train()
        optimizer_fore_pre.zero_grad()
        total_loss1 = 0.0
        total_loss2 = 0.0

        for station_id in station_ids:
            train_target = clients_train_tensor[station_id]['target'].to(device)
            train_input = clients_train_tensor[station_id]['input'].to(device)
            train_outputs_pre = model_fore_pre(train_input)
            loss1 = penalty(train_outputs_pre, train_target)
            loss2 = loss_fn_1(train_outputs_pre, train_target)
            loss_en = penalty_weight * loss1 + loss2
            (loss_en / task_num).backward()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

        optimizer_fore_pre.step()
        update_convergence_record(pretrain_convergence_record, i, total_loss2 / task_num)

        if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
            end_time = time.time()
            loss1_display = total_loss1 / task_num
            loss2_display = total_loss2 / task_num
            print(end_time - start_time)
            print("[Epoch %d/%d] [loss_mse: %f] " % (i, epoch1_pre, loss2_display))
            writer1.add_scalar("loss_mse_pre", loss1_display, i)
            writer2.add_scalar("loss_mse_pre", loss2_display, i)

    register_convergence_record(pretrain_convergence_record)
    model_fore_pre.eval()
    torch.save(model_fore_pre.state_dict(), PRETRAIN_MODEL_PATH)
    print(f"\n✓ 联邦预训练完成: {PRETRAIN_MODEL_PATH}")


def run_local_pretraining():
    for station_id in station_ids:
        print("\n" + "=" * 70)
        print(f"开始本地预训练: station {station_id}")
        print("=" * 70)
        save_path = get_local_pretrain_model_path(station_id)
        if SKIP_LOCAL_PRETRAIN:
            if not os.path.exists(save_path):
                raise FileNotFoundError(
                    f"SKIP_LOCAL_PRETRAIN=1 但未找到已有 checkpoint: {save_path}"
                )
            progress_log(f"✓ 跳过本地预训练，复用已有 checkpoint: {save_path}")
            continue

        model_fore_pre.load_state_dict(copy.deepcopy(pretrain_random_init_state))
        optimizer_local = torch.optim.Adam(
            model_fore_pre.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999)
        )
        start_time = time.time()
        pretrain_convergence_record = initialize_convergence_record(
            stage_type="local_pretrain",
            stage_id=station_id,
            total_epochs=epoch1_pre,
            patience=CONVERGENCE_PATIENCE_PRETRAIN,
        )
        train_input = clients_train_tensor[station_id]['input']
        train_target = clients_train_tensor[station_id]['target']

        for i in range(epoch1_pre):
            penalty_weight = get_pretrain_penalty_weight(i)
            loss1_display, loss2_display = run_single_pretrain_epoch(
                train_input,
                train_target,
                optimizer_local,
                penalty_weight,
            )
            update_convergence_record(pretrain_convergence_record, i, loss2_display)

            if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
                end_time = time.time()
                print(end_time - start_time)
                print(
                    f"[station {station_id}] [Epoch {i}/{epoch1_pre}] [loss_mse: {loss2_display:.6f}] "
                )
                writer1.add_scalar(f"loss_mse_pre_station{station_id}", loss1_display, i)
                writer2.add_scalar(f"loss_mse_pre_station{station_id}", loss2_display, i)

        register_convergence_record(pretrain_convergence_record)
        model_fore_pre.eval()
        torch.save(model_fore_pre.state_dict(), save_path)
        print(f"✓ 本地预训练完成: {save_path}")


if USE_FEDERATION and USE_PSEUDO_FED:
    run_federated_pretraining()
elif USE_FEDERATION:
    run_local_pretraining()
else:
    start_time = time.time()
    pretrain_convergence_record = initialize_convergence_record(
        stage_type="local_pretrain",
        stage_id="single_station",
        total_epochs=epoch1_pre,
        patience=CONVERGENCE_PATIENCE_PRETRAIN,
    )
    for i in range(epoch1_pre):
        penalty_weight = get_pretrain_penalty_weight(i)
        loss1_display, loss2_display = run_single_pretrain_epoch(
            Train_input_c,
            Train_target_p,
            optimizer_fore_pre,
            penalty_weight,
        )
        update_convergence_record(pretrain_convergence_record, i, loss2_display)

        if should_log_epoch(i, epoch1_pre, interval=PRETRAIN_LOG_INTERVAL):
            end_time = time.time()
            print(end_time - start_time)
            print("[Epoch %d/%d] [loss_mse: %f] " % (i, epoch1_pre, loss2_display))
            writer1.add_scalar("loss_mse_pre", loss1_display, i)
            writer2.add_scalar("loss_mse_pre", loss2_display, i)

    register_convergence_record(pretrain_convergence_record)
    model_fore_pre.eval()
    torch.save(model_fore_pre.state_dict(), PRETRAIN_MODEL_PATH)
    print(f"\n✓ 预训练完成: {PRETRAIN_MODEL_PATH}")


def sample_meta_batch(sample_station_ids=None):
    """按给定场站范围采样 meta 任务。"""
    if sample_station_ids is None:
        sample_station_ids = station_ids

    task_pool = []
    for station_id in sample_station_ids:
        p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']
        total_station_classes = np.size(p_conven_class_st, axis=1)
        for i_class in range(total_station_classes):
            task_pool.append((station_id, i_class))

    tasks_per_epoch = min(META_TASKS_PER_EPOCH, len(task_pool))
    sampled_tasks = random.sample(task_pool, tasks_per_epoch)

    selected_tasks = []
    for station_id, i_class in sampled_tasks:
        nwp_conven_class_st = all_stations_full_data[station_id]['nwp_conven_class']
        p_conven_class_st = all_stations_full_data[station_id]['p_conven_class']

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

        selected_tasks.append({
            'nwp': nwp_conven_class_1,
            'p': p_conven_class_1
        })

    train_input_dataset = [task['nwp'] for task in selected_tasks]
    train_target_dataset = [task['p'] for task in selected_tasks]
    num_tasks = len(selected_tasks)

    for i_task in range(num_tasks):
        index_shot = random.sample(range(0, np.size(train_input_dataset[i_task], axis=0)), 20)
        train_input_support_ = train_input_dataset[i_task][index_shot[0:10], :, :]
        train_input_query_ = train_input_dataset[i_task][index_shot[10:20], :, :]
        train_target_support_ = train_target_dataset[i_task][index_shot[0:10], :, :]
        train_target_query_ = train_target_dataset[i_task][index_shot[10:20], :, :]
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


def run_meta_training(
    meta_tag,
    init_state_dict,
    support_model_path,
    query_model_path,
    epoch_train_task=70000,
    use_cdrm=True,
    train_all_params=False,
    disable_lwp=False,
    sample_station_ids=None
):
    """
    单次元训练过程：
    - proposed: init_state_dict 为 pre-train 权重（CDRM + LWP 轻量更新）
    - meta_only: init_state_dict 为随机初始化（传统基线：可关闭CDRM、全参数更新）
    """
    print("\n" + "=" * 70)
    print(f"开始元训练: {meta_tag}")
    print(f"  use_cdrm={use_cdrm}, train_all_params={train_all_params}, disable_lwp={disable_lwp}")
    if sample_station_ids is None:
        sample_station_ids = station_ids
    total_task_pool = sum(np.size(all_stations_full_data[s]['p_conven_class'], axis=1) for s in sample_station_ids)
    print(
        f"  tasks_per_epoch={META_TASKS_PER_EPOCH}, task_pool={total_task_pool} "
        f"({len(sample_station_ids)} stations: {', '.join(sample_station_ids)})"
    )
    print("=" * 70)

    support_params = get_meta_trainable_params(
        model_fore_train_task_support,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp
    )
    query_params = get_meta_trainable_params(
        model_fore_train_task_query,
        train_all_params=train_all_params,
        disable_lwp=disable_lwp
    )

    optimizer_support = torch.optim.Adam(support_params, lr=0.0002, betas=(0.5, 0.999))
    optimizer_query = torch.optim.Adam(query_params, lr=0.0002, betas=(0.5, 0.999))
    if meta_tag.startswith("local_meta_station"):
        stage_type = "local_meta"
        convergence_record = initialize_convergence_record(
            stage_type="local_meta",
            stage_id=meta_tag,
            total_epochs=epoch_train_task,
            patience=CONVERGENCE_PATIENCE_META,
        )
    else:
        stage_type = meta_tag
        convergence_record = initialize_convergence_record(
            stage_type=stage_type,
            stage_id=meta_tag,
            total_epochs=epoch_train_task,
            patience=CONVERGENCE_PATIENCE_META,
        )

    for i_t in range(epoch_train_task):
        Train_target_support, Train_input_support, Train_target_query, Train_input_query = sample_meta_batch(
            sample_station_ids=sample_station_ids
        )

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
        optimizer_support.zero_grad()
        loss_en.backward()
        optimizer_support.step()
        model_fore_train_task_support.eval()
        support_state = copy.deepcopy(model_fore_train_task_support.state_dict())
        torch.save(support_state, support_model_path)

        writer1.add_scalar(f"loss_penalty_train_task_support_{meta_tag}", loss1.item(), i_t)
        writer2.add_scalar(f"loss_mse_train_task_support_{meta_tag}", loss2.item(), i_t)

        print(
            "[##################################################################"
            f"——{meta_tag}:train_task_query_Epoch {i_t}/{epoch_train_task}——"
            "############################################################]"
        )

        # 严格 support->query 链路：query 以本轮 support 更新后的参数为起点
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
        optimizer_query.zero_grad()
        loss_en_q.backward()
        optimizer_query.step()
        model_fore_train_task_query.eval()
        torch.save(model_fore_train_task_query.state_dict(), query_model_path)

        writer1.add_scalar(f"loss_penalty_train_task_query_{meta_tag}", loss1_q.item(), i_t)
        writer2.add_scalar(f"loss_mse_train_task_query_{meta_tag}", loss2_q.item(), i_t)
        update_convergence_record(convergence_record, i_t, loss2_q.item())

        if should_log_epoch(i_t, epoch_train_task, interval=META_LOG_INTERVAL):
            progress_log(
                f"  收敛追踪[{stage_type}:{meta_tag}] "
                f"epoch={i_t + 1}/{epoch_train_task} "
                f"query_mse={loss2_q.item():.6f}"
            )

    register_convergence_record(convergence_record)
    print(f"✓ 元训练完成: {query_model_path}")


def run_shared_meta_training():
    proposed_init_state = torch.load(PRETRAIN_MODEL_PATH)
    run_meta_training(
        meta_tag="proposed",
        init_state_dict=proposed_init_state,
        support_model_path=PROPOSED_SUPPORT_MODEL_PATH,
        query_model_path=PROPOSED_META_MODEL_PATH,
        epoch_train_task=PROPOSED_META_EPOCHS,
        use_cdrm=True,
        train_all_params=False,
        disable_lwp=False
    )

    if TRAIN_META_ONLY_BASELINE:
        run_meta_training(
            meta_tag="meta_only",
            init_state_dict=meta_only_random_init_state,
            support_model_path=META_ONLY_SUPPORT_MODEL_PATH,
            query_model_path=META_ONLY_MODEL_PATH,
            epoch_train_task=META_ONLY_META_EPOCHS,
            use_cdrm=META_ONLY_USE_CDRM,
            train_all_params=META_ONLY_TRAIN_ALL_PARAMS,
            disable_lwp=META_ONLY_DISABLE_LWP
        )


def run_local_meta_training():
    for station_id in station_ids:
        print("\n" + "=" * 70)
        print(f"开始本地元训练: station {station_id}")
        print("=" * 70)

        local_pretrain_path = get_local_pretrain_model_path(station_id)
        local_meta_path = get_local_meta_model_path(station_id)
        if SKIP_LOCAL_META:
            if not os.path.exists(local_meta_path):
                raise FileNotFoundError(
                    f"SKIP_LOCAL_META=1 但未找到已有 checkpoint: {local_meta_path}"
                )
            progress_log(f"✓ 跳过本地元训练，复用已有 checkpoint: {local_meta_path}")
            continue
        run_meta_training(
            meta_tag=f"local_meta_station{station_id}",
            init_state_dict=torch.load(local_pretrain_path),
            support_model_path=get_local_meta_support_model_path(station_id),
            query_model_path=local_meta_path,
            epoch_train_task=PROPOSED_META_EPOCHS,
            use_cdrm=True,
            train_all_params=False,
            disable_lwp=False,
            sample_station_ids=[station_id]
        )

        if TRAIN_META_ONLY_BASELINE:
            run_meta_training(
                meta_tag=f"meta_only_station{station_id}",
                init_state_dict=meta_only_random_init_state,
                support_model_path=get_local_meta_only_support_model_path(station_id),
                query_model_path=get_local_meta_only_model_path(station_id),
                epoch_train_task=META_ONLY_META_EPOCHS,
                use_cdrm=META_ONLY_USE_CDRM,
                train_all_params=META_ONLY_TRAIN_ALL_PARAMS,
                disable_lwp=META_ONLY_DISABLE_LWP,
                sample_station_ids=[station_id]
            )


if USE_FEDERATION and not USE_PSEUDO_FED:
    run_local_meta_training()
else:
    run_shared_meta_training()


## test_task_support
few_shot_model_count = len(station_ids) * 4 * (4 if TRAIN_META_ONLY_BASELINE else 3)
print(f"##################################################################——————————test_task_support（Few-shot适应：共{few_shot_model_count}个模型）——————————############################################################")

# ========== [联邦修改] 为所有场站的所有极端天气类别训练个性化模型 ==========
all_personalized_models = {}  # 存储所有个性化模型

def extract_extreme_windows_for_station_class(station_id, class_idx):
    nwp_extre_st = all_stations_full_data[station_id]['nwp_extre']
    p_extre_st = all_stations_full_data[station_id]['p_extre']

    nwp_windows = None
    num_samples = 0
    for i_nwp in range(np.size(nwp_extre_st, axis=1)):
        nwp_data = nwp_extre_st[0, i_nwp][0, class_idx]
        num_samples = nwp_data.shape[0] // len_realp
        nwp_reshaped = nwp_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)
        if i_nwp == 0:
            nwp_windows = nwp_reshaped
        else:
            nwp_windows = np.concatenate((nwp_windows, nwp_reshaped), axis=2)

    p_data = p_extre_st[0, class_idx]
    num_samples = p_data.shape[0] // len_realp
    power_windows = p_data[:num_samples * len_realp].reshape(num_samples, len_realp, 1)

    if nwp_windows is None:
        nwp_windows = np.empty((0, len_realp, dem_realc), dtype=np.float32)
    return nwp_windows.astype(np.float32), power_windows.astype(np.float32)


def split_extreme_adapt_val(nwp_windows, power_windows, adapt_ratio=EXTREME_ADAPT_RATIO):
    nwp_array = np.asarray(nwp_windows, dtype=np.float32)
    power_array = np.asarray(power_windows, dtype=np.float32)

    if nwp_array.shape[0] == 0:
        empty_nwp = np.empty((0, len_realp, dem_realc), dtype=np.float32)
        empty_power = np.empty((0, len_realp, 1), dtype=np.float32)
        return {
            "adapt_nwp": empty_nwp,
            "adapt_power": empty_power,
            "val_nwp": empty_nwp,
            "val_power": empty_power,
            "full_window_count": 0,
        }

    if nwp_array.shape[0] == 1:
        horizon = int(nwp_array.shape[1])
        adapt_horizon = int(round(float(adapt_ratio) * horizon))
        adapt_horizon = min(max(1, adapt_horizon), max(1, horizon - EXTREME_MIN_VAL_HORIZON))
        if horizon - adapt_horizon <= 0:
            adapt_horizon = max(1, horizon - 1)
        return {
            "adapt_nwp": nwp_array[:, :adapt_horizon, :],
            "adapt_power": power_array[:, :adapt_horizon, :],
            "val_nwp": nwp_array[:, adapt_horizon:, :],
            "val_power": power_array[:, adapt_horizon:, :],
            "full_window_count": 1,
        }

    val_window_count = min(max(1, int(round((1.0 - float(adapt_ratio)) * nwp_array.shape[0]))), nwp_array.shape[0] - 1)
    adapt_window_count = nwp_array.shape[0] - val_window_count
    return {
        "adapt_nwp": nwp_array[:adapt_window_count],
        "adapt_power": power_array[:adapt_window_count],
        "val_nwp": nwp_array[adapt_window_count:],
        "val_power": power_array[adapt_window_count:],
        "full_window_count": int(nwp_array.shape[0]),
    }


def apply_target_adapt_kshot_limit(split_payload):
    """Limit only target-station adapt windows; source windows keep the normal split."""
    max_windows = int(EXTREME_TARGET_ADAPT_MAX_WINDOWS)
    if max_windows <= 0:
        return split_payload

    adapt_nwp = split_payload["adapt_nwp"]
    adapt_power = split_payload["adapt_power"]
    if adapt_nwp.shape[0] <= max_windows:
        return split_payload

    limited_payload = copy.deepcopy(split_payload)
    heldout_nwp = adapt_nwp[max_windows:]
    heldout_power = adapt_power[max_windows:]
    limited_payload["adapt_nwp"] = adapt_nwp[:max_windows]
    limited_payload["adapt_power"] = adapt_power[:max_windows]
    limited_payload["val_nwp"] = np.concatenate((heldout_nwp, split_payload["val_nwp"]), axis=0)
    limited_payload["val_power"] = np.concatenate((heldout_power, split_payload["val_power"]), axis=0)
    limited_payload["target_adapt_max_windows"] = max_windows
    return limited_payload


def to_tensor_payload(split_payload):
    return {
        "adapt_input": torch.tensor(split_payload["adapt_nwp"], dtype=torch.float32),
        "adapt_target": torch.tensor(split_payload["adapt_power"], dtype=torch.float32),
        "val_input": torch.tensor(split_payload["val_nwp"], dtype=torch.float32),
        "val_target": torch.tensor(split_payload["val_power"], dtype=torch.float32),
        "full_window_count": int(split_payload["full_window_count"]),
    }


def evaluate_state_dict_loss(state_dict, input_tensor, target_tensor):
    if input_tensor.shape[0] == 0 or target_tensor.shape[0] == 0:
        return None
    model_fore_test_task_query.load_state_dict(copy.deepcopy(state_dict))
    model_fore_test_task_query.eval()
    with torch.no_grad():
        outputs = model_fore_test_task_query(input_tensor.to(device))
        loss_value = loss_fn_1(outputs, target_tensor.to(device))
    return float(loss_value.item())


def adapt_state_dict(base_state_dict, adapt_input_tensor, adapt_target_tensor, epochs, log_tag=None, model_label=None):
    if adapt_input_tensor.shape[0] == 0 or adapt_target_tensor.shape[0] == 0:
        return copy.deepcopy(base_state_dict), None

    if epochs <= 0:
        return copy.deepcopy(base_state_dict), evaluate_state_dict_loss(base_state_dict, adapt_input_tensor, adapt_target_tensor)

    model_fore_test_task_support.load_state_dict(copy.deepcopy(base_state_dict))
    optimizer = torch.optim.Adam(
        model_fore_test_task_support.get_trainable_params(), lr=0.0002, betas=(0.5, 0.999)
    )
    adapt_input_device = adapt_input_tensor.to(device)
    adapt_target_device = adapt_target_tensor.to(device)
    final_loss = None
    convergence_record = initialize_convergence_record(
        stage_type="few_shot",
        stage_id=log_tag or (model_label or "few_shot"),
        total_epochs=epochs,
        patience=CONVERGENCE_PATIENCE_FEW_SHOT,
    )

    for i in range(epochs):
        model_fore_test_task_support.train()
        outputs = model_fore_test_task_support(adapt_input_device)
        loss_mse = loss_fn_1(outputs, adapt_target_device)
        optimizer.zero_grad()
        loss_mse.backward()
        optimizer.step()
        final_loss = float(loss_mse.item())
        update_convergence_record(convergence_record, i, final_loss)

        log_interval = FEW_SHOT_LOG_INTERVAL if log_tag is not None else max(1, min(20, epochs))
        if log_tag is not None and should_log_epoch(i, epochs, interval=log_interval):
            if model_label is not None:
                print(
                    f"      [{model_label}] [Epoch {i+1}/{epochs}] "
                    f"[loss_mse: {loss_mse.item():.6f}]"
                )
            writer1.add_scalar(f"loss_penalty_{log_tag}", 0.0, i)
            writer2.add_scalar(f"loss_mse_{log_tag}", loss_mse.item(), i)

    model_fore_test_task_support.eval()
    register_convergence_record(convergence_record)
    return copy.deepcopy(model_fore_test_task_support.state_dict()), final_loss


def run_few_shot_adaptation(base_model_path, save_path, log_tag, model_label, test_input_tensor, test_target_tensor):
    adapted_state, _ = adapt_state_dict(
        base_state_dict=torch.load(base_model_path, map_location=device),
        adapt_input_tensor=test_input_tensor,
        adapt_target_tensor=test_target_tensor,
        epochs=FEW_SHOT_EPOCHS,
        log_tag=log_tag,
        model_label=model_label,
    )
    torch.save(adapted_state, save_path)
    print(f"    ✓ 保存({model_label}): {save_path}")


def compute_window_self_losses(base_model_path, nwp_windows, power_windows):
    if nwp_windows.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    base_state_dict = torch.load(base_model_path, map_location=device)
    losses = []
    for window_index in range(nwp_windows.shape[0]):
        input_tensor = torch.tensor(nwp_windows[window_index:window_index + 1], dtype=torch.float32)
        target_tensor = torch.tensor(power_windows[window_index:window_index + 1], dtype=torch.float32)
        loss_value = evaluate_state_dict_loss(base_state_dict, input_tensor, target_tensor)
        losses.append(1e6 if loss_value is None else float(loss_value))
    return np.asarray(losses, dtype=np.float32)


def apply_source_quality_gate(source_model_path, nwp_windows, power_windows):
    quality_losses = compute_window_self_losses(source_model_path, nwp_windows, power_windows)
    if quality_losses.size == 0:
        return {
            "nwp_windows": np.empty((0, len_realp, dem_realc), dtype=np.float32),
            "power_windows": np.empty((0, len_realp, 1), dtype=np.float32),
            "selected_indices": np.empty((0,), dtype=np.int64),
            "quality_losses": quality_losses,
        }

    keep_count = min(
        quality_losses.size,
        max(EXTREME_MIN_EFFECTIVE_WINDOWS, int(math.ceil(EXTREME_SOURCE_QUALITY_KEEP_RATIO * quality_losses.size))),
    )
    selected_indices = np.argsort(quality_losses)[:keep_count]
    selected_indices = np.sort(selected_indices.astype(np.int64))
    return {
        "nwp_windows": np.asarray(nwp_windows, dtype=np.float32)[selected_indices],
        "power_windows": np.asarray(power_windows, dtype=np.float32)[selected_indices],
        "selected_indices": selected_indices,
        "quality_losses": quality_losses[selected_indices],
    }


def compute_target_conditioned_usefulness(shared_init_state_dict, gated_nwp_windows, gated_power_windows, target_val_input, target_val_target):
    if gated_nwp_windows.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    base_target_val_loss = evaluate_state_dict_loss(shared_init_state_dict, target_val_input, target_val_target)
    if base_target_val_loss is None:
        return np.empty((0,), dtype=np.float32)

    usefulness_scores = []
    for window_index in range(gated_nwp_windows.shape[0]):
        source_input = torch.tensor(gated_nwp_windows[window_index:window_index + 1], dtype=torch.float32)
        source_target = torch.tensor(gated_power_windows[window_index:window_index + 1], dtype=torch.float32)
        adapted_state, _ = adapt_state_dict(
            base_state_dict=shared_init_state_dict,
            adapt_input_tensor=source_input,
            adapt_target_tensor=source_target,
            epochs=EXTREME_USEFULNESS_ADAPT_STEPS,
        )
        adapted_target_val_loss = evaluate_state_dict_loss(adapted_state, target_val_input, target_val_target)
        if adapted_target_val_loss is None:
            usefulness_scores.append(-1e6)
        else:
            usefulness_scores.append(float(base_target_val_loss - adapted_target_val_loss))
    return np.asarray(usefulness_scores, dtype=np.float32)


def select_effective_source_windows(shared_init_state_dict, source_station_id, class_idx, target_payload):
    source_nwp_windows, source_power_windows = extract_extreme_windows_for_station_class(source_station_id, class_idx)
    gated_payload = apply_source_quality_gate(
        source_model_path=get_local_meta_model_path(source_station_id),
        nwp_windows=source_nwp_windows,
        power_windows=source_power_windows,
    )
    gated_nwp_windows = gated_payload["nwp_windows"]
    gated_power_windows = gated_payload["power_windows"]

    usefulness_scores = compute_target_conditioned_usefulness(
        shared_init_state_dict=shared_init_state_dict,
        gated_nwp_windows=gated_nwp_windows,
        gated_power_windows=gated_power_windows,
        target_val_input=target_payload["val_input"],
        target_val_target=target_payload["val_target"],
    )

    positive_indices = np.where(usefulness_scores > 0.0)[0]
    if positive_indices.size == 0:
        return {
            "candidate_window_count": int(source_nwp_windows.shape[0]),
            "gated_window_count": int(gated_nwp_windows.shape[0]),
            "effective_window_count": 0,
            "nwp_windows": np.empty((0, len_realp, dem_realc), dtype=np.float32),
            "power_windows": np.empty((0, len_realp, 1), dtype=np.float32),
            "usefulness_scores": np.empty((0,), dtype=np.float32),
        }

    borrow_budget = min(
        positive_indices.size,
        max(
            EXTREME_MIN_EFFECTIVE_WINDOWS,
            int(math.ceil(EXTREME_SOURCE_BORROW_BUDGET_GAMMA * max(1, target_payload["full_window_count"]))),
        ),
    )
    ranked_positive = positive_indices[np.argsort(usefulness_scores[positive_indices])[-borrow_budget:]]
    ranked_positive = np.sort(ranked_positive.astype(np.int64))
    return {
        "candidate_window_count": int(source_nwp_windows.shape[0]),
        "gated_window_count": int(gated_nwp_windows.shape[0]),
        "effective_window_count": int(ranked_positive.size),
        "nwp_windows": gated_nwp_windows[ranked_positive],
        "power_windows": gated_power_windows[ranked_positive],
        "usefulness_scores": usefulness_scores[ranked_positive],
    }


def compute_sample_count_reliability(sample_count):
    return float(math.log1p(max(0, int(sample_count))))


def compute_query_reliability(query_loss):
    if query_loss is None:
        return 1.0
    return float(math.exp(-EXTREME_WEIGHT_TAU_Q * float(query_loss)))


def compute_target_transferability(target_val_loss):
    if target_val_loss is None:
        return 1.0
    return float(math.exp(-EXTREME_WEIGHT_TAU_T * float(target_val_loss)))


def aggregate_state_dicts(weighted_states):
    aggregate_state = copy.deepcopy(weighted_states[0][0])
    for name in aggregate_state.keys():
        reference_tensor = aggregate_state[name]
        if not torch.is_floating_point(reference_tensor):
            aggregate_state[name] = reference_tensor.clone()
            continue
        weighted_sum = None
        for state_dict, alpha in weighted_states:
            contrib = state_dict[name].detach().float().cpu() * float(alpha)
            weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        aggregate_state[name] = weighted_sum.type_as(reference_tensor)
    return aggregate_state


def aggregate_extreme_updates_uniform(self_update_payload, source_update_payloads):
    all_payloads = [self_update_payload] + list(source_update_payloads)
    uniform_alpha = 1.0 / len(all_payloads)
    aggregate_state = aggregate_state_dicts(
        [(payload["state_dict"], uniform_alpha) for payload in all_payloads]
    )
    return aggregate_state, {
        payload["station_id"]: uniform_alpha for payload in all_payloads
    }


def aggregate_extreme_updates_weighted(target_station_id, self_update_payload, source_update_payloads):
    if not source_update_payloads:
        return copy.deepcopy(self_update_payload["state_dict"]), {target_station_id: 1.0}

    source_scores = {}
    for payload in source_update_payloads:
        m_k_to_s_c = compute_sample_count_reliability(payload["effective_window_count"])
        q_k_to_s_c = compute_query_reliability(payload["source_val_loss"])
        t_k_to_s_c = compute_target_transferability(payload["target_val_loss"])
        source_scores[payload["station_id"]] = float(
            (m_k_to_s_c ** EXTREME_WEIGHT_LAMBDA)
            * (q_k_to_s_c ** EXTREME_WEIGHT_MU)
            * (t_k_to_s_c ** EXTREME_WEIGHT_NU)
        )

    total_source_score = float(sum(source_scores.values()))
    beta_self = min(max(float(EXTREME_WEIGHT_BETA_SELF), 0.0), 1.0)
    remaining_weight = 1.0 - beta_self

    if total_source_score <= 0:
        uniform_source_alpha = remaining_weight / len(source_update_payloads)
        weighted_states = [(self_update_payload["state_dict"], beta_self)]
        weight_map = {target_station_id: beta_self}
        for payload in source_update_payloads:
            weighted_states.append((payload["state_dict"], uniform_source_alpha))
            weight_map[payload["station_id"]] = uniform_source_alpha
        return aggregate_state_dicts(weighted_states), weight_map

    weighted_states = [(self_update_payload["state_dict"], beta_self)]
    weight_map = {target_station_id: beta_self}
    for payload in source_update_payloads:
        alpha_value = remaining_weight * source_scores[payload["station_id"]] / total_source_score
        weighted_states.append((payload["state_dict"], alpha_value))
        weight_map[payload["station_id"]] = alpha_value
    return aggregate_state_dicts(weighted_states), weight_map


def save_state_dict(state_dict, save_path):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(state_dict, save_path)
    return save_path


def run_target_refinement(base_state_dict, target_payload, log_tag, model_label):
    return adapt_state_dict(
        base_state_dict=base_state_dict,
        adapt_input_tensor=target_payload["adapt_input"],
        adapt_target_tensor=target_payload["adapt_target"],
        epochs=EXTREME_TARGET_REFINEMENT_EPOCHS,
        log_tag=log_tag,
        model_label=model_label,
    )

for station_id in station_ids:
    print(f"\n{'='*70}")
    print(f"场站 {station_id} 的Few-shot适应")
    print(f"{'='*70}")
    
    for i_class in range(4):
        print(f"\n  极端天气类别 {i_class+1}:")

        nwp_extre_class, p_extre_class = extract_extreme_windows_for_station_class(station_id, i_class)
        num_samples = nwp_extre_class.shape[0]
        target_split_payload = split_extreme_adapt_val(nwp_extre_class, p_extre_class)
        target_split_payload = apply_target_adapt_kshot_limit(target_split_payload)
        target_payload = to_tensor_payload(target_split_payload)

        print(f"    样本数: {num_samples}")
        if EXTREME_TARGET_ADAPT_MAX_WINDOWS > 0:
            print(f"    target K-shot adapt窗口上限: {EXTREME_TARGET_ADAPT_MAX_WINDOWS}")
        print(
            f"    训练轮数: {FEW_SHOT_EPOCHS}, "
            f"few-shot loss={'CDRM+MSE' if FEW_SHOT_USE_CDRM else 'MSE'}"
        )
        print(
            f"    adapt/val: {target_payload['adapt_input'].shape[0]} / "
            f"{target_payload['val_input'].shape[0]}"
        )

        shared_init_state = torch.load(
            get_local_meta_model_path(station_id) if USE_FEDERATION and not USE_PSEUDO_FED else PROPOSED_META_MODEL_PATH,
            map_location=device,
        )

        lmt_state_dict, lmt_final_loss = adapt_state_dict(
            base_state_dict=shared_init_state,
            adapt_input_tensor=target_payload["adapt_input"],
            adapt_target_tensor=target_payload["adapt_target"],
            epochs=FEW_SHOT_EPOCHS,
            log_tag=f"lmt_station{station_id}_class{i_class}",
            model_label="LMT",
        )
        lmt_model_name = get_lmt_model_path(station_id, i_class)
        save_state_dict(lmt_state_dict, lmt_model_name)
        print(f"    ✓ 保存(LMT): {lmt_model_name}")
        all_personalized_models[f'lmt_{station_id}_class{i_class}'] = lmt_model_name

        target_val_loss = evaluate_state_dict_loss(
            lmt_state_dict,
            target_payload["val_input"],
            target_payload["val_target"],
        )
        self_update_payload = {
            "station_id": station_id,
            "state_dict": lmt_state_dict,
            "effective_window_count": target_payload["full_window_count"],
            "source_val_loss": lmt_final_loss if lmt_final_loss is not None else target_val_loss,
            "target_val_loss": target_val_loss,
        }

        source_update_payloads = []
        for source_station_id in station_ids:
            if source_station_id == station_id:
                continue
            source_screening_payload = select_effective_source_windows(
                shared_init_state_dict=shared_init_state,
                source_station_id=source_station_id,
                class_idx=i_class,
                target_payload=target_payload,
            )
            print(
                f"    [screen] source_station={source_station_id} -> target_station={station_id}, "
                f"candidate={source_screening_payload['candidate_window_count']}, "
                f"gated={source_screening_payload['gated_window_count']}, "
                f"effective={source_screening_payload['effective_window_count']}"
            )
            if source_screening_payload["effective_window_count"] == 0:
                continue

            source_split_payload = split_extreme_adapt_val(
                source_screening_payload["nwp_windows"],
                source_screening_payload["power_windows"],
            )
            source_payload = to_tensor_payload(source_split_payload)
            source_state_dict, source_final_loss = adapt_state_dict(
                base_state_dict=shared_init_state,
                adapt_input_tensor=source_payload["adapt_input"],
                adapt_target_tensor=source_payload["adapt_target"],
                epochs=FEW_SHOT_EPOCHS,
                log_tag=f"extreme_source_station{source_station_id}_to_{station_id}_class{i_class}",
                model_label=f"source{source_station_id}->target{station_id}",
            )
            source_update_payloads.append({
                "station_id": source_station_id,
                "state_dict": source_state_dict,
                "effective_window_count": source_screening_payload["effective_window_count"],
                "source_val_loss": evaluate_state_dict_loss(
                    source_state_dict,
                    source_payload["val_input"],
                    source_payload["val_target"],
                ) if source_payload["val_input"].shape[0] > 0 else source_final_loss,
                "target_val_loss": evaluate_state_dict_loss(
                    source_state_dict,
                    target_payload["val_input"],
                    target_payload["val_target"],
                ),
            })

        fedavg_aggregate_state, fedavg_weight_map = aggregate_extreme_updates_uniform(
            self_update_payload,
            source_update_payloads,
        )
        fedavg_final_state, _ = run_target_refinement(
            base_state_dict=fedavg_aggregate_state,
            target_payload=target_payload,
            log_tag=f"extreme_fedavg_station{station_id}_class{i_class}",
            model_label="Extreme-FedAvg:target_refine",
        )
        fedavg_model_name = get_extreme_fedavg_model_path(station_id, i_class)
        save_state_dict(fedavg_final_state, fedavg_model_name)
        print(f"    ✓ 保存(Extreme-FedAvg): {fedavg_model_name} weights={fedavg_weight_map}")
        all_personalized_models[f'extreme_fedavg_{station_id}_class{i_class}'] = fedavg_model_name

        proposed_aggregate_state, proposed_weight_map = aggregate_extreme_updates_weighted(
            target_station_id=station_id,
            self_update_payload=self_update_payload,
            source_update_payloads=source_update_payloads,
        )
        proposed_final_state, _ = run_target_refinement(
            base_state_dict=proposed_aggregate_state,
            target_payload=target_payload,
            log_tag=f"proposed_a_station{station_id}_class{i_class}",
            model_label="Proposed-A:target_refine",
        )
        proposed_model_name = get_proposed_a_model_path(station_id, i_class)
        save_state_dict(proposed_final_state, proposed_model_name)
        print(f"    ✓ 保存(Proposed-A): {proposed_model_name} weights={proposed_weight_map}")
        all_personalized_models[f'proposed_a_{station_id}_class{i_class}'] = proposed_model_name

        # Meta-only：同口径执行 step-11 few-shot，确保与论文消融对齐
        if TRAIN_META_ONLY_BASELINE:
            meta_only_model_name = get_meta_only_extreme_model_path(station_id, i_class)
            run_few_shot_adaptation(
                base_model_path=get_local_meta_only_model_path(station_id) if USE_FEDERATION and not USE_PSEUDO_FED else META_ONLY_MODEL_PATH,
                save_path=meta_only_model_name,
                log_tag=f"meta_only_station{station_id}_class{i_class}",
                model_label="Meta-only",
                test_input_tensor=target_payload["adapt_input"],
                test_target_tensor=target_payload["adapt_target"]
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
        model_paths = {
            "lmt": get_lmt_model_path(station_id, i_class),
            "extreme_fedavg": get_extreme_fedavg_model_path(station_id, i_class),
            "proposed_a": get_proposed_a_model_path(station_id, i_class),
        }

        for model_key, model_name in model_paths.items():
            model_fore_test_task_query.load_state_dict(torch.load(model_name, map_location=device))
            with torch.no_grad():
                Test_input_device = Test_input_c_st.to(device)
                Test_output = model_fore_test_task_query(Test_input_device)
                test_output = Test_output.to(device0)
                test_output_np = np.array(test_output.reshape(-1,dem_realp))
                all_test_results[station_id]['predictions'][f'{model_key}_extreme_{i_class}'] = test_output_np

        print(f"  ✓ 极端类别{i_class+1}（LMT/Extreme-FedAvg/Proposed-A）")

# 保存所有结果
print("\n保存所有场站测试结果...")
all_results_dir = os.path.dirname(ALL_STATIONS_TEST_RESULTS_PATH)
if all_results_dir:
    os.makedirs(all_results_dir, exist_ok=True)
scio.savemat(ALL_STATIONS_TEST_RESULTS_PATH, {'all_test_results': all_test_results, 'Cap': Cap})
print(f"✓ 已保存: {ALL_STATIONS_TEST_RESULTS_PATH}")
export_convergence_report(
    CONVERGENCE_REPORT_PATH,
    convergence_records,
    {
        "protocol_name": PROTOCOL_NAME,
        "protocol_data_dir": PROTOCOL_DATA_DIR,
        "protocol_metadata_path": PROTOCOL_METADATA_PATH,
        "artifact_dir": ARTIFACT_DIR,
        "model_output_dir": MODEL_OUTPUT_DIR,
        "logs_train_dir": LOGS_TRAIN_DIR,
        "all_stations_test_results_path": ALL_STATIONS_TEST_RESULTS_PATH,
        "sample_interval_hours": SAMPLE_INTERVAL_HOURS,
        "downsample_offset": DOWNSAMPLE_OFFSET,
        "len_realp": LEN_REALP,
        "points_per_day": POINTS_PER_DAY,
        "window_span_hours": WINDOW_SPAN_HOURS,
        "yearly_protocol_enabled": YEARLY_PROTOCOL_ENABLED,
        "seasonal_protocol_enabled": SEASONAL_PROTOCOL_ENABLED,
        "use_federation": USE_FEDERATION,
        "use_pseudo_fed": USE_PSEUDO_FED,
        "train_meta_only_baseline": TRAIN_META_ONLY_BASELINE,
        "skip_local_pretrain": SKIP_LOCAL_PRETRAIN,
        "skip_local_meta": SKIP_LOCAL_META,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "proposed_meta_epochs": PROPOSED_META_EPOCHS,
        "meta_only_meta_epochs": META_ONLY_META_EPOCHS,
        "few_shot_epochs": FEW_SHOT_EPOCHS,
        "pretrain_log_interval": PRETRAIN_LOG_INTERVAL,
        "meta_log_interval": META_LOG_INTERVAL,
        "few_shot_log_interval": FEW_SHOT_LOG_INTERVAL,
        "extreme_weight_beta_self": EXTREME_WEIGHT_BETA_SELF,
        "extreme_source_borrow_budget_gamma": EXTREME_SOURCE_BORROW_BUDGET_GAMMA,
        "extreme_target_refinement_epochs": EXTREME_TARGET_REFINEMENT_EPOCHS,
        "extreme_target_adapt_max_windows": EXTREME_TARGET_ADAPT_MAX_WINDOWS,
        "enable_convergence_monitor": ENABLE_CONVERGENCE_MONITOR,
    },
)

print("\n" + "="*70)
print("✓✓✓ 训练和测试全部完成！")
print(f"生成的模型: {len(all_personalized_models)}个个性化模型（LMT/Extreme-FedAvg/Proposed-A）")
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
