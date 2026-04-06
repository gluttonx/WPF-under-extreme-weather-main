# 2026-04-06 Yearly Clean Extreme Ablation Design

## Context
- 当前 yearly 路线的 `LMT-new` 走 `local_pretrain -> local_meta -> few-shot`。
- 当前 yearly 路线的 `Extreme-FedAvg` / `Proposed-A` 仍然依赖联邦上游初始化，导致 extreme-stage 消融掺入了上游跨站共享。
- 更关键的是，现有代码虽然声明了 `Extreme-FedAvg` / `Proposed-A`，但 yearly few-shot 主循环里还没有把 same-class extreme plain aggregation / reliability-aware aggregation 真正用于模型更新，当前差异主要来自初始化路径。

## Goal
- 把 yearly 三条主线统一改为：
  - `LMT-new`: `local_pretrain -> local_meta -> local extreme few-shot`
  - `Extreme-FedAvg`: `local_pretrain -> local_meta -> same-class extreme plain aggregation`
  - `Proposed-A`: `local_pretrain -> local_meta -> same-class extreme reliability-aware aggregation`
- 把 yearly 结果主输出改为论文 `TABLE IV` 风格宽表，同时保留 task-level 长表。

## Design Decision 1: Common Upstream and Midstream
- 共同上游固定为各场站 `local_pretrain`。
- 共同中游固定为各场站 `local_meta`。
- 这样 three-way ablation 的唯一差异落在 extreme-stage：
  - 不协作
  - plain FedAvg
  - reliability-aware weighted aggregation

## Design Decision 2: Extreme-Stage Aggregation Scope
- 聚合对象保持为 `model_fore` 的完整 state dict，不在本次额外引入 shared/local 参数拆分。
- 每个 `(target_station, extreme_class)` 任务中：
  - `LMT-new`：从目标站 `local_meta` 起步，仅在目标站 support set 上本地 few-shot。
  - `Extreme-FedAvg`：
    - 每个具备该 extreme class 的站点都从自己的 `local_meta` 起步；
    - 在各自该类 support set 上做本地 few-shot；
    - 服务器对这些站点的 tuned state 做等权平均；
    - 将聚合后的 state 另存为目标站该类模型。
  - `Proposed-A`：
    - 客户端本地 few-shot 同上；
    - 用 `m_k,c * q_k,c * sim_k→s^nu` 计算 target-conditioned 权重；
    - 对 tuned state 做加权平均得到目标站该类模型。

## Design Decision 3: Reliability Signals
- `m_k,c`：support window 数量可靠性，继续用 `log1p(sample_count)`。
- `q_k,c`：few-shot 后的本地 query/support 质量代理。本轮实现沿用 few-shot final loss 作为质量代理，并通过 `exp(-tau * loss)` 映射为可靠性。
- `sim_k→s^c`：使用现有 `phi(x, y)` / class prototype 与 RBF similarity。
- `Extreme-FedAvg` 直接使用 uniform weights，不读取上面三类信号。

## Design Decision 4: Result Export Contract
- `multi_station_performance_task_level.csv`
  - 保留原始 yearly task-level 结果；
  - 每行一条 `(station_id, extreme_class, model)`。
- `multi_station_performance.csv`
  - 改为 paper-facing `TABLE IV` 风格宽表；
  - 每行一个模型；
  - 横向展开四类天气：
    - `HighWind_E_M_%`, `HighWind_E_R_%`, `HighWind_WD`
    - `HighTemperature_E_M_%`, `HighTemperature_E_R_%`, `HighTemperature_WD`
    - `ColdWave_E_M_%`, `ColdWave_E_R_%`, `ColdWave_WD`
    - `Frost_E_M_%`, `Frost_E_R_%`, `Frost_WD`
  - 右侧追加：
    - `Training_duration_s`
    - `R_p<0.05_%`

## Design Decision 5: Aggregation for Paper Table
- 宽表不能简单对已有 task-level 百分比做普通平均。
- 对每个 `(model, extreme_class)`：
  - 先收集三站该类的全部 `true_events` / `pred_events`；
  - 再重新调用 `calc_paper_metrics(...)` 计算 `E_M / E_R / WD / R_p<0.05`。
- 总体 `R_p<0.05_%` 也按该模型 across all classes 的拼接事件重新计算。
- `Training_duration_s` 使用 yearly 模型名重新映射：
  - 三条 yearly 方法都包含 `local_pretrain + local_meta + 对应 extreme-stage few-shot`
  - 其中 `Extreme-FedAvg` / `Proposed-A` 的 few-shot 时长各自按自身 tag span 估计。

## Non-Goals
- 不修改旧的 six-client seasonal protocol 主表逻辑。
- 不在本轮引入 secure aggregation、DP 或额外隐私机制。
- 不在本轮改变 backbone、loss 定义或 weather class 划分。

## Validation
- AST / unit tests 需要覆盖：
  - yearly baselines 的 common `local_meta` init routing；
  - same-class extreme plain / weighted aggregation helpers 的存在与调用；
  - yearly evaluation 同时输出 task-level 长表与 `TABLE IV` 宽表；
  - 宽表包含 `Training_duration_s` 和 overall `R_p<0.05_%`。
- 运行级验证：
  - `py_compile`
  - targeted unittest
  - yearly launcher `--smoke --dry-run`
