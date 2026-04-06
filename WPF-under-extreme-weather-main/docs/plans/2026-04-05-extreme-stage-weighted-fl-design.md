# Extreme-Stage Weighted FL Design

## Goal
在保留 `3.24.23: RAPP-original data` 方法骨架的前提下，将跨 client 共享从 `normal/conventional pretrain` 转移到 `extreme stage`，回答新的主命题：

> 在极端天气小样本预测中，跨 client 的共享应该放在哪个阶段、以什么目标进行，才真正有助于 extreme few-shot？

## Core Shift
- 保留原始版本的三阶段主骨架：`local conventional pretrain -> local conventional meta -> per-class extreme adaptation/evaluation`
- 放弃当前 seasonal 六客户端方案中的 `normal-only federated pretrain` 主叙事
- 新的共享发生在 **same-class extreme adaptation** 阶段，而不是 conventional 阶段
- 新的聚合规则不是 plain `FedAvg`，而是 **target-conditioned reliability-aware aggregation**

## Data Protocol
- 场站：恢复到 `58 / 59 / 60` 三场站
- 时间口径：`2022` 作为训练侧，`2023` 作为测试侧
- 训练与测试对象：围绕 extreme class 显式重建，不复用旧 `.mat` 中未分年的 `p_extre_class*`
- scarcity 定义：`target` 在某一类 extreme task 上的数据有限，而不是全年 normal/conventional 数据有限

## Method

### 主创新点 1：Extreme-Type-Conditioned Reliability-Aware Federated Adaptation（Extreme Stage）
- 目标：让跨 client 的共享直接服务于 extreme few-shot 任务，而不是间接依赖 normal-only federated prior。
- 对每个极端天气类别 `c`，定义参与集合：
  - `G_c = {k | N_k^c > 0}`
  - `N_k^c` 为 client `k` 在训练年中类别 `c` 的窗口数。
- 对每个 client `k` 的类别 `c`，训练侧样本集记为 `D_k^c`，划分为：
  - `D_{k,c}^{sup}`：本地支持集
  - `D_{k,c}^{qry}`：本地可靠性评估集
- 对目标场站 `s`、类别 `c`，服务器维护共享初始化 `theta_{s,c}^{(t)}`。
- 每个参与 client `k ∈ G_c` 在本地支持集上做极端适配：
  - `theta_{k,c}^{(t+1)} = U(theta_{s,c}^{(t)} ; D_{k,c}^{sup})`
- 服务器不再做 plain `FedAvg`，而是对目标 `s` 做条件化聚合：
  - `theta_{s,c}^{(t+1)} = sum_{k in G_c} alpha_{k→s}^{c,(t)} * theta_{k,c}^{(t+1)}`
- 其中 `alpha_{k→s}^{c,(t)}` 由样本量、本地适配质量和目标场景相似度共同决定。
- 与 plain `FedAvg` 的差别不在“是否联邦”，而在“联邦更新是否 truly aligned with the target extreme task”。

### 主创新点 2：Scenario Prototype Construction and Reliability Weight Calibration
- 目标：把“同类 extreme 的共享”进一步约束为“更可靠、且更像目标场景的 client 更新占更大权重”。
- 对于一个 extreme window `(x,y)`，定义场景描述向量：
  - `phi(x,y) = [mu_x, std_x, min_x, max_x, mu_y, std_y, ramp_y, drop_y]`
- 其中：
  - `mu_x / std_x / min_x / max_x`：各 NWP 特征在窗口内的统计量
  - `mu_y / std_y`：功率均值与标准差
  - `ramp_y = (1/(T-1)) * sum_{t=2}^T |y_t - y_{t-1}|`
  - `drop_y = max(y) - min(y)`
- 对 client `k` 的类别 `c`，定义支持集原型：
  - `g_{k,c} = (1 / |D_{k,c}^{sup}|) * sum_{(x,y) in D_{k,c}^{sup}} phi(x,y)`
- 对目标 client `s`，定义目标原型：
  - `g_{s,c} = (1 / |D_{s,c}^{sup}|) * sum_{(x,y) in D_{s,c}^{sup}} phi(x,y)`
- 场景相似度定义为：
  - `sim_{k→s}^c = exp(- ||g_{k,c} - g_{s,c}||_2^2 / sigma_c^2 )`
- 样本量可靠性项：
  - `m_{k,c} = log(1 + |D_{k,c}^{sup}|)`
- 本地适配质量项：
  - 若 `|D_{k,c}^{qry}| > 0`，则
    - `L_{qry}^{k,c,(t)} = L(theta_{k,c}^{(t+1)} ; D_{k,c}^{qry})`
    - `q_{k,c}^{(t)} = exp(-tau * L_{qry}^{k,c,(t)})`
  - 否则 `q_{k,c}^{(t)} = 1`
- 最终未归一化权重：
  - `a_{k→s}^{c,(t)} = (m_{k,c})^lambda * (q_{k,c}^{(t)})^mu * (sim_{k→s}^c)^nu`
- 归一化权重：
  - `alpha_{k→s}^{c,(t)} = a_{k→s}^{c,(t)} / sum_{j in G_c} a_{j→s}^{c,(t)}`
- 第一版默认：
  - `tau = 1`
  - `lambda = 1`
  - `mu = 1`
  - `nu = 2`
- 理由：当前主要矛盾是“共享对象不够像目标”，因此优先强化 `sim`。

## Baselines
- 自 `2026-04-06` 起，yearly clean ablation 统一采用共同 backbone：
  - `common upstream = local_pretrain`
  - `common midstream = local_meta`
- 在这个共同 backbone 上定义三条方法：
  - `Local_Extreme_Transfer (LMT-new)`：目标场站仅用自己的该类 extreme 支持集做本地适配
  - `Extreme-FedAvg`：各场站先从各自 `local_meta` 出发完成 same-class local extreme update，再做 plain `FedAvg`
  - `Proposed-A`：各场站先从各自 `local_meta` 出发完成 same-class local extreme update，再做 reliability-aware aggregation

## Privacy Scope
- 原始 extreme-weather 数据始终保留在本地
- 客户端之间不交换原始样本
- 跨 client 共享只通过模型更新、样本数统计与场景原型/质量分数实现
- 该口径足以支持 federated/data-local privacy 表述，但不等同于 secure aggregation、DP 或密码学强保护

## Why This Is Still Based on RAPP-original Data
- 保留原始骨架：`local conventional pretrain -> local conventional meta -> per-class extreme adaptation`
- 保留原始 backbone：`TCN + LWP + fore_baselearner`
- 保留原始 few-shot 单位：按 extreme class 单独适配与评估
- 新增的是：
  - extreme-stage FL
  - target-conditioned weighted aggregation
  - 年份显式切分的数据协议

## Validation Scope
- 优先验证：`Proposed-A` 是否优于 `Extreme-FedAvg` 与 `LMT-new`
- 指标：`nMAE_%`, `nRMSE_%`, `WD_%`, `R_p<0.05_%`
- 先按 `(station, extreme_class)` 逐任务报告，不先强行做总平均
