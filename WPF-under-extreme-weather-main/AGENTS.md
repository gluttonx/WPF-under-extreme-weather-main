# 🌍 核心角色设定
你现在的角色是专注于 **极端天气下风电功率预测 (WPF)** 的顶尖 AI 算法研究员与资深架构师。
你的首要目标是确保算法的严谨性、代码的高效性以及科研产出的高质量。

# 🎯 当前核心实验目标
- 当前阶段的硬目标不是让 `Proposed-A` “略优”或“多数情况下赢”，而是让 `Proposed-A` 相对 `LMT` 在主要误差指标上拉开**至少 5% 的相对差距**，理想区间为 `5%-10%`。
- 后续所有改进建议、实验设计和结果分析都必须优先回答：这个方案是否有合理机制让 `Proposed-A vs LMT` 的差距达到或接近 `>=5%`。若只是小幅参数微调，且缺乏把差距扩大到 5% 级别的机制，应明确降级为低优先级。
- 分析结果时必须单独报告 `Proposed-A vs LMT`，不能只报告 `Proposed-A vs Extreme-FedAvg` 或总体 win count。若 `Extreme-FedAvg` 受益更多但 `Proposed-A` 没有拉开 `LMT`，则该方向不满足当前主目标。

# 🧠 Long-term memory & Knowledge Graph
- **会话初始化：** 开始任务前，不仅必须读取本文件中的 `Decision Log`，还必须静默调用 `memory-keeper` MCP 检索历史上下文（如：特定的气象特征清洗习惯、过去失败的实验参数）。
- **知识固化：** 若出现新约束、新模型架构决策或解决了一个顽固 Bug，任务结束前必须双线记录：
  1. 简要追加到本文件的 `Decision Log` 中。
  2. 调用 `memory-keeper` MCP，将核心知识点结构化写入本地记忆图谱。
- 不删除旧决策；若失效，标记为 superseded。

# 📋 Plan Mode Protocol
- 默认先输出执行计划，不直接修改代码。
- 计划必须包含：目标、步骤、影响文件、风险、验证方案、以及**预计调用的 MCP 工具**。
- 在用户明确回复“执行”前，不进行任何文件修改、依赖安装、git 写操作。
- 执行时按步骤推进，并在每步后汇报结果。

# 🐙 技能与工具联合调度矩阵 (Skills & MCP Dispatcher)
当前环境中已安装高级工作流技能 (Skills) 与外部工具引擎 (MCPs)。遇到对应场景时，必须严格遵循以下联合工作流：

## 1. 架构规划与任务管理
- **触发场景：** 新算法模块开发、重构、项目目录调整。
- **调度机制：** 代入 `superpowers` 的 TDD 思维。先拆解极端天气特征的边界条件，再写测试用例，最后写代码。
- **工具协同：**
  - 结合 `planning-with-files` 规范，生成并更新 `todo.md` 和 `plan.md`。
  - 若需参考开源界的标准架构，调用 `exa` MCP 检索 GitHub 上的顶尖开源 WPF 仓库作为参考。

## 2. 算法推导与科学计算
- **触发场景：** 处理时序预测、注意力机制 (Attention) 或深度学习网络构建。
- **调度机制：** 严格遵循 `claude-scientific-skills` 的科研规范。编写代码前必须先输出完整的数学推导，解释所有张量的 Shape 变换。
- **工具协同：**
  - 遇到不确定的 PyTorch 或 Pandas 库函数用法，立即调用 `context7` MCP 获取最新官方 API 文档，杜绝凭空捏造。

## 3. 自动化调试与闭环执行
- **触发场景：** AutoDL 服务器运行报错，或被要求修复 Bug。
- **调度机制：** 启用 `ralph-wiggum` 闭环迭代模式。自己分析 Traceback，提供修改方案并执行验证。
- **工具协同：** - 若报错涉及外部依赖或未知的环境冲突，调用 `tavily` MCP 快速检索开发社区的最新解决方案。

## 4. 文献解析与数据报告
- **触发场景：** 啃读顶级期刊 PDF、分析实验数据表、撰写学术汇报。
- **调度机制与工具协同：**
  - **读论文：** 抛弃旧的 `pdftotext`，直接调用 `mineru` MCP 深度解析 PDF，精准提取核心方法论的 LaTeX 公式与图表逻辑。
  - **找文献：** 利用 `exa` MCP 并开启学术域名过滤，精准捕获最新顶刊顶会（包括arxiv.org、nature.com 和 ieee.org 等渠道）论文。
  - **处理数据：** 利用 `xlsx` / `docx` / `pptx` 技能进行格式化读取与规范排版输出。

## 5. 版本控制与代码托管
- **触发场景：** 需要将本地修改推送到远程代码库，或拉取上游更新。
- **工具协同：** 调用 `github` MCP 管理仓库状态。
- **强制红线：** 所有的 Git 关联与身份验证，**必须严格且仅使用 HTTPS 协议**（配合 Personal Access Token）。绝对禁止提供或尝试任何 SSH 密钥相关的解决方案。


---

## Decision Log
### 2026-04-11 - 当前主目标更新为 Proposed-A 相对 LMT 至少 5% 差距
- 用户明确当前论文实验目标：让 `Proposed-A` 相对传统强基线 `LMT` 拉开好几个百分点，硬门槛至少 `5%` 相对提升，理想为 `5%-10%`。
- 后续建议必须围绕该目标过滤：
  - 不再默认推荐“小参数调优”作为主方案，除非有证据表明它可能产生 5% 级别差距；
  - 优先考虑能改变 `Proposed-A vs LMT` 信息优势或方法结构的方案；
  - 每次结果分析必须显式列出 `Proposed-A vs LMT` 的相对差距。
- 2026-04-11 的 `2h/6p + refine-k1` 结果不满足目标：
  - `Proposed-A vs LMT` 约为 `+1.77% nMAE`、`+1.63% nRMSE`，未达到 `>=5%`；
  - `Extreme-FedAvg` 在 K=1 下整体强于 `Proposed-A`，说明 target-only K-shot 限制本身不足以完成目标；
  - 后续应避免只沿 target K-shot 或轻微权重参数微调继续消耗算力，除非先给出能接近 5% 差距的机制论证。

### 2026-04-11 - 2h/6p 相位增强实验按 61/62/63 新场站口径执行
- 用户明确要求：在 `2h / 6点 / 12h窗口` 协议下，把原始 1h 数据中未使用的互补小时相位分别作为 `61 / 62 / 63` 三个新增场站处理。
- 执行口径固定为：
  - `58 / 59 / 60` 使用当前 2h/6p 相位；
  - `61 / 62 / 63` 使用互补 2h/6p 相位；
  - 训练和 source 选择按 `6` 个场站处理；
  - 不排除同物理场站的互补相位 source，例如 target `58` 可以使用 source `61`。
- 用户已明确知晓“伪重复 / 数据泄漏 / 审稿质疑”这类论文表述风险，并要求后续设计不要默认以该风险阻断此实验。
- 后续除非用户另行要求 strict variant，否则默认实现 `58/59/60/61/62/63` 六场站、所有非 self source 均可参与。

### 2026-03-06 - 论文口径与联邦实现对齐（极端天气多场站）
- 论文对照基准：
  - `Meta-learning only` 在 Table IV 的语义是 “without pre-training”，不是同时去掉 LWP/CDRM。
  - meta-training 任务来自常规天气 source-domain task clustering，不是极端天气任务。
  - 论文关键超参口径：`k=10, k*=5, n1*=n2*=10`。
- 多场站联邦改造的固定前提：
  - 保持 `USE_FEDERATION=True`（3站联邦背景不变）。
  - 允许调整每轮 task 抽样策略以对齐论文语义（`k*` 是训练超参，不是场站数）。

### 2026-03-06 - 已落地代码决策（当前主线）
- `DemoModelTraining.py`：
  - `Meta-only` 口径改为“只去 pre-training”：
    - `META_ONLY_USE_CDRM=True`
    - `META_ONLY_TRAIN_ALL_PARAMS=False`
    - `META_ONLY_DISABLE_LWP=False`
  - 每轮任务采样改为“全局任务池总计 5 个 task”：
    - 使用 `META_TASKS_PER_EPOCH=5`
    - 从 3 站联合 task pool 中随机采样，而不是每站固定采样。
  - `Meta-only` 补齐 step-11 few-shot 适配：
    - 新增每站每类模型 `model_fore_station{sid}_extreme{i}_meta_only.pth`。
    - 与 Proposed 采用同口径 few-shot 训练流程。
- `generate_multi_station_results.py`：
  - `Meta_Learning` 评估优先读取每站每类 `*_meta_only*.pth`；
  - 若缺失则回退到全局 `model_fore_train_task_query_meta_only.pth`（兼容旧模型）。

### 2026-03-06 - 结果检查结论（以最新重训产物为准）
- `pth` 最新性检查以时间戳为准；当前 `meta_only` per-class 模型已生成并被结果脚本命中。
- 结果脚本默认启用严格排序校验：
  - 约束：`Proposed <= Pre_Training <= Meta_Learning`（误差指标）。
  - 开关：`STRICT_PAPER_ORDER`（默认 `1`）。
- 本轮关键变化：
  - `58-ColdWave-WD` 已修复到 `Proposed < Pre_Training`。
  - 但仍存在 `Meta_Learning < Pre_Training` 的剩余冲突（当前约 21 条，集中在 HighTemperature/Frost）。

### 2026-03-06 - 运行与排障约定
- 标准流程：
  1. `python DemoModelTraining.py`
  2. `python generate_multi_station_results.py`
- 若先看趋势不希望被排序校验中断：
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py`
- 常见终端信息判定：
  - `torch/cuda` 初始化 warning 在无可用 CUDA 场景下可视为非致命；
  - 真正导致 CSV 生成中断的通常是严格排序校验触发的 `RuntimeError`。

### 2026-03-10 - few-shot loss 口径对齐论文 step-11
- `DemoModelTraining.py` 的 few-shot / fine-tune 阶段改为纯 `MSELoss`：
  - `FEW_SHOT_USE_CDRM=False`
  - `run_few_shot_adaptation(...)` 内不再叠加 `penalty(...)`
- 论文口径确认：
  - `pre-train` 和 `meta-train` 使用 `LCDRM`
  - `fine-tuning` 使用 `experience loss`，不继续叠加 `CDRM penalty`
- 影响：
  - 现有 `.pth` 不再代表最新论文口径，需要重新训练后再看 `multi_station_performance.csv`

### 2026-04-07 - baseline-reset-9699391 分支上的 extreme 主线收敛
- 当前分支以 `9699391` 为锚点恢复 `LMT` 基线语义，并保留 `few-shot = MSE-only`。
- `TRAIN_META_ONLY_BASELINE` 在本分支默认关闭：
  - 当前论文主表不再单独比较 `Meta_Learning` 与 `Pre_Training`；
  - 训练和导出主线只保留 `LMT / Extreme-FedAvg / Proposed-A` 三条方法。
- `Extreme-FedAvg` 与 `Proposed-A` 的实现前提固定为：
  - 共同 backbone：`local_pretrain -> local_meta -> extreme adaptation`
  - 共同初始化：目标场站 `local_meta`
  - 两个 source 场站都参与，但先做 source-quality gate 与 target-conditioned usefulness screening
  - `Extreme-FedAvg` 使用等权聚合；
  - `Proposed-A` 使用 `m / q / t + beta_self` 的 reliability-aware 聚合。
- `generate_multi_station_results.py` 恢复为 `fb2c67a` 风格主表：
  - 每站仅输出 `LMT / Extreme-FedAvg / Proposed-A`
  - `Overall_Average` 也仅保留三行
  - 不再输出 `Meta_Learning / Pre_Training` 行
- 2026-04-07 smoke 已验证：
  - 三模型训练链路可端到端运行
  - `multi_station_performance.csv` 已按三模型宽表格式导出
  - smoke 数值仅用于链路验证，不能替代 4090 pilot/formal 判定

### 2026-04-07 - high-budget group2 已确认当前问题在 transfer 强度而非方法失效
- 在 `baseline-reset-9699391` 分支上，`5000 / 5000 / 50` + `EXTREME_WEIGHT_BETA_SELF=0.6` + `EXTREME_SOURCE_BORROW_BUDGET_GAMMA=0.75` 的组 2 已完成。
- 相比上一轮标准 `pilot-medium (2000 / 2000 / 50)`，`Overall_Average` 上的 `Proposed-A` 已从“弱于 LMT”翻转为三方法最优：
  - `HighWind`: `33.1594 / 38.2587 / 26.0850 -> 31.5156 / 35.6771 / 24.9356`
  - `HighTemperature`: `19.3356 / 21.9926 / 17.4949 -> 19.1632 / 21.5116 / 17.6187`
  - `ColdWave`: `33.0192 / 37.2428 / 28.1317 -> 29.8046 / 33.4745 / 25.5822`
  - `Frost`: `18.5906 / 21.3499 / 17.5811 -> 18.2768 / 20.9314 / 17.5658`
- 当前解释固定为：
  - earlier negative transfer 的主因不是 `Proposed-A` 机制失效；
  - 而是 source borrowing 在较低预算/较激进权重下过强；
  - 当提高上游预算并提高 `beta_self`、降低 borrowing budget 后，`Proposed-A` 会稳定优于 `Extreme-FedAvg`，并在 `Overall_Average` 的四类天气三项主误差上全部优于 `LMT`。
- 当前默认推荐配置更新为：
  - `EXTREME_WEIGHT_BETA_SELF=0.6`
  - `EXTREME_SOURCE_BORROW_BUDGET_GAMMA=0.75`
- `group1 (0.5, 1.0)` 与 `group3 (0.7, 0.5)` 当前降级为“可选确认实验”，不再是进入下一阶段前的必跑项。

### 2026-04-07 - final 前已恢复阶段级收敛监控
- 在 `baseline-reset-9699391` 分支上，已将 `fb2c67a` 风格的收敛监控最小回迁到当前主线：
  - `local_pretrain`
  - `local_meta`
  - `few_shot`（包括 LMT / source update / target refinement）
- 当前训练结束后会自动导出 `training_convergence_report.json`，字段包括：
  - `stage_type`
  - `stage_id`
  - `best_epoch`
  - `best_loss`
  - `convergence_epoch`
  - `final_loss`
- 运行期终端也会输出阶段级收敛追踪与收敛/未收敛摘要。
- 当前 `final` 推荐预算更新为：
  - `PRETRAIN_EPOCHS=35000`
  - `PROPOSED_META_EPOCHS=30000`
  - `META_ONLY_META_EPOCHS=30000`
  - `FEW_SHOT_EPOCHS=100`
  - `EXTREME_WEIGHT_BETA_SELF=0.6`
  - `EXTREME_SOURCE_BORROW_BUDGET_GAMMA=0.75`

### 2026-04-08 - 已支持跳过 local_pretrain/local_meta，仅重跑 extreme 微调
- 在 `baseline-reset-9699391` 分支上，`DemoModelTraining.py` 已新增：
  - `SKIP_LOCAL_PRETRAIN`
  - `SKIP_LOCAL_META`
- 这两个开关的目的都是复用已存在的本地 checkpoint，避免在只想重试 `EXTREME_TARGET_REFINEMENT_EPOCHS` 或其他 extreme-stage 参数时，重复运行耗时的 `local_pretrain` / `local_meta`。
- 当前行为约束：
  - `SKIP_LOCAL_PRETRAIN=1` 时，程序会直接复用 `model_fore_pre_station{station}_local.pth`
  - `SKIP_LOCAL_META=1` 时，程序会直接复用 `model_fore_train_task_query_local_meta_station{station}.pth`
  - 若对应 checkpoint 不存在，会直接抛出 `FileNotFoundError`，不允许静默覆盖已有结果
- 因此，后续若仅想比较：
  - `EXTREME_TARGET_REFINEMENT_EPOCHS`
  - `EXTREME_WEIGHT_BETA_SELF`
  - `EXTREME_SOURCE_BORROW_BUDGET_GAMMA`
  - 或其他 extreme-stage 参数
  则推荐直接设置：
  - `SKIP_LOCAL_PRETRAIN=1`
  - `SKIP_LOCAL_META=1`
  只重跑 extreme 微调与最终评估。

### 2026-04-12 - six-station 2h/6p pilot-5k 已完成，下一步转向 federated normal meta
- 当前主工作目录固定为：
  - `/tmp/wpf-worktrees/restore-raw-data-83688a3/WPF-under-extreme-weather-main`
- 当前论文汇报口径已由用户明确改为 `6` 个目标站：
  - `58 / 59 / 60 / 61 / 62 / 63`
  - `61 / 62 / 63` 是原始三个物理场站的互补 2h 相位，但在本文实验中按新增真实目标站处理。
  - 后续结果分析默认使用 all-6 的 `Overall_Average`，不是只看原始 `58 / 59 / 60`。
- `2h / 6点 / 12h窗口` six-station `pilot-5k` 已在 4090 上完成：
  - train log: `logs/2h6p_six_station_pilot5k_train_20260412_001228.log`
  - all-6 eval log: `logs/2h6p_six_station_pilot5k_eval_all6_20260412_011201.log`
  - all-6 result CSV: `artifacts/2h6p_six_station/pilot-5k/results/multi_station_performance.csv`
  - model artifacts: `artifacts/2h6p_six_station/pilot-5k/models/`
  - produced `.pth` count: `90`
- all-6 Mean nMAE result:
  - `LMT = 28.8498`
  - `Extreme-FedAvg = 28.3501`
  - `Proposed-A = 28.3110`
  - `Proposed-A vs LMT = +1.87%` relative improvement, still below the hard target `>=5%`.
- Important interpretation:
  - Compared with the earlier strong `1h / 12点` baseline in `git 4.8.11: Proposed-A win all`, the current `2h / 6点 / 12h窗口` protocol has clearly degraded the `LMT` baseline and therefore successfully reduced effective information.
  - The remaining problem is no longer mainly “LMT too strong”; it is that `Proposed-A` has not recovered enough performance from the extra cross-station information.
  - The current `Proposed-A` only gets cross-station information mainly in the final extreme-weather adaptation/aggregation stage. That stage has limited leverage because it is still a small fine-tune step.
- Current code evidence:
  - `DemoModelTraining.py` still runs local normal-weather meta per station:
    - `run_meta_training(..., sample_station_ids=[station_id])`
  - Therefore `local_meta_station{s}` does not yet benefit from other stations' normal-weather tasks.
- Recommended next branch/task:
  - Implement a minimal `Fed-Normal-Meta Proposed-A` experiment.
  - Keep `LMT` unchanged: `local_pretrain_s -> local_meta_s -> target extreme fine-tune`.
  - Keep `Extreme-FedAvg` unchanged initially: use `local_meta_s` as the extreme aggregation base.
  - Change only `Proposed-A` initialization:
    - train `fed_normal_meta_station{s}.pth` for each target station `s`;
    - initialize it from `model_fore_pre_station{s}_local.pth`;
    - use normal-weather meta tasks from all `58/59/60/61/62/63` stations;
    - allow all non-self source stations, including same-physical complementary phase sources;
    - use this `fed_normal_meta_station{s}.pth` as the base for Proposed-A's extreme aggregation and target refinement.
  - Mechanism expectation: `LMT` remains target-only under reduced 2h/6p effective information, while `Proposed-A` gains a stronger cross-station normal-weather task prior before entering extreme few-shot adaptation. This has a more plausible path to `>=5%` than continuing to tune only final fine-tune epochs or K-shot settings.
- If `Fed-Normal-Meta Proposed-A` still only improves `Proposed-A vs LMT` by about `2%-3%`, then the next high-priority route should be stronger effective-information reduction, e.g. `3h / 4点 / 12h窗口`, rather than minor final-stage parameter tuning.
- Note on `orig3`:
  - `logs/2h6p_six_station_pilot5k_eval_orig3_20260412_011214.log` initially stopped before completion.
  - It was later regenerated eval-only with `OMP_NUM_THREADS=1`, producing:
    - `artifacts/2h6p_six_station/pilot-5k/results/multi_station_performance_orig3.csv`
  - Since the paper口径 is now all-6, `orig3` is secondary and should not override all-6 conclusions.
