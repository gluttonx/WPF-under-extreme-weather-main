# 🌍 核心角色设定
你现在的角色是专注于 **极端天气下风电功率预测 (WPF)** 的顶尖 AI 算法研究员与资深架构师。
你的首要目标是确保算法的严谨性、代码的高效性以及科研产出的高质量。

# 🎯 当前核心实验目标
- 当前阶段的硬目标不是让 `Proposed-A` “略优”或“多数情况下赢”，而是让 `Proposed-A` 相对 `LMT` 在主要误差指标上拉开**至少 5%-10% 的相对改善率**。
- 当前有效判据为：`relative_gap = (LMT_metric_% - Proposed_metric_%) / LMT_metric_% * 100`。`relative_gap >= 5%` 是最低目标，`5%-10%` 是理想区间。
- 2026-04-13 曾短暂改为“5.0 个绝对百分点”目标，但用户在 2026-04-14 明确放弃该口径，原因是过于严苛，难以仅靠联邦或轻量处理达到。后续只把绝对百分点差值作为辅助统计。
- 后续所有改进建议、实验设计和结果分析都必须优先回答：这个方案是否有合理机制让 `Proposed-A vs LMT` 的**相对改善率**达到或接近 `5%-10%`。若只是小幅参数微调，且缺乏把相对差距扩大到 5% 级别的机制，应明确降级为低优先级。
- 分析结果时必须单独报告 `Proposed-A vs LMT`，不能只报告 `Proposed-A vs Extreme-FedAvg` 或总体 win count。若 `Extreme-FedAvg` 受益更多但 `Proposed-A` 没有拉开 `LMT`，则该方向不满足当前主目标。
- 结果分析必须优先报告相对改善率，并可同时报告绝对差值：`Δ_abs = LMT_metric_% - Proposed_metric_%`。

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
### 2026-04-27 - New main protocol fixed as two-year train / 2024 test, with K6 as strict main table and all-extreme as ablation
- 本轮长会话已确认并固定新的主协议，不再继续沿用旧的 `2022 support / 2023 test` 六场站协议作为主线：
  - `train = 2022 + 2023`
  - `test = 2024`
  - `2h / 6-point / 12h window`
  - six-station 仍为 `58/59/60 + 61/62/63 phase augment`
- `2024` 三个 base workbook 的测试标幺值容量固定为：
  - `58 -> 50`
  - `59 -> 100`
  - `60 -> 300`
- 新协议 builder 已落地到代码，且不单独新建 `24jilin_58-60.py`；统一由 yearly protocol builder 在线处理 train/test workbook 与各自 capacity。
- normal-stage 的 few-shot 预算固定为：
  - `30-day-equivalent = 360` 个 `2h` points
  - 从 `22-23 normal` 池中做 `2022:180 + 2023:180` 的平衡抽样
- 论文主表当前采用严格 low-resource extreme 口径：
  - `EXTREME_SUPPORT_WINDOW_CAP = 6`
  - 即每站每类训练 extreme 最多 `6` 个 windows
- 同时保留 `all-extreme` 作为补充/诊断协议：
  - `--two-year-2024-all-extreme`
  - 仅用于判断 `K=6` 截断是否是主矛盾，不作为当前主表口径。

### 2026-04-27 - Normal meta under the new protocol fixed to k=5 and 3/3 shots after elbow + feasibility analysis
- 用户曾质疑新的 normal meta 不能直接沿用例文的 `k=10, support/query=10/10`；本轮已先做数据分析再定参数。
- 分析口径：
  - 先在 `22-23` 全 normal pool、按 `2h` 协议下采样后做 station-wise elbow；
  - 纯 elbow 倾向于 `58/61 -> 7`, `59/62 -> 5~6`, `60/63 -> 6`；
  - 但在 `360-point` capped normal budget 下，`k>=6` 会让 many-shot meta task 过薄，不适合作为 first-run 公共配置。
- 因此当前主协议下的 normal meta 参数固定为：
  - `k = 5`
  - `k* = 5`
  - `META_SUPPORT_SHOTS = 3`
  - `META_QUERY_SHOTS = 3`
- 训练脚本已改成真正读取 `META_SUPPORT_SHOTS / META_QUERY_SHOTS`，不再硬编码 `10/10`。

### 2026-04-27 - The main bottleneck under the new protocol is class-specific transferability, not Fed-Normal-Meta and not K=6 truncation
- 在 `two_year_24_k6` 协议下，先后跑过：
  1. 不带 `Fed-Normal-Meta` 的 `LMT / Extreme-FedAvg / Proposed-A`
  2. 带 `Fed-Normal-Meta` 主干的对应版本
  3. `all-extreme` 对照，只重跑 extreme 阶段
- 已验证并确认的结论：
  - `Fed-Normal-Meta` 在新协议下没有把 base initialization 做得更好；它不是当前收益来源，反而可能引入额外噪声。
  - `K=6 -> all-extreme` 的 ablation 没有把整体结果救回来，因此 `K=6` 不是当前主矛盾。
  - 当前更稳定的解释是 **class-specific transferability / negative transfer**：
    - `HighWind / Class1`：联邦迁移最容易出问题，属于当前主负迁移来源。
    - `HighTemp / Class2`：联邦通常有正收益。
    - `ColdWave / Class3`：最不稳定，当前机制下常常不值得强行联邦。
    - `Frost / Class4`：联邦通常有正收益。
- 结果分析口径也已固定：
  - 不能只看 `Overall_Average`
  - 必须同时看 `Overall_SampleWeighted`
  - 必须按 class 拆开，而不是只看 overall。

### 2026-04-27 - CA-FEMR is deferred; current next step stays inside the existing Proposed-A family with selective class-wise controls
- 用户曾考虑直接切到新的 `CA-FEMR`（Class-Adaptive Federated Extreme Meta-Refinement）。
- 本轮最终判断：**暂不切 CA-FEMR**。
- 原因已固定为：
  - 当前问题已经定位到“哪些 class 该联邦、哪些 class 不该联邦”的负迁移控制，而不是缺一个全新主干。
  - 现在直接上 CA-FEMR 会把变量重新搅混，无法清楚判断收益来自新机制还是来自更合理的 class-wise 策略。
- 当前优先级高于 CA-FEMR 的方向是：
  - 在现有 `LMT / Extreme-FedAvg / Proposed-A` 框架内做 **selective federation**；
  - 特别是 `Class1` 的更强 source rejection / gate；
  - `Class3` 允许保守 local fallback；
  - `Class2 / Class4` 尽量保留联邦收益，不要被过度 fallback。

### 2026-04-27 - Class-wise extreme transfer controls are implemented in code; first classwise-gate run only achieved partial damage control
- 本轮已在 `DemoModelTraining.py` 中实现 per-class extreme transfer controls，可通过环境变量直接配置：
  - `EXTREME_WEIGHT_BETA_SELF_BY_CLASS`
  - `EXTREME_SOURCE_HARD_GATE_BY_CLASS`
  - `EXTREME_SOURCE_MIN_TARGET_GAIN_BY_CLASS`
  - `EXTREME_SOURCE_GAIN_WEIGHT_ETA_BY_CLASS`
  - `EXTREME_SOURCE_TOP_K_BY_CLASS`
  - `EXTREME_FORCE_LOCAL_FALLBACK_BY_CLASS`
  - `EXTREME_PROPOSED_VAL_FALLBACK_BY_CLASS`
  - `EXTREME_PROPOSED_VAL_FALLBACK_MARGIN_BY_CLASS`
- 相关训练代码已支持：
  - per-class top-k source selection
  - per-class hard gate / gain weighting
  - per-class direct fallback to LMT
  - per-class proposed validation fallback
- 第一轮 `pilot-1k-classwise-gate` 已跑完，结论固定为：
  - `Class3 / ColdWave` 的 local fallback 成功去掉了该类的负迁移；
  - `Extreme-FedAvg` 相对原始 `K6` 版有小幅改善；
  - `Proposed-A` 反而变差，因为 validation fallback 误伤了本来有联邦收益的 `Class2 / Class4`。
- 因此当前推荐的后续修正不是“推翻 class-wise”，而是：
  - 继续保留 `Class3` 的保守策略；
  - 对 `Class1` 使用更强 rejection，而不只是 top-k；
  - 对 `Class2 / Class4` 减少或关闭 `Proposed-A` 的 fallback_lmt 触发。

### 2026-04-16 - 2h6p self08 equal-budget meta10k final checkpoint did not improve target gap
- 实验：复用 `2h6p_six_station/fed-normal-meta-5k` 的 6 个 local-pretrain final checkpoint，重新训练 local meta `10k` 与 Fed-Normal-Meta `10k`，extreme-stage 保持 `FEW_SHOT_EPOCHS=200`、`EXTREME_TARGET_REFINEMENT_EPOCHS=200`。
- 产物：
  - Train log: `logs/2h6p_six_station_fed_normal_meta_self08_meta10k_finalckpt_finetune200_20260414_151657_train.log`
  - Eval log: `logs/2h6p_six_station_fed_normal_meta_self08_meta10k_finalckpt_finetune200_20260414_151657_eval_all6.log`
  - CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-meta10k-finalckpt-finetune200_151657/results/multi_station_performance.csv`
- 数据完整性：
  - `skip_local_pretrain=True`、`skip_local_meta=False`、`skip_fed_normal_meta=False`、`fed_normal_meta_save_best=True`、`fed_normal_meta_use_best=False`。
  - metric NaN 数为 0；`training_convergence_report.json` 中 `non_finite_loss=0`、`non_finite_state=0`。
  - 6 个 Fed-Normal-Meta best checkpoint 已保存，但本次 extreme 结果仍使用 final checkpoint。
- Overall_Average 相对改善率（Proposed-A vs LMT）相对 5k final200 下降：
  - `nMAE`: `3.08% -> 1.83%`
  - `nRMSE`: `3.63% -> 2.03%`
  - `WD`: `1.19% -> 0.85%`
- 分天气 nMAE：
  - HighWind 仍达 `5.43%`，但低于 5k final200 的 `7.27%`。
  - HighTemperature 为 `-0.28%`，Frost 为 `-0.37%`，即 Proposed-A 反而弱于 LMT。
  - ColdWave 仅 `1.35%`，低于 5k final200 的 `2.49%`。
- 关键解释：
  - 10k meta 使三条方法的绝对误差都下降，但 `LMT` 与 `Extreme-FedAvg` 受益更多。
  - Compared with 5k final200, mean absolute improvements were:
    - LMT: `nMAE -1.4533`, `nRMSE -1.6265`, `WD -1.0092`
    - Proposed-A: `nMAE -1.0695`, `nRMSE -1.0755`, `WD -0.9157`
  - 因此等预算增加 normal-meta 步数没有把 `Proposed-A vs LMT` 推向 `5%-10%`，反而压缩了差距。
- 结论：
  - `2h6p self08 5k final200` 仍是当前最强主线结果。
  - 后续不应优先继续增加 local/Fed-Normal-Meta 轮数；需要转向能限制 LMT target adaptation 或改变 Proposed-A 信息优势/transfer 机制的方向。

### 2026-04-14 - 2h6p self08 final base + 500 extreme fine-tuning shrinks Proposed-A gap
- 实验：复用 `2h6p_six_station/fed-normal-meta-5k` 的 local pretrain/local meta/Fed-Normal-Meta final checkpoint，不使用 best checkpoint，只把 extreme-stage `FEW_SHOT_EPOCHS` 与 `EXTREME_TARGET_REFINEMENT_EPOCHS` 从 `200` 提到 `500`。
- 产物：
  - Train log: `logs/2h6p_six_station_fed_normal_meta_self08_5k_finetune500_20260414_133046_train.log`
  - Eval log: `logs/2h6p_six_station_fed_normal_meta_self08_5k_finetune500_20260414_142915_eval_all6.log`
  - CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-5k-finetune500/results/multi_station_performance.csv`
- 数据完整性：
  - 72 个 extreme personalized `.pth` 均由本次 run 写入，时间连续覆盖 `2026-04-14 13:31-14:29`。
  - metric NaN 数为 0；`training_convergence_report.json` 中 `non_finite_loss=0`、`non_finite_state=0`。
  - `Training_duration_s=NaN` 属于复用 base checkpoint 后只重跑 extreme 阶段的预期表现。
- Overall_Average 相对改善率（Proposed-A vs LMT）：
  - `nMAE`: `3.08% -> 2.00%`
  - `nRMSE`: `3.63% -> 2.23%`
  - `WD`: `1.19% -> 0.56%`
- 关键变化：
  - 500 轮使三种方法绝对误差均下降，但 `LMT` 与 `Extreme-FedAvg` 下降更多，导致 `Proposed-A` 相对 `LMT` 的 gap 缩小。
  - HighWind nMAE gap 从 `7.27%` 降到 `3.61%`；ColdWave nMAE gap 仅小幅从 `2.49%` 到 `2.72%`。
  - `Proposed-A` vs `Extreme-FedAvg` 的 Overall gap 也明显收缩，且 HighTemperature/ColdWave nMAE 已略弱于 `Extreme-FedAvg`。
- 结论：
  - 单纯增加 extreme fine-tuning/refinement 轮数不是通向 `5%-10%` 相对改善目标的主线。
  - 更长 target adaptation 会削弱初始化/联邦 normal-meta 的优势，使 LMT 追平更多；后续不应优先继续加 few-shot/refinement 轮数。
  - 下一步应转向等预算 meta continuation 或改进 normal-meta/transfer 机制，而不是继续把 extreme-stage 轮数加大。

### 2026-04-14 - Supersedes absolute-point target: return to relative 5%-10% gap
- 用户重新确认：放弃 2026-04-13 的“5.0 个绝对百分点”目标，因为该目标太严苛，难以仅通过联邦或轻量处理达到。
- 当前目标恢复为原口径：`Proposed-A` 相对 `LMT` 在主要误差指标上达到 `5%-10%` 的相对改善率。
- 当前有效公式：
  - `relative_gap = (LMT_metric_% - Proposed_metric_%) / LMT_metric_% * 100`
  - `relative_gap >= 5%` 为最低目标。
- 后续结果分析仍可报告绝对差值 `LMT_metric_% - Proposed_metric_%`，但它不是当前达标判据。
- This supersedes the 2026-04-13 absolute percentage-point target note.

### 2026-04-13 - Superseded: 5% target is absolute percentage-point gap, not relative improvement
- Superseded again on 2026-04-14: 用户放弃绝对 5.0 个百分点目标，恢复为相对 `5%-10%` 改善率目标。本节仅保留历史记录。
- 用户明确纠正目标口径：`Proposed-A vs LMT >= 5%` 指的是结果表中标幺化百分数指标的绝对差值，即 `LMT_metric_% - Proposed_metric_% >= 5.0`。
- 这不是相对改善率。此前若将 `+3.08%`、`+3.35%` 等写成相对提升，只能作为辅助统计，不能作为论文目标达标判据。
- 后续评估必须优先报告：
  - `Δ_abs_nMAE_points = LMT_nMAE_% - Proposed_nMAE_%`
  - `Δ_abs_nRMSE_points = LMT_nRMSE_% - Proposed_nRMSE_%`
  - `Δ_abs_WD_points = LMT_WD_% - Proposed_WD_%`
- 示例：若 `LMT HighWind_nMAE_%` 为 `30+`，则 `Proposed-A` 需要进入 `20+` 区间并至少低 `5.0` 个百分点；按相对改善率理解会严重低估目标难度。
- 该条 supersedes 2026-04-11 以及后续结果记录中关于“5% relative improvement / 相对差距”的目标解释。旧数值记录保留，但后续决策必须按绝对百分点差值重算。

### 2026-04-11 - 当前主目标更新为 Proposed-A 相对 LMT 至少 5% 差距
- Superseded on 2026-04-13: 本节中的“相对提升 / 相对差距”口径已失效。正确口径是结果表内 `metric_%` 的绝对百分点差值 `LMT - Proposed >= 5.0`。
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

### 2026-04-12 - Fed-Normal-Meta Proposed-A implemented for six-station 2h/6p
- Implemented the minimal `Fed-Normal-Meta Proposed-A` path in `DemoModelTraining.py`.
- The design is target-conditioned FedAvg-style normal-weather meta learning, not a single pooled meta model:
  - for each target station `s`, initialize from `model_fore_pre_station{s}_local.pth`;
  - broadcast the target state to all six stations;
  - each client performs one normal-weather support/query meta update on its own tasks;
  - aggregate client state dicts with a target self floor (`FED_NORMAL_META_SELF_FLOOR`, default `0.3`) and source weights proportional to normal-meta window counts;
  - save `model_fore_train_task_query_fed_normal_meta_station{s}.pth`.
- Model-role separation:
  - `LMT` still uses the local-meta base;
  - `Extreme-FedAvg` still uses the local-meta base;
  - `Proposed-A` uses the fed-normal-meta base only when `ENABLE_FED_NORMAL_META_PROPOSED=1`.
- New control flags:
  - `ENABLE_FED_NORMAL_META_PROPOSED=1`
  - `FED_NORMAL_META_SELF_FLOOR=0.3`
  - `SKIP_FED_NORMAL_META=1` for reusing existing fed-normal-meta checkpoints.
- Smoke verification completed on CPU with:
  - artifact dir: `artifacts/2h6p_six_station/fed-normal-meta-smoke/`
  - command: `ARTIFACT_DIR=artifacts/2h6p_six_station/fed-normal-meta-smoke ENABLE_FED_NORMAL_META_PROPOSED=1 FED_NORMAL_META_SELF_FLOOR=0.3 python -u run_three_station_yearly_protocol.py train --smoke --six-station`
  - eval command: `ARTIFACT_DIR=artifacts/2h6p_six_station/fed-normal-meta-smoke ENABLE_FED_NORMAL_META_PROPOSED=1 FED_NORMAL_META_SELF_FLOOR=0.3 python -u run_three_station_yearly_protocol.py eval --smoke --six-station`
  - produced `102` `.pth` files, including `12` fed-normal-meta support/query checkpoints and the expected `72` LMT/Extreme-FedAvg/Proposed-A extreme checkpoints.
  - generated: `artifacts/2h6p_six_station/fed-normal-meta-smoke/results/multi_station_performance.csv`.
- Regression verification passed:
  - `python -m unittest tests.test_extreme_fl_contract_ast tests.test_training_protocol_config_ast tests.test_skip_stage_reuse_ast tests.test_2h_6point_launcher_ast -v`
  - `python -m py_compile DemoModelTraining.py generate_multi_station_results.py run_three_station_yearly_protocol.py`
  - `git diff --check`
- Next formal experiment should run on 4090, not CPU. Recommended first formal run:
  - `ARTIFACT_DIR=artifacts/2h6p_six_station/fed-normal-meta-5k ENABLE_FED_NORMAL_META_PROPOSED=1 FED_NORMAL_META_SELF_FLOOR=0.3 python -u run_three_station_yearly_protocol.py train --preset pilot-5k --six-station`
  - then same env with `eval --preset pilot-5k --six-station`.

### 2026-04-12 - Fed-Normal-Meta self_floor=0.8 pilot-5k result
- Formal 4090 six-station 2h/6p pilot-5k run completed with:
  - `ENABLE_FED_NORMAL_META_PROPOSED=1`
  - `FED_NORMAL_META_SELF_FLOOR=0.8`
  - `ARTIFACT_DIR=artifacts/2h6p_six_station/fed-normal-meta-5k`
  - train log: `logs/2h6p_six_station_fed_normal_meta_self08_5k_train_20260412_155932.log`
  - eval log: `logs/2h6p_six_station_fed_normal_meta_self08_5k_eval_all6_20260412_193528.log`
  - result CSV: `artifacts/2h6p_six_station/fed-normal-meta-5k/results/multi_station_performance.csv`
- This run reused local pretrain/local meta checkpoints and reran Fed-Normal-Meta plus extreme adaptation.
- All-6 Overall_Average Mean nMAE:
  - `LMT = 28.7320`
  - `Extreme-FedAvg = 28.3359`
  - `Proposed-A = 27.8476`
  - `Proposed-A vs LMT = +3.08%`
  - `Proposed-A vs Extreme-FedAvg = +1.72%`
- Compared with the previous all-6 pilot-5k:
  - old `Proposed-A = 28.3109`
  - self08 `Proposed-A = 27.8476`
  - absolute gain: `-0.4633` Mean nMAE
  - relative gain vs old Proposed-A: about `1.64%`.
- Hard target still not met:
  - current-run 5% threshold from LMT is `27.2954`;
  - self08 Proposed-A is `27.8476`, still `+0.5523` above threshold.
- Improvement is uneven:
  - HighWind improves strongly vs LMT: `7.27%`;
  - HighTemperature only `0.60%`;
  - ColdWave `2.49%`;
  - Frost `0.90%`.
- Per-station Mean nMAE Proposed-A vs LMT:
  - station 58: `-0.03%` (slightly worse)
  - station 59: `+8.14%`
  - station 60: `+3.16%`
  - station 61: `+1.32%`
  - station 62: `+6.43%`
  - station 63: `+3.00%`
- Interpretation: self_floor=0.8 is a real improvement over self_floor=0.3/old Proposed-A direction, but still not enough for the 5% claim. The next most plausible low-risk lever is adding Fed-Normal-Meta best-checkpoint restore / smoothing, because all six fed-normal-meta stages had final loss worse than best loss.

### 2026-04-12 - Fed-Normal-Meta restore-best switch added
- Added `FED_NORMAL_META_RESTORE_BEST=1` to restore the best weighted-query Fed-Normal-Meta checkpoint at the end of each target station's Fed-Normal-Meta stage.
- Scope:
  - only affects `fed_normal_meta_station{s}` checkpoints used by `Proposed-A`;
  - does not alter `LMT` or `Extreme-FedAvg` local-meta bases;
  - records `restored_best_checkpoint`, `restored_best_epoch`, and `restored_best_loss` in the convergence report.
- This is not an extreme-only rerun. To test it, rerun Fed-Normal-Meta plus extreme adaptation, while reusing local pretrain/local meta:
  - set `SKIP_LOCAL_PRETRAIN=1`
  - set `SKIP_LOCAL_META=1`
  - do not set `SKIP_FED_NORMAL_META=1`
- Recommended command:
  - `ARTIFACT_DIR=artifacts/2h6p_six_station/fed-normal-meta-restore-best-5k ENABLE_FED_NORMAL_META_PROPOSED=1 FED_NORMAL_META_SELF_FLOOR=0.8 FED_NORMAL_META_RESTORE_BEST=1 SKIP_LOCAL_PRETRAIN=1 SKIP_LOCAL_META=1 python -u run_three_station_yearly_protocol.py train --preset pilot-5k --six-station`
  - then same env with `eval --preset pilot-5k --six-station`.

### 2026-04-13 - 4h/3p six-station half-phase protocol and separated Fed-Normal-Meta best checkpoint
- Supersedes the checkpoint behavior in the previous `FED_NORMAL_META_RESTORE_BEST` note: the current implementation no longer overwrites the 5000-step final Fed-Normal-Meta checkpoint when saving best.
- Current checkpoint semantics:
  - final Fed-Normal-Meta checkpoint is always kept at `model_fore_train_task_query_fed_normal_meta_station{s}.pth`;
  - best weighted-query checkpoint is saved separately when `FED_NORMAL_META_SAVE_BEST=1` or `FED_NORMAL_META_USE_BEST=1`;
  - best checkpoint path is `model_fore_train_task_query_fed_normal_meta_best_station{s}.pth`;
  - `FED_NORMAL_META_USE_BEST=1` makes `Proposed-A` initialize from the separate best checkpoint;
  - legacy `FED_NORMAL_META_RESTORE_BEST=1` is kept as a backward-compatible alias for save+use best.
- Added a 4h/3p six-station protocol path:
  - launcher flag: `--four-hour`;
  - protocol name: `six_station_4h_3point_phase_augmented_protocol`;
  - data dir: `protocol_data/4h_3p_six_station`;
  - artifact root: `artifacts/4h3p_six_station`;
  - `SAMPLE_INTERVAL_HOURS=4`, `LEN_REALP=3`, `POINTS_PER_DAY=6`, `WINDOW_SPAN_HOURS=12`;
  - base stations `58/59/60` use `DOWNSAMPLE_OFFSET=1`;
  - complement stations `61/62/63` use half-phase `PHASE_AUGMENT_COMPLEMENTARY_OFFSET=3`.
- Generated and verified `protocol_data/4h_3p_six_station`:
  - station offsets: `58/59/60 -> 1`, `61/62/63 -> 3`;
  - metadata reports `sample_interval_hours=4`, `len_realp=3`, `points_per_day=6`, `window_span_hours=12`.
- Recommended 4090 pilot command:
  - `ENABLE_FED_NORMAL_META_PROPOSED=1 FED_NORMAL_META_SELF_FLOOR=0.8 FED_NORMAL_META_SAVE_BEST=1 FED_NORMAL_META_USE_BEST=1 python -u run_three_station_yearly_protocol.py train --preset pilot-5k --four-hour`
  - then same env with `eval --preset pilot-5k --four-hour`.
- Important: this 4h/3p setting changes all training-stage data, so it cannot reuse 2h/6p local pretrain/local meta checkpoints. Run all training stages from scratch unless a matching `artifacts/4h3p_six_station/...` checkpoint set already exists.

### 2026-04-13 - Hybrid 4h3p normal base + 2h6p extreme rerun path
- The 4h/3p self08-best pilot-5k eval produced `NaN` values in some HighTemperature/ColdWave metrics because several extreme few-shot classes had too few 4h/3p windows.
- Root cause fixed in `DemoModelTraining.py`:
  - for one-window extreme support splits, keep the full `len_realp` window for adaptation and leave validation empty instead of slicing the horizon down to `T=1`;
  - add non-finite loss/state guards in few-shot adaptation and restore the last finite state before saving.
- Added separate `BASE_MODEL_OUTPUT_DIR` so extreme reruns can write new few-shot checkpoints under a new artifact dir while reading normal-stage base checkpoints from an existing run.
- Added launcher flag `--hybrid-extreme-2h`:
  - protocol name: `hybrid_4h3p_normal_2h6p_extreme_protocol`;
  - normal/base checkpoints default to `artifacts/4h3p_six_station/fed-normal-meta-self08-best-5k/models`;
  - extreme support/test data use `protocol_data/2h_6p_six_station`;
  - output goes to `artifacts/4h3p_normal_2h6p_extreme/pilot-5k`;
  - it forces `SKIP_LOCAL_PRETRAIN=1`, `SKIP_LOCAL_META=1`, and `SKIP_FED_NORMAL_META=1`.
- This path is intended to test whether the reduced-information 4h/3p normal prior can help `Proposed-A`, while avoiding the 4h/3p extreme-event sparsity that caused NaN few-shot checkpoints.
- Verification completed:
  - `python -m unittest discover -s tests -p 'test_*.py' -v`
  - `python -m py_compile DemoModelTraining.py run_three_station_yearly_protocol.py generate_multi_station_results.py build_three_station_extreme_yearly_protocol.py`
  - `run_three_station_yearly_protocol.py train/eval --preset pilot-5k --hybrid-extreme-2h --dry-run`
  - `git diff --check`

### 2026-04-13 - Hybrid 4h3p normal base + 2h6p extreme eval result
- Completed hybrid run:
  - train log: `logs/hybrid_4h3p_normal_2h6p_extreme_self08_best_5k_train_20260413_164108.log`
  - eval log: `logs/hybrid_4h3p_normal_2h6p_extreme_self08_best_5k_eval_all6_20260413_170655.log`
  - result CSV: `artifacts/4h3p_normal_2h6p_extreme/pilot-5k/results/multi_station_performance.csv`
- Data quality:
  - result table has no NaN in nMAE/nRMSE/WD metrics;
  - `Training_duration_s=NaN` is expected for eval output and should not be treated as metric failure;
  - `training_convergence_report.json` reports `non_finite_loss=0` and `non_finite_state=0`.
- Hybrid all-6 Overall_Average mean metrics:
  - `LMT = 29.2431 nMAE / 33.0666 nRMSE / 25.7096 WD`
  - `Extreme-FedAvg = 28.2558 nMAE / 31.9173 nRMSE / 25.1822 WD`
  - `Proposed-A = 28.2624 nMAE / 31.9255 nRMSE / 25.0195 WD`
  - `Proposed-A vs LMT = +3.35% nMAE / +3.45% nRMSE / +2.68% WD`
  - `Proposed-A vs Extreme-FedAvg = -0.02% nMAE / -0.03% nRMSE / +0.65% WD`
- Compared with 2026-04-12 `Fed-Normal-Meta_six_station_pilot5k` (`artifacts/2h6p_six_station/fed-normal-meta-5k/results/multi_station_performance.csv`):
  - previous `Proposed-A vs LMT = +3.08% nMAE / +3.63% nRMSE / +1.19% WD`;
  - hybrid only improves the nMAE gap by about `+0.27` percentage points, and worsens the nRMSE gap by about `-0.18` percentage points;
  - hybrid `Proposed-A` absolute error is worse in mean nMAE (`+0.4147`, `+1.49%`) and mean nRMSE (`+0.7256`, `+2.33%`), though WD is better (`-0.3721`, `-1.47%`);
  - the apparent nMAE gap increase mostly comes from hybrid `LMT` becoming worse (`+0.5112`, `+1.78%`), not from `Proposed-A` becoming stronger.
- Conclusion:
  - Hybrid 4h3p normal base + 2h6p extreme is numerically valid and fixes the 4h3p extreme NaN problem, but it does not meet the hard `>=5%` Proposed-A vs LMT goal.
  - It should not be the main next route unless used only as supporting ablation. The more promising next direction is a stronger mechanism that improves Proposed-A itself, not just weakening LMT.

### 2026-04-13 - Next run: 2h6p self08 Fed-Normal-Meta save/use best
- Next recommended experiment returns to the best current 2h/6p all-6 mainline and tests only separate Fed-Normal-Meta best checkpoint usage:
  - base local checkpoints: `artifacts/2h6p_six_station/fed-normal-meta-5k/models`
  - output artifact dir: `artifacts/2h6p_six_station/fed-normal-meta-self08-best-5k`
  - settings: `ENABLE_FED_NORMAL_META_PROPOSED=1`, `FED_NORMAL_META_SELF_FLOOR=0.8`, `FED_NORMAL_META_SAVE_BEST=1`, `FED_NORMAL_META_USE_BEST=1`
  - skip only local stages: `SKIP_LOCAL_PRETRAIN=1`, `SKIP_LOCAL_META=1`
  - do not skip Fed-Normal-Meta: `SKIP_FED_NORMAL_META=0`
- Implementation detail fixed before this run:
  - `BASE_MODEL_OUTPUT_DIR` may now differ from `MODEL_OUTPUT_DIR` when local pretrain/local meta are skipped, even if Fed-Normal-Meta is rerun;
  - local pretrain/local meta helpers read from `BASE_MODEL_OUTPUT_DIR`;
  - Fed-Normal-Meta helpers write to `MODEL_OUTPUT_DIR` when `SKIP_FED_NORMAL_META=0`, and read from `BASE_MODEL_OUTPUT_DIR` only when `SKIP_FED_NORMAL_META=1`.
- Verification completed:
  - RED contract test failed before the path fix;
  - `python -m unittest discover -s tests -p 'test_*.py' -v` passed with 58 tests;
  - `python -m py_compile DemoModelTraining.py run_three_station_yearly_protocol.py generate_multi_station_results.py` passed;
  - train/eval dry-runs confirmed the intended 2h/6p protocol, old base dir, new output dir, local skip flags, and `SKIP_FED_NORMAL_META=0`.

### 2026-04-14 - 2h6p self08 Fed-Normal-Meta save/use-best result
- Completed run:
  - train log: `logs/2h6p_six_station_fed_normal_meta_self08_saveusebest_5k_train_20260413_194520.log`
  - eval log: `logs/2h6p_six_station_fed_normal_meta_self08_saveusebest_5k_eval_all6_20260413_232959.log`
  - result CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-best-5k/results/multi_station_performance.csv`
- Data quality:
  - result table has no NaN in nMAE/nRMSE/WD metrics;
  - `Training_duration_s=NaN` is expected for eval output;
  - `training_convergence_report.json` reports `non_finite_loss=0` and `non_finite_state=0`.
- The run correctly reused local pretrain/local meta from `artifacts/2h6p_six_station/fed-normal-meta-5k/models`, reran Fed-Normal-Meta, saved best checkpoints, and used best checkpoints for `Proposed-A`.
- All six Fed-Normal-Meta stages had best loss substantially below final loss, but this did not translate into a better Proposed-A vs LMT gap.
- All-6 Overall_Average mean metrics:
  - `LMT = 27.8665 nMAE / 31.4657 nRMSE / 24.9162 WD`
  - `Extreme-FedAvg = 27.2456 nMAE / 30.6690 nRMSE / 24.6863 WD`
  - `Proposed-A = 27.4346 nMAE / 30.8408 nRMSE / 24.5236 WD`
  - `Proposed-A vs LMT = +1.55% nMAE / +1.99% nRMSE / +1.58% WD` by relative-improvement target formula.
  - `Proposed-A vs Extreme-FedAvg = -0.69% nMAE / -0.56% nRMSE / +0.66% WD`.
- Compared with 2026-04-12 `Fed-Normal-Meta self08 final` (`artifacts/2h6p_six_station/fed-normal-meta-5k/results/multi_station_performance.csv`):
  - previous `Proposed-A vs LMT = +3.08% nMAE / +3.63% nRMSE / +1.19% WD`;
  - save/use-best worsens the nMAE and nRMSE relative gap, and improves WD only slightly;
  - `Proposed-A` absolute mean nMAE improves from `27.8476` to `27.4346`, but `LMT` and `Extreme-FedAvg` improve more, so the relative gap shrinks.
- Per-weather nMAE:
  - HighWind: `+5.31%` vs LMT, but weaker than previous `+7.27%`;
  - HighTemperature: `+0.58%`, still weak;
  - ColdWave: `-0.97%`, Proposed-A becomes worse than LMT;
  - Frost: `+0.75%`, still weak.
- Conclusion:
  - `FED_NORMAL_META_SAVE_BEST=1` + `FED_NORMAL_META_USE_BEST=1` should not replace the 2026-04-12 self08 final checkpoint as the main result.
  - Best-checkpoint selection on normal-meta weighted query loss is not aligned enough with final extreme-weather performance; it improves absolute errors for all methods in this artifact but erodes the Proposed-A vs LMT/FedAvg advantage.

### 2026-04-17 - Next mechanism: target-constrained, transfer-preserving extreme adaptation
- Rationale:
  - More extreme fine-tuning epochs and longer Fed-Normal-Meta did not widen the Proposed-A vs LMT gap enough.
  - Event/class-aware Fed-Normal-Meta is not the preferred next route because the meta stage uses normal-weather data; the stronger lever is the extreme adaptation stage, where source borrowing can directly hurt or help target validation.
- Implemented mechanism:
  - `EXTREME_ANCHOR_REG_LAMBDA` adds optional proximal anchor regularization during `adapt_state_dict`: `loss_total = target_MSE + lambda * ||theta - theta_anchor||^2`.
  - Anchor defaults to the adaptation base state; target-refinement anchors to the aggregate state.
  - `EXTREME_SOURCE_HARD_GATE=1` enables a hard target-validation gate for source updates.
  - `EXTREME_SOURCE_MIN_TARGET_GAIN` requires the source-adapted state to beat the target self/reference validation loss by at least this margin before the source is included.
  - Defaults preserve old behavior: `EXTREME_ANCHOR_REG_LAMBDA=0.0`, `EXTREME_SOURCE_HARD_GATE=0`, `EXTREME_SOURCE_MIN_TARGET_GAIN=0.0`.
- Recommended first run:
  - reuse `artifacts/2h6p_six_station/fed-normal-meta-5k/models`;
  - skip local pretrain/local meta/Fed-Normal-Meta reruns;
  - keep `FEW_SHOT_EPOCHS=200`, `EXTREME_TARGET_REFINEMENT_EPOCHS=200`;
  - set `EXTREME_TARGET_ADAPT_MAX_WINDOWS=1`, `EXTREME_ANCHOR_REG_LAMBDA=0.001`, `EXTREME_SOURCE_HARD_GATE=1`, `EXTREME_SOURCE_MIN_TARGET_GAIN=0.0`.
- Verification:
  - `python -m unittest tests/test_extreme_fl_contract_ast.py` passed with 4 tests.
  - `python -m unittest tests/test_extreme_fl_contract_ast.py tests/test_target_kshot_refine_ast.py tests/test_skip_stage_reuse_ast.py` passed with 9 tests.
  - `python -m unittest discover -s tests` passed with 59 tests.
  - `python -m py_compile DemoModelTraining.py run_three_station_yearly_protocol.py` passed.

### 2026-04-17 - 2h6p self08 5k K1 anchor+gate result
- Completed run:
  - train log: `logs/2h6p_six_station_self08_5k_k1_anchor1e3_gate_train_20260417_184010.log`
  - eval log: `logs/2h6p_six_station_self08_5k_k1_anchor1e3_gate_eval_all6_20260417_191339.log`
  - result CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-5k-k1-anchor1e3-gate/results/multi_station_performance.csv`
- Config:
  - reused `artifacts/2h6p_six_station/fed-normal-meta-5k/models`;
  - `SKIP_LOCAL_PRETRAIN=1`, `SKIP_LOCAL_META=1`, `SKIP_FED_NORMAL_META=1`;
  - `ENABLE_FED_NORMAL_META_PROPOSED=1`, `FED_NORMAL_META_SELF_FLOOR=0.8`, `FED_NORMAL_META_USE_BEST=0`;
  - `EXTREME_TARGET_ADAPT_MAX_WINDOWS=1`, `EXTREME_ANCHOR_REG_LAMBDA=0.001`, `EXTREME_SOURCE_HARD_GATE=1`, `EXTREME_SOURCE_MIN_TARGET_GAIN=0.0`;
  - `FEW_SHOT_EPOCHS=200`, `EXTREME_TARGET_REFINEMENT_EPOCHS=200`.
- Data quality:
  - result table has no NaN/non-finite values in nMAE/nRMSE/WD metrics;
  - convergence report has `non_finite_loss=0`, `non_finite_state=0`;
  - train log shows hard gate active (`[hard_gate:...]` count 29), anchor logging active, and no `non-finite`.
- All-6 Overall_Average mean metrics:
  - `LMT = 28.8076 nMAE / 32.6607 nRMSE / 25.0829 WD`
  - `Extreme-FedAvg = 28.0283 nMAE / 31.7434 nRMSE / 24.7144 WD`
  - `Proposed-A = 28.0797 nMAE / 31.6948 nRMSE / 24.9440 WD`
  - `Proposed-A vs LMT = +2.53% nMAE / +2.96% nRMSE / +0.55% WD` by ratio-of-means relative formula.
  - `Proposed-A vs Extreme-FedAvg = -0.18% nMAE / +0.15% nRMSE / -0.93% WD`.
- Compared with 2026-04-12 `2h6p self08 5k final200`:
  - previous `Proposed-A vs LMT = +3.08% nMAE / +3.63% nRMSE / +1.19% WD`;
  - K1 anchor+gate shrinks the overall gap and should not replace the main result.
- Per-weather signal:
  - HighWind improves strongly: `Proposed-A vs LMT` reaches `+9.72% nMAE / +9.54% nRMSE / +2.62% WD`, better than the previous HighWind nMAE gap (`+7.27%`).
  - Frost collapses under global K=1: `Proposed-A vs LMT = -0.24% nMAE / -0.24% nRMSE / -0.83% WD`, and Proposed-A absolute Frost metrics are much worse than the 2026-04-12 mainline.
  - HighTemperature remains near neutral and ColdWave remains modest.
- Conclusion:
  - The mechanism is not globally successful when `EXTREME_TARGET_ADAPT_MAX_WINDOWS=1` is applied to every class.
  - The useful signal is class-dependent: K1+anchor+gate is promising for HighWind, while Frost needs more target adaptation windows.
  - Next recommended route is class-adaptive target caps, e.g. K1 for HighWind and all/uncapped windows for Frost, or first isolate anchor+gate with no global K cap.

### 2026-04-17 - 2h6p self08 5k no-K-cap anchor+gate result
- Completed run:
  - train log: `logs/2h6p_six_station_self08_5k_anchor1e3_gate_nokcap_train_20260417_195037.log`
  - eval log: `logs/2h6p_six_station_self08_5k_anchor1e3_gate_nokcap_eval_all6_20260417_202232.log`
  - result CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-5k-anchor1e3-gate-nokcap/results/multi_station_performance.csv`
- Config:
  - reused `artifacts/2h6p_six_station/fed-normal-meta-5k/models`;
  - `SKIP_LOCAL_PRETRAIN=1`, `SKIP_LOCAL_META=1`, `SKIP_FED_NORMAL_META=1`;
  - `ENABLE_FED_NORMAL_META_PROPOSED=1`, `FED_NORMAL_META_SELF_FLOOR=0.8`, `FED_NORMAL_META_USE_BEST=0`;
  - `EXTREME_TARGET_ADAPT_MAX_WINDOWS=0`, `EXTREME_ANCHOR_REG_LAMBDA=0.001`, `EXTREME_SOURCE_HARD_GATE=1`, `EXTREME_SOURCE_MIN_TARGET_GAIN=0.0`;
  - `FEW_SHOT_EPOCHS=200`, `EXTREME_TARGET_REFINEMENT_EPOCHS=200`.
- Data quality:
  - result table has no NaN/non-finite values in nMAE/nRMSE/WD metrics;
  - convergence report has `non_finite_loss=0`, `non_finite_state=0`;
  - train log shows hard gate active (`[hard_gate:...]` count 76), anchor logging active, and no `non-finite`.
- All-6 Overall_Average mean metrics:
  - `LMT = 27.8206 nMAE / 31.4453 nRMSE / 24.8788 WD`
  - `Extreme-FedAvg = 27.3246 nMAE / 30.7211 nRMSE / 24.8282 WD`
  - `Proposed-A = 27.0657 nMAE / 30.3739 nRMSE / 24.5831 WD`
  - `Proposed-A vs LMT = +2.71% nMAE / +3.41% nRMSE / +1.19% WD` by ratio-of-means relative formula.
  - `Proposed-A vs Extreme-FedAvg = +0.95% nMAE / +1.13% nRMSE / +0.99% WD`.
- Compared with 2026-04-12 `2h6p self08 5k final200`:
  - previous `Proposed-A vs LMT = +3.08% nMAE / +3.63% nRMSE / +1.19% WD`;
  - no-K-cap anchor+gate improves Proposed-A absolute mean errors (`-0.782 nMAE`, `-0.826 nRMSE`, `-0.809 WD`) but improves LMT too, so the relative gap does not increase.
- Per-weather signal:
  - HighWind worsens versus K1 and is slightly worse than the 2026-04-12 mainline in absolute Proposed-A metrics, indicating full target adaptation overfits or drifts HighWind.
  - Frost recovers sharply versus K1 and slightly improves over the 2026-04-12 mainline.
  - HighTemperature absolute Proposed-A improves strongly, but relative gap is near neutral because LMT also improves.
  - ColdWave is unchanged from K1 because the available target windows make the effective split nearly identical.
- Simulated class-adaptive cap from existing per-class outputs:
  - K1 for HighWind + no-K-cap for remaining classes gives only about `+3.15% nMAE / +3.53% nRMSE / +1.48% WD` vs LMT;
  - K1 for HighWind and HighTemperature + no-K-cap for ColdWave/Frost gives about `+3.26% nMAE / +3.68% nRMSE / +1.61% WD`;
  - even the optimistic per-class combination remains below the `>=5%` target.
- Conclusion:
  - Anchor+gate is numerically stable and improves Proposed-A absolute error, but because the same target adaptation also improves LMT, it does not widen the Proposed-A-vs-LMT gap enough.
  - Single class-adaptive K is likely insufficient.
  - Next recommended mechanism is Proposed-specific source gain weighting: store source target-validation gain over the target self/reference update and include that gain in Proposed-A weighted aggregation, while leaving LMT unchanged and keeping FedAvg as the uniform-source baseline.

### 2026-04-17 - Implemented class-adaptive cap + source gain weighting
- Added two minimal switches for the next extreme-stage experiment:
  - `EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS`, comma-separated per-class target adapt caps. Example: `1,1,0,0` means HighWind K1, HighTemperature K1, ColdWave uncapped, Frost uncapped. Empty value preserves old global `EXTREME_TARGET_ADAPT_MAX_WINDOWS` behavior.
  - `EXTREME_SOURCE_GAIN_WEIGHT_ETA`, default `0.0`. When positive, Proposed-A source scores are multiplied by `target_gain ** ETA`, where `target_gain = max(0, reference_target_val_loss - source_target_val_loss)`.
- The gain weighting is only used by `aggregate_extreme_updates_weighted`, which is the Proposed-A source aggregation path. LMT remains local target adaptation; Extreme-FedAvg remains uniform aggregation.
- Recommended next run:
  - reuse `artifacts/2h6p_six_station/fed-normal-meta-5k/models`;
  - `SKIP_LOCAL_PRETRAIN=1`, `SKIP_LOCAL_META=1`, `SKIP_FED_NORMAL_META=1`;
  - `EXTREME_TARGET_ADAPT_MAX_WINDOWS=0`;
  - `EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS=1,1,0,0`;
  - `EXTREME_ANCHOR_REG_LAMBDA=0.001`;
  - `EXTREME_SOURCE_HARD_GATE=1`;
  - `EXTREME_SOURCE_MIN_TARGET_GAIN=0.0`;
  - `EXTREME_SOURCE_GAIN_WEIGHT_ETA=1.0`.
- Verification:
  - `python -m unittest tests/test_extreme_fl_contract_ast.py` passed with 6 tests.
  - `python -m unittest discover -s tests` passed with 61 tests.
  - `python -m py_compile DemoModelTraining.py run_three_station_yearly_protocol.py` passed.
  - train/eval dry-runs confirmed the intended `six_station_2h_6point_phase_augmented_protocol`, base checkpoint dir, output artifact dir, class cap string, and gain eta.

### 2026-04-17 - 2h6p self08 5k classcap11xx + gaineta1 + anchor+gate result
- Completed run:
  - train log: `logs/2h6p_six_station_self08_5k_classcap11xx_gaineta1_anchor1e3_gate_train_20260417_204025.log`
  - eval log: `logs/2h6p_six_station_self08_5k_classcap11xx_gaineta1_anchor1e3_gate_eval_all6_20260417_211233.log`
  - result CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-5k-classcap11xx-gaineta1-anchor1e3-gate/results/multi_station_performance.csv`
- Config:
  - reused `artifacts/2h6p_six_station/fed-normal-meta-5k/models`;
  - `SKIP_LOCAL_PRETRAIN=1`, `SKIP_LOCAL_META=1`, `SKIP_FED_NORMAL_META=1`;
  - `ENABLE_FED_NORMAL_META_PROPOSED=1`, `FED_NORMAL_META_SELF_FLOOR=0.8`, `FED_NORMAL_META_USE_BEST=0`;
  - `EXTREME_TARGET_ADAPT_MAX_WINDOWS=0`, `EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS=1,1,0,0`;
  - `EXTREME_ANCHOR_REG_LAMBDA=0.001`, `EXTREME_SOURCE_HARD_GATE=1`, `EXTREME_SOURCE_MIN_TARGET_GAIN=0.0`, `EXTREME_SOURCE_GAIN_WEIGHT_ETA=1.0`;
  - `FEW_SHOT_EPOCHS=200`, `EXTREME_TARGET_REFINEMENT_EPOCHS=200`, `EXTREME_WEIGHT_BETA_SELF=0.5`.
- Data quality:
  - result table has no NaN/non-finite values in nMAE/nRMSE/WD metrics;
  - convergence report has `non_finite_loss=0`, `non_finite_state=0` across 2140 records;
  - train log shows hard gate active (`[hard_gate:...]` count 68), anchor logging active, and no `nonfinite`/`nan`.
- All-6 Overall_Average mean metrics:
  - `LMT = 25.6769 nMAE / 29.4791 nRMSE / 22.3528 WD`
  - `Extreme-FedAvg = 24.8976 nMAE / 28.5440 nRMSE / 21.9407 WD`
  - `Proposed-A = 24.8204 nMAE / 28.3675 nRMSE / 21.9855 WD`
  - `Proposed-A vs LMT = +3.34% nMAE / +3.77% nRMSE / +1.64% WD` by ratio-of-means relative formula.
  - `Proposed-A vs Extreme-FedAvg = +0.31% nMAE / +0.62% nRMSE / -0.20% WD`.
- Compared with recent variants:
  - vs 2026-04-12 `2h6p self08 5k final200`, relative gap improves from `+3.08% / +3.63% / +1.19%` to `+3.34% / +3.77% / +1.64%`;
  - vs K1 anchor+gate and no-K-cap anchor+gate, current Proposed-A absolute mean errors are lower on all three metrics;
  - however the current relative gap is still below the `>=5%` target.
- Per-weather Proposed-A vs LMT:
  - HighWind: `+9.80% nMAE / +9.64% nRMSE / +2.55% WD`;
  - HighTemperature: `+0.18% nMAE / +0.94% nRMSE / -0.09% WD`;
  - ColdWave: `+1.87% nMAE / +2.73% nRMSE / +1.48% WD`;
  - Frost: `+1.58% nMAE / +1.47% nRMSE / +2.88% WD`.
- Interpretation:
  - Class-adaptive cap plus gain weighting is directionally better than the two immediate predecessors and is the best recent relative-gap result, but the gain mainly preserves HighWind and recovers Frost; it does not create enough advantage on HighTemperature/ColdWave.
  - Proposed-A only beats Extreme-FedAvg slightly, which means the source weighting is not yet strong enough to produce a distinct Proposed-A advantage.
  - Next highest-leverage cheap run is to keep this exact config but reduce Proposed-A self weight, e.g. `EXTREME_WEIGHT_BETA_SELF=0.3`, so accepted gated/gain-weighted sources can have more influence. This changes only the Proposed-A weighted aggregation path and leaves LMT/Extreme-FedAvg semantics unchanged.

### 2026-04-17 - 2h6p self08 5k classcap11xx + gaineta1 beta03 result
- Completed run:
  - train log: `logs/2h6p_six_station_self08_5k_classcap11xx_gaineta1_beta03_anchor1e3_gate_train_20260417_214109.log`
  - eval log: `logs/2h6p_six_station_self08_5k_classcap11xx_gaineta1_beta03_anchor1e3_gate_eval_all6_20260417_214109.log`
  - result CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-5k-classcap11xx-gaineta1-beta03-anchor1e3-gate/results/multi_station_performance.csv`
- Config:
  - identical to `classcap11xx + gaineta1 + anchor+gate` except `EXTREME_WEIGHT_BETA_SELF=0.3`;
  - reused `artifacts/2h6p_six_station/fed-normal-meta-5k/models`;
  - skipped local pretrain/local meta/Fed-Normal-Meta;
  - `EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS=1,1,0,0`, `EXTREME_ANCHOR_REG_LAMBDA=0.001`, `EXTREME_SOURCE_HARD_GATE=1`, `EXTREME_SOURCE_GAIN_WEIGHT_ETA=1.0`.
- Data quality:
  - result table has no NaN/non-finite values in nMAE/nRMSE/WD metrics;
  - convergence report has `non_finite_loss=0`, `non_finite_state=0` across 2140 records;
  - train log shows hard gate active (`[hard_gate:...]` count 68), anchor logging active, and no `nonfinite`/`nan`.
- All-6 Overall_Average mean metrics:
  - `LMT = 25.6769 nMAE / 29.4791 nRMSE / 22.3528 WD`
  - `Extreme-FedAvg = 24.8976 nMAE / 28.5440 nRMSE / 21.9407 WD`
  - `Proposed-A = 24.8274 nMAE / 28.3855 nRMSE / 21.9892 WD`
  - `Proposed-A vs LMT = +3.31% nMAE / +3.71% nRMSE / +1.63% WD` by ratio-of-means relative formula.
  - `Proposed-A vs Extreme-FedAvg = +0.28% nMAE / +0.56% nRMSE / -0.22% WD`.
- Compared with beta05:
  - beta05 was `+3.34% nMAE / +3.77% nRMSE / +1.64% WD`;
  - beta03 worsens all three overall gaps by `-0.027 pp nMAE / -0.061 pp nRMSE / -0.016 pp WD`;
  - beta03 Proposed-A absolute means are also slightly worse: `+0.0069 nMAE`, `+0.0180 nRMSE`, `+0.0037 WD`.
- Per-weather Proposed-A vs LMT:
  - HighWind: `+9.60% nMAE / +9.23% nRMSE / +2.86% WD`;
  - HighTemperature: `+0.13% nMAE / +0.87% nRMSE / -0.14% WD`;
  - ColdWave: `+2.01% nMAE / +2.95% nRMSE / +1.33% WD`;
  - Frost: `+1.52% nMAE / +1.41% nRMSE / +2.76% WD`.
- Interpretation:
  - Lowering beta from `0.5` to `0.3` increases accepted source influence but does not improve the overall target gap.
  - It slightly helps ColdWave nMAE/nRMSE, but hurts HighWind, HighTemperature, Frost, and overall Proposed-A vs FedAvg separation.
  - Do not continue broad `EXTREME_WEIGHT_BETA_SELF` sweeps as a main route; beta05 remains the current best recent result.
  - A test-oracle upper bound using only beta05-or-LMT fallback reaches about `+4.37% nMAE / +4.29% nRMSE`, and beta05/beta03/FedAvg/LMT fallback reaches about `+4.73% nMAE / +4.60% nRMSE`, so the next plausible mechanism is validation-selected candidate/fallback rather than another fixed beta.

### 2026-04-17 - Implemented Proposed-A target-validation fallback
- Added minimal transfer-or-abstain support for the next experiment:
  - `EXTREME_PROPOSED_VAL_FALLBACK`, default `0`, enables target-validation selection for the final `Proposed-A` checkpoint.
  - `EXTREME_PROPOSED_VAL_FALLBACK_MARGIN`, default `0.0`, requires `Proposed-A` validation loss to be at least this much lower than `LMT` validation loss; otherwise `Proposed-A` falls back to the local `LMT` state for that station/class.
- Behavior:
  - LMT and Extreme-FedAvg are unchanged.
  - Proposed-A still trains the weighted transfer candidate exactly as before.
  - After Proposed-A target refinement, `select_proposed_final_state_by_target_validation(...)` compares `Proposed-A` vs `LMT` on the same target validation split and saves either the transfer state or the LMT fallback state to the Proposed-A model path.
  - This avoids test-set oracle selection; the test set is still untouched until evaluation.
- Launcher/runtime notes:
  - `run_three_station_yearly_protocol.py` now passes through `EXTREME_PROPOSED_VAL_FALLBACK`, `EXTREME_PROPOSED_VAL_FALLBACK_MARGIN`, and `EXTREME_WEIGHT_BETA_SELF`.
  - For the 4090 screen run, prefer direct `env ... /root/miniconda3/bin/python -u DemoModelTraining.py` and then direct eval, because launcher presets may override epoch counts if not carefully set.
- Verification:
  - `python -m unittest tests/test_extreme_fl_contract_ast.py` passed with 7 tests.
  - `python -m unittest discover -s tests` passed with 62 tests.
  - `python -m py_compile DemoModelTraining.py run_three_station_yearly_protocol.py` passed.

### 2026-04-17 - 2h6p valfallback result is worse than beta05 mainline
- Completed run:
  - train log: `logs/2h6p_six_station_self08_5k_classcap11xx_gaineta1_valfallback_anchor1e3_gate_train_20260417_224323.log`
  - eval log: `logs/2h6p_six_station_self08_5k_classcap11xx_gaineta1_valfallback_anchor1e3_gate_eval_all6_20260417_224323.log`
  - result CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-5k-classcap11xx-gaineta1-valfallback-anchor1e3-gate/results/multi_station_performance.csv`
- Config:
  - same as `classcap11xx + gaineta1 + anchor+gate` beta05, plus `EXTREME_PROPOSED_VAL_FALLBACK=1`, `EXTREME_PROPOSED_VAL_FALLBACK_MARGIN=0.0`;
  - reused `artifacts/2h6p_six_station/fed-normal-meta-5k/models`;
  - skipped local pretrain/local meta/Fed-Normal-Meta.
- Data quality:
  - result table has no NaN/non-finite values in nMAE/nRMSE/WD metrics;
  - convergence report has no non-finite loss/state records in sampled records;
  - 21-row all-6 evaluation CSV was generated successfully.
- All-6 Overall_Average mean metrics:
  - `LMT = 25.6769 nMAE / 29.4791 nRMSE / 22.3528 WD`
  - `Extreme-FedAvg = 24.8976 nMAE / 28.5440 nRMSE / 21.9407 WD`
  - `Proposed-A = 25.1095 nMAE / 28.7865 nRMSE / 22.2468 WD`
  - `Proposed-A vs LMT = +2.21% nMAE / +2.35% nRMSE / +0.47% WD`
  - `Proposed-A vs Extreme-FedAvg = -0.85% nMAE / -0.85% nRMSE / -1.40% WD`
- Compared with beta05 mainline:
  - beta05 remained better at `+3.34% nMAE / +3.77% nRMSE / +1.64% WD`;
  - valfallback worsened Proposed-A absolute means by `+0.2890 nMAE`, `+0.4189 nRMSE`, `+0.2613 WD`.
- Selection diagnosis:
  - `proposed_val_select` chose `fallback_lmt` for 12/24 station-class pairs and `proposed` for 12/24.
  - Fallback heavily hurt HighWind and Frost by removing useful transfer:
    - station 62 HighWind beta05 Proposed-A nMAE was `15.0376` vs LMT `19.6897`, but valfallback reverted to LMT;
    - station 58 HighWind beta05 Proposed-A nMAE was `26.8222` vs LMT `28.0483`, but valfallback reverted to LMT;
    - 5/6 Frost classes reverted to LMT, mostly losing small but consistent beta05 gains.
  - Fallback helped some negative-transfer cells such as station 59 ColdWave, but the net effect was negative.
- Conclusion:
  - Do not use raw `EXTREME_PROPOSED_VAL_FALLBACK_MARGIN=0.0` as the mainline.
  - The target validation split is too sparse/noisy under `EXTREME_TARGET_ADAPT_MAX_WINDOWS_BY_CLASS=1,1,0,0`; raw point-loss selection does not reliably predict held-out extreme test performance.
  - If fallback is revisited, it should be made conservative, e.g. relative tolerance, minimum validation-window requirement, or class-specific fallback mask. Otherwise keep beta05 mainline as the current best recent result.

### 2026-04-22 - Long-session handoff and current strategic interpretation
- Current effective target remains relative, not absolute:
  - `relative_gap = (LMT_metric_% - Proposed_metric_%) / LMT_metric_% * 100`;
  - `>=5%` is the minimum target, `5%-10%` is the ideal range;
  - always report `Proposed-A vs LMT` directly, not only `Proposed-A vs Extreme-FedAvg` or win counts.
- Current pipeline must be described accurately:
  - it is **not** merely `Fed-Normal-Meta -> target-only fine-tuning`;
  - current mainline is `Fed-Normal-Meta -> Fed-extreme target fine-tuning / aggregation -> target refinement`;
  - any future analysis must verify this implementation context before proposing mechanism changes.
- Confirmed 2h/6p six-station protocol:
  - six stations are `58/59/60/61/62/63`;
  - `58/59/60` use one 2h phase and `61/62/63` use the complementary 2h phase;
  - for 2h sampling, `POINTS_PER_DAY=12` is correct because a full day has twelve 2h points;
  - `LEN_REALP=6` means a 12-hour input window, not 6 points per full day.
- Current best recent experimental result is still:
  - `classcap11xx + gaineta1 + anchor1e3 + gate`, beta/self weight `EXTREME_WEIGHT_BETA_SELF=0.5`;
  - CSV: `artifacts/2h6p_six_station/fed-normal-meta-self08-5k-classcap11xx-gaineta1-anchor1e3-gate/results/multi_station_performance.csv`;
  - logs: `logs/2h6p_six_station_self08_5k_classcap11xx_gaineta1_anchor1e3_gate_train_20260417_204025.log`, `logs/2h6p_six_station_self08_5k_classcap11xx_gaineta1_anchor1e3_gate_eval_all6_20260417_211233.log`;
  - Overall `Proposed-A vs LMT = +3.34% nMAE / +3.77% nRMSE / +1.64% WD`;
  - this improves the 2026-04-12 self08 5k final200 result (`+3.08% / +3.63% / +1.19%`) but remains below the `>=5%` target.
- Routes already shown to have low leverage:
  - increasing extreme fine-tuning/refinement from 200 to 500 shrank the gap (`+3.08%/+3.63%/+1.19%` to `+2.00%/+2.23%/+0.56%`);
  - increasing equal-budget meta from 5k to 10k also shrank the gap (`+1.83%/+2.03%/+0.85%`);
  - beta03 did not beat beta05;
  - raw validation fallback with `EXTREME_PROPOSED_VAL_FALLBACK=1` and `margin=0.0` was worse (`+2.21%/+2.35%/+0.47%`) because sparse target validation misclassified useful transfer.
- Current strategic interpretation:
  - do not claim the overall `Fed-Normal-Meta -> Fed-extreme fine-tuning` framework is invalid yet;
  - the evidence only shows that current implementation/constraints do not reliably produce the desired 5%-10% gap;
  - LMT is a strong baseline because it can use the same target extreme support/refinement path and may erase the initialization/FL advantage when target adaptation is too strong;
  - further small weight/fallback tweaks are low priority unless they directly test a mechanism that can plausibly widen `Proposed-A vs LMT`.
- Most useful next diagnostic, before changing the framework:
  - reuse the existing `2h6p_six_station/fed-normal-meta-5k` base and run a controlled target extreme support-cap sweep: `K=1`, `K=2`, `K=4`, and current/all;
  - keep all other settings fixed and plot `Proposed-A vs LMT relative_gap` versus K;
  - if smaller K widens the gap, the framework is probably valid but the current target extreme setting is not sufficiently few-shot;
  - if smaller K still cannot widen the gap, then the Fed-extreme mechanism itself is weak;
  - if `Extreme-FedAvg` beats `Proposed-A` under small K, the issue is the Proposed-A source weighting rather than FL itself.
- Git/reproducibility note:
  - commit `e30c3c17f9a36a486ef5b60fd25292cf28611715` (`4.12.20: Fed-Normal-Meta_six_station_pilot5k`) was pushed to GitHub branch `fed-normal-meta-six-station-pilot5k`;
  - in that commit, root `WPF-under-extreme-weather-main/multi_station_performance.csv` is a **three-station** `2h6p_refine_k2` result, corresponding to `logs/2h6p_refine_k2_eval_20260411_211137.log`;
  - the real six-station Fed-Normal-Meta result for `4.12.20` is `artifacts/2h6p_six_station/fed-normal-meta-5k/results/multi_station_performance.csv`, corresponding to `logs/2h6p_six_station_fed_normal_meta_self08_5k_eval_all6_20260412_193528.log`.
- Interaction/process note:
  - for expensive 4090 experiments, prefer giving the user exact `screen`/terminal commands unless the user explicitly asks Codex to start the run;
  - do not over-agree with the user's hypothesis. First anchor the answer in the actual pipeline, logs, and known results, then separate confirmed facts from interpretation.
