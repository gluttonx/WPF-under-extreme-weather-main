# 🌍 核心角色设定
你现在的角色是专注于 **极端天气下风电功率预测 (WPF)** 的顶尖 AI 算法研究员与资深架构师。
你的首要目标是确保算法的严谨性、代码的高效性以及科研产出的高质量。

# 通用需求分析与方案设计原则
- 先从原始需求出发，不默认用户已经完全想清楚目标、约束和实现路径。
- 只有当需求存在关键歧义，且不同理解会导致明显不同方案或较高错误成本时，才先停下来澄清；否则基于最合理解释继续，并明确说明假设。
- 当需要给出修改或重构方案时，默认只围绕用户明确提出的目标设计方案，不擅自扩展业务目标，不引入替代业务路径。
- 优先给出满足目标的最小完整方案，而不是补丁式兼容方案；但如果“最短路径”与“非补丁”冲突，应优先选择不会引入结构性错误的最小正确方案。
- 不做与当前需求无关的兜底、降级或额外分支设计；但为保证逻辑闭合，允许加入必要的输入约束、状态检查和边界保护。
- 输出方案前，按输入、处理流程、状态变化、输出、上下游影响进行链路检查。
- 对无法验证的部分必须明确标注假设和未验证前提，不得将推测表述为已确认事实。

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

# 🖥️ Runtime Environment Contract
- **正式训练与正式实验结论**：默认以云服务器 **RTX 4090** 环境为准；涉及性能、训练时长、吞吐、最终指标时，必须按“最终在 4090 上跑”来设计与表述。
- **本地/当前会话验证**：若当前环境无可用 CUDA，可使用 **CPU** 做短轮次 smoke test / 结构验证 / 语法验证，但必须明确标注其性质仅为“链路验证”，**不能**把 CPU 短程结果当作正式实验结论。
- **运行分工约束**：
  - 小轮次与中轮次验证（例如 `1`、`2`、`50`、`100` epochs 这类 smoke / debug / sanity check）默认由代理直接在当前环境中执行；若当前无 CUDA，则按 **CPU** 口径执行。
  - 长轮次或正式版本训练（例如上万轮、完整预算、最终汇报口径）默认**不**由代理在当前会话里代跑；代理只提供可直接执行的终端命令，由用户在自己的 `RTX 4090` 终端中运行。
- **设备口径**：回答用户或记录实验时，必须显式区分：
  1. `CPU smoke / debug validation`
  2. `4090 formal run`
- **默认行为**：当前会话若检测到无 CUDA，可继续做小步验证；但进入正式训练前，必须提醒用户切回 4090 环境。

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
  - **读论文：** 直接调用 `mineru` MCP 深度解析 PDF，精准提取核心方法论的 LaTeX 公式与图表逻辑。
  - **找文献：** 利用 `exa` MCP 并开启学术域名过滤，精准捕获最新顶刊顶会（包括arxiv.org、nature.com 和 ieee.org 等渠道）论文。
  - **处理数据：** 利用 `xlsx` / `docx` / `pptx` 技能进行格式化读取与规范排版输出。

## 5. 版本控制与代码托管
- **触发场景：** 需要将本地修改推送到远程代码库，或拉取上游更新。
- **工具协同：** 调用 `github` MCP 管理仓库状态。
- **强制红线：** 所有的 Git 关联与身份验证，**必须严格且仅使用 HTTPS 协议**（配合 Personal Access Token）。绝对禁止提供或尝试任何 SSH 密钥相关的解决方案。


---

## Decision Log
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

### 2026-03-11 - 联邦严格版架构口径
- 当前 `USE_FEDERATION=True` 的实现只在 pre-train 阶段做多场站平均；meta-train 仍使用全局任务池，因此不能再表述为“严格数据孤岛联邦”。
- 严格主线决策：
  - 共享模块限定为 `TCN backbone` 或其等价共享表征模块；
  - `LWP + base learner + extreme few-shot adaptation params` 保持本地，不作为默认聚合对象；
  - step-11 few-shot 继续沿用 `experience loss / MSE` 口径。
- 论文映射修正：
  - `F2L` 保留为二阶段候选增强：`dual-model + MI + PKD`；
  - `pFedFSL` 保留为三阶段候选增强：客户端基于本地表现的加权路由与模型选择；
  - “历史 Teacher + PKD” 降级为研究型候选方案，不作为第一阶段主线。

### 2026-03-11 - DemoModelTraining.py 严格联邦 baseline 已落地
- 第一阶段执行范围只覆盖 `DemoModelTraining.py`：
  - 新增 `ENABLE_FED_META_TRAIN=False`，默认关闭全局 task-pool meta-training；
  - 新增共享/本地参数边界函数：`is_shared_param / is_local_param / extract_* / load_mixed_state_dict`；
  - 常规天气 pre-train 改为 strict FL：服务器仅聚合共享 `TCN backbone`，客户端保留本地 `LWP + fore_baselearner`。
- 新增输出约定：
  - 共享 backbone 快照：`model_fore_shared_backbone_federated.pth`
  - 每站个性化预训练模型：`model_fore_pre_station{station_id}_personalized.pth`
  - 每站结果文件：`station{station_id}_test_results.mat`
- 本阶段 few-shot 口径：
  - `Proposed` 默认从每站个性化预训练模型启动；
  - `Meta-only` 仅在未来重新打开 `ENABLE_FED_META_TRAIN` 后才恢复主路径。
- 后续阶段边界固定：
  - `F2L-main` 可直接借鉴联邦轮次组织、dual-model、MI/KD 接口；
  - `pFedFSL` 无现成项目代码，后续如引入路由/个性化聚合，必须自写实现。

### 2026-03-11 - generate_multi_station_results.py 已对齐严格联邦 baseline
- 评估主线仍然按极端天气类别重跑模型，不直接复用 `station{station_id}_test_results.mat` 的全年预测。
- `Pre_Training` 评估优先读取每站个性化预训练模型：
  - `model_fore_pre_station{station_id}_personalized.pth`
  - 若缺失才回退到旧的全局 pretrain 文件。
- `Meta_Learning` 评估口径修正：
  - 当 `DemoModelTraining.py` 中 `ENABLE_FED_META_TRAIN=False` 时，默认不再读取旧的 `meta_only` / `train_task_query` 模型；
  - 此时 `Meta_Learning` 行只保留兼容占位，回退到该场站个性化 pretrain 快照。
- 训练时长与排序校验同步修正：
  - strict baseline 下 `Proposed` 时长按 `pre-train + few-shot` 估计；
  - `Meta_Learning` 时长记为 `NaN`；
  - 排序校验退化为只强制 `Proposed <= Pre_Training`，不再使用旧的 `Meta < Pre` 假设。

### 2026-03-13 - DemoModelTraining.py 已接入 F2L-style Phase 2 开关
- 第二阶段实现口径：
  - 新增 `ENABLE_F2L_PHASE2=False`，默认关闭；
  - 保持 strict FL 边界：服务器只聚合共享 `TCN backbone`，不采用 `F2L-main` released code 的 whole-state 聚合再掩码广播；
  - `F2L` 论文中的 `server-model/client-model` 在当前 WPF 回归架构中映射为“共享 backbone 视图 / 本地 LWP+head 视图”。
- 已落地的代码能力：
  - `model_fore` 新增 `forward_features(...)` 与 `forward_with_features(...)`；
  - 新增本地 episodic task 采样 `sample_local_meta_task(...)`；
  - 新增 `compute_f2l_mi_proxy_loss(...)` 与 `compute_f2l_kd_loss(...)`；
  - 新增 `client_local_f2l_round(...)`，按 support fine-tune -> server support update -> client query update 的顺序执行本地更新。
- 损失口径：
  - server 侧使用回归版 `((1-λ_MI) * (MSE + CDRM) + λ_MI * MI_proxy)`；
  - client 侧使用回归版 `((1-λ_KD) * MSE + λ_KD * KD_proxy)`；
  - 当前 `MI/KD` 为可运行代理形式，不是论文 Eq.(13)/(16)-(19) 的逐式复现。
- 输出兼容性：
  - 继续产出兼容的 `model_fore_pre_station{station_id}_personalized.pth` 与 `model_fore_pre_federated.pth`；
  - 额外保存 Phase 2 专用快照：`model_fore_shared_backbone_f2l_federated.pth` 与 `model_fore_pre_station{station_id}_f2l_personalized.pth`。

### 2026-03-13 - F2L-style Phase 2 smoke test 与设备回退修复
- 为适配当前无可用 CUDA 的运行环境，`DemoModelTraining.py` 的设备选择改为：
  - `device=torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  - `seed_torch(...)` 中的 CUDA seed 调用仅在 `torch.cuda.is_available()` 时执行。
- smoke test 配置已实跑通过：
  - `ENABLE_F2L_PHASE2=True`
  - `ENABLE_FED_META_TRAIN=False`
  - `PRETRAIN_EPOCHS=200`
  - `FEW_SHOT_EPOCHS=5`
  - `F2L_LOCAL_TASKS_PER_ROUND=1`
- smoke test 观测结论：
  - `Phase 2` 训练日志已正常输出 `support_mse / query_mse / mi / kd`；
  - F2L 专用 checkpoint `model_fore_shared_backbone_f2l_federated.pth` 已成功生成；
  - 兼容 checkpoint 与每站预测结果文件也已完整生成；
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py` 已成功产出 CSV。
- smoke test 指标结论：
  - `Overall_Average` 下，`Proposed` 相对 `Pre_Training` 在绝大多数指标上改善；
  - 但仍有 4 个轻微反向项（集中在 `58-HighTemperature` 与 `59-Frost`），因此当前只能说明通路有效，不能直接宣称正式配置下稳定优于 strict baseline。

### 2026-03-13 - 设备选择回归已重新修复
- `DemoModelTraining.py` 曾回退到 `device=torch.device("cuda")` 的硬编码状态，现已恢复为：
  - `device=torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  - `seed_torch(...)` 中 CUDA seed 仅在 GPU 可用时调用。
- 运行时可见性增强：
  - 启动时直接打印 `运行设备` 与 `torch.cuda.is_available()`，避免再次误判是在 CPU 还是 GPU 上训练。
- 当前环境检查结论：
  - 虽然代码层已修复，但这份 Python 环境里 `torch.cuda.is_available()` 仍为 `False`，因此训练仍会回退到 CPU；若用户侧已修通 4090，则需在同一 Python 环境中再次验证。

### 2026-03-13 - Phase 2 第一波性能优化已落地
- 目标：减少 `Phase 2` 中 GPU/CPU 往返、小张量重复构造和每 task 的模型/优化器重建开销。
- 已完成的结构优化：
  - `clone_state_dict(...)`、`extract_shared_state_dict(...)`、`extract_local_state_dict(...)` 默认保持当前 device，不再默认 `.cpu().clone()`；
  - 新增 `local_meta_task_cache` 与 `prepare_local_meta_task_cache(...)`，将常规天气 local episodic task 数据预缓存为 tensor；
  - 新增 `create_phase2_station_context(...)` 与 `reset_optimizer_state(...)`，在 `client_local_f2l_round(...)` 内复用 station 级 `client_model / server_model` 和对应 optimizer；
  - 新增 `cpu_state_snapshot(...)`，仅在 checkpoint 保存时才把 state 转到 CPU。
- 运行时可见性：
  - 启动日志继续打印 `运行设备`，可直接判断当前是在 CPU 还是 GPU 上。
- 验证情况：
  - 静态测试与语法检查均已通过；
  - 短程启动验证已进入 `client_local_f2l_round(...)` 的反向传播路径后人工中断，未见新结构性异常。

### 2026-03-13 - Phase 2 第二波性能优化已落地
- 目标：在保留 strict FL / F2L-style Phase 2 语义的前提下，进一步提高 4090 上的有效吞吐，而不是继续让 Python 小任务调度主导总耗时。
- 已完成的性能向改动：
  - 新增 `PERF_PRIORITIZE_SPEED=True`，当 `torch.cuda.is_available()` 为真时，`seed_torch(...)` 改为 `cudnn.deterministic=False`、`cudnn.benchmark=True`；CPU 或无 CUDA 场景仍保留确定性路径；
  - 新增 `F2L_BATCH_LOCAL_TASKS=True` 与 `sample_local_meta_task_batch(...)`，将同一 station round 内多个 local task 的 `support/query` 直接拼成更大的 batch；
  - `client_local_f2l_round(...)` 改为优先按 batched task bundle 执行 `client-support -> server-support -> client-query`，减少极小张量下的 Python 循环与 repeated kernel launch 开销；
  - 训练横幅新增性能模式打印，显式展示 `PERF_PRIORITIZE_SPEED` 与 `F2L_BATCH_LOCAL_TASKS` 的实际状态。
- 验证情况：
  - `python -m unittest tests.test_phase2_perf_wave2_ast tests.test_phase2_perf_ast tests.test_device_selection_ast tests.test_f2l_phase2_ast tests.test_strict_federated_baseline_ast tests.test_generate_multi_station_results_ast tests.test_few_shot_loss_ast` 通过；
  - `python -m py_compile DemoModelTraining.py` 通过；
  - 短程启动验证已打印 `性能优先模式` 横幅并进入 `client_local_f2l_round(...)` 的反向传播路径后人工中断，未见新的结构性异常。
- 解释边界：
  - 这一步主要减少的是碎任务调度开销，不会改变 strict FL 的 shared-only aggregation 口径；
  - 若用户侧 4090 环境仍然没有明显提速，下一优先级应是更细粒度 runtime profiling 与升级 PyTorch/CUDA 软件栈，而不是继续改硬件。

### 2026-03-14 - Diag-1 结果：batching 不是 Phase 2 退化主因
- 对照文件：
  - `multi_station_performance_baseline_2000.csv`
  - `multi_station_performance_phase2_2000.csv`
  - `multi_station_performance_diag1_phase2_nobatch_2000.csv`
- 固定预算：`PRETRAIN_EPOCHS=2000`、`FEW_SHOT_EPOCHS=20`、`F2L_LOCAL_TASKS_PER_ROUND=3`。
- Diag-1 设置：保持 `ENABLE_F2L_PHASE2=True`，仅将 `F2L_BATCH_LOCAL_TASKS=False`，其余 `MI/KD` 配置不变。
- 结论：
  - `no-batch Phase 2` 仍明显劣于 strict baseline，因此 batching 不是当前 Phase 2 退化的主因；
  - 相比 batched Phase 2，`no-batch` 在 `HighTemperature/Frost` 上有所回升，但在 `HighWind/ColdWave` 上进一步恶化，说明 batching 只是次级影响因子而非根因；
  - `Pre_Training` 行同样未被救回，说明主要问题发生在 Phase 2 的预训练/shared update 阶段，而不是仅仅出现在 extreme-weather few-shot 适配。
- 下一诊断优先级：
  - 保持 `F2L_BATCH_LOCAL_TASKS=False`；
  - 下调 `F2L_LAMBDA_MI`（例如 `0.01`），继续做 Diag-2 以验证 server-side MI proxy 是否是退化主因。

### 2026-03-14 - Diag-2 结果：降低 MI 权重也未救回 Phase 2
- 对照文件：
  - `multi_station_performance_baseline_2000.csv`
  - `multi_station_performance_diag1_phase2_nobatch_2000.csv`
  - `multi_station_performance_diag2_phase2_mi001_2000.csv`
- 固定设置：`ENABLE_F2L_PHASE2=True`、`F2L_BATCH_LOCAL_TASKS=False`、`PRETRAIN_EPOCHS=2000`、`FEW_SHOT_EPOCHS=20`、`F2L_LOCAL_TASKS_PER_ROUND=3`；Diag-2 仅将 `F2L_LAMBDA_MI` 从 `0.1` 下调到 `0.01`。
- 结论：
  - `Diag-2` 相比 `Diag-1` 未显著回升，且在 `HighWind/ColdWave` 上普遍进一步恶化；
  - `Pre_Training` 行同样没有恢复，说明“MI 过强”不是当前 Phase 2 退化的主因；
  - 结合 Diag-1，可排除“batching 是主因”和“仅靠降低 MI 即可修复”两种简单解释。
- 当前更可信的判断：
  - 问题出在 F2L-style Phase 2 的整体训练设定与当前 WPF 回归映射之间存在结构性不匹配，尤其是将 F2L 的 client/server knowledge transfer 直接映射到 `shared backbone + local LWP/head` 后，并未稳定改善 strict FL baseline。
- 主线决策：
  - strict baseline 继续作为当前主线；
  - Phase 2 暂不升格为默认方案，后续若继续研究，应作为诊断/附录分支而非主结果来源。

### 2026-03-14 - 旧版 3.8.17.32 与 strict baseline 不可直接横比
- `git` 历史显示：用户提到的旧版对应提交为 `c385634 (3.8.17.32)`。
- 该版本与当前 strict baseline 的核心差异不止一个：
  - 旧版默认仍执行 `Proposed` 的 `meta-training`（`run_meta_training(...)`），且使用 3 站联合 `global task pool`；
  - 当前 strict baseline 默认关闭 `ENABLE_FED_META_TRAIN`，只保留 strict FL pre-train + local few-shot；
  - 旧版 few-shot 默认 `FEW_SHOT_USE_CDRM=True`，当前论文对齐版改为 `FEW_SHOT_USE_CDRM=False`（MSE only）；
  - 旧版 `Pre_Training` 评估默认读取全局 `model_fore_pre_federated.pth`，当前 strict baseline 优先读取每站 `model_fore_pre_station{station_id}_personalized.pth`；
  - 用户近期用于 strict baseline / Phase 2 诊断的预算是 `PRETRAIN_EPOCHS=2000`、`FEW_SHOT_EPOCHS=20`，而旧版默认预算是 `PRETRAIN_EPOCHS=35000`、`PROPOSED_META_EPOCHS=30000`、`FEW_SHOT_EPOCHS=50`。
- 因此：旧版 `3.8.17.32` 指标显著更好并不意外，因为它同时享受了更强的中心化/伪联邦 meta-training、更重的 few-shot 正则以及大得多的训练预算；它不能被当成“strict baseline 只改了少量代码”的公平对照。

### 2026-03-14 - 主线收敛：放弃 F2L，转向 strict baseline + pFedFSL-lite
- 针对 `F2L-style Phase 2` 的同预算 A/B、Diag-1（关 batching）和 Diag-2（`MI=0.01`）均已完成。
- 结论：
  - `F2L-style Phase 2` 在当前 WPF strict federated setting 下持续劣于 strict baseline；
  - `batching` 不是主退化原因；
  - 仅降低 `MI` 权重也无法救回性能；
  - 更可信的解释是：将 F2L 的 client/server knowledge transfer 直接映射到当前 `shared backbone + local LWP/head` 的回归任务上存在结构性不匹配。
- 主线决策更新：
  - 不再将 F2L 作为当前主线增强；
  - 当前主线固定为 `strict baseline`；
  - 下一阶段转向 `pFedFSL-lite` 风格的 personalized FL。
- `3` 场站条件下的 personalized FL 口径：
  - 第一版不引入完整 `Q` 矩阵，先做 `Q-less pFedFSL-lite`；
  - 服务器每轮向目标客户端发送 `{self, peer1, peer2}` 三个 shared backbones；
  - 客户端基于本地 route split 对三个候选 shared states 打分并 softmax 聚合；
  - 只路由/聚合 shared backbone，`LWP + base learner + few-shot head` 保持本地；
  - `Q-lite` 仅作为后续通信裁剪增强，而不是第一版主干。
- 关于“致命问题 1”的最终口径：
  - `CDRM` 可以保留为客户端本地损失；
  - 客户端与服务器仍可通信，通信对象是基于本地损失训练后的 shared backbone 更新，而不是原始样本或细粒度梯度结构；
  - 因此 strict FL 可保留 `local CDRM regularization`，但不能宣称完整保留了原始 WPF 的精确 CDRM 联邦形式。

### 2026-03-14 - Q-less pFedFSL-lite 第一版骨架已落地
- `DemoModelTraining.py` 主线默认值已切换为：
  - `ENABLE_PFEDFSL_LITE=True`
  - `ENABLE_F2L_PHASE2=False`
  - `ENABLE_FED_META_TRAIN=False`
- 预训练主路径已不再维护单一 `shared_global_state`：
  - 服务器改为维护每站独立 shared backbone：`station_shared_states[station_id]`
  - 每轮客户端基于全部候选 shared states 做本地路由，再只上传更新后的本地 shared backbone
  - 单全局 `FedAvg` 赋值主路径 `shared_global_state = server_aggregate_shared_updates(...)` 已从主循环移除
- Q-less 路由口径：
  - 新增 route/update split；`D_i^route` 与 `D_i^update` 分离
  - 路由分数基于本地损失做 `softmax` 聚合 shared backbone
  - `CDRM` 继续只保留在客户端本地 route/update 损失中
- 输出兼容性：
  - `generate_multi_station_results.py` 依赖的 `model_fore_pre_station{station_id}_personalized.pth` 保持不变
  - 额外保存每站 shared backbone：`model_fore_shared_backbone_station{station_id}_federated.pth`
- 测试与工具：
  - 新增 `pfedfsl_lite_utils.py`
  - 新增 `tests/test_pfedfsl_lite_ast.py` 与 `tests/test_pfedfsl_routing_utils.py`
  - 现有 AST 回归与 `py_compile` 已通过

### 2026-03-14 - pFedFSL-lite smoke run 与 route split 维度约束
- 短程 smoke run 已在临时目录完成：
  - 使用临时预算 `PRETRAIN_EPOCHS=3`、`FEW_SHOT_EPOCHS=2`
  - `DemoModelTraining.py` 主流程已完整走通到 per-station shared backbone 保存、few-shot 保存和 `all_stations_test_results.mat`
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py` 在临时目录下可正常读取这些产物并生成 `multi_station_performance.csv`
- 新确认的实现约束：
  - 当前联邦常规天气预训练张量不是多 batch 样本，而是单序列形态：`1 x T x C`
  - 因此 `Q-less pFedFSL-lite` 的 `route/update split` 不能默认沿 batch 维切分
  - 当 batch 维为 `1` 时，必须沿时间维 `T` 切分 `D_i^route` 与 `D_i^update`
- 本轮已修复：
  - `pfedfsl_lite_utils.split_route_update_batch(...)` 现在支持两种口径：
    - `B>=2` 时沿 batch 维切分
    - `B=1` 且为 `1 x T x C` 时沿时间维切分
  - 对应测试已补齐：`tests/test_pfedfsl_routing_utils.py`
- smoke 观测：
  - route weights 日志已正常打印
  - 在 `3` 个极短预训练 epoch 下，平均权重仍接近 `58=0.333, 59=0.333, 60=0.333`
  - 这更像是 smoke 预算下候选 shared states 仍近似同质的正常现象，不能据此评价个性化路由是否已形成稳定偏好

### 2026-03-14 - pFedFSL-lite 中等预算验证（CPU-only 20/5）
- 中等预算验证已在隔离目录完成：
  - `PRETRAIN_EPOCHS=20`
  - `FEW_SHOT_EPOCHS=5`
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py` 成功运行
- 训练侧现象：
  - pre-train 4 个日志点（epoch `5/10/15/20`）均打印出 route weights
  - 平均 route weights 仍保持 `58=0.333, 59=0.333, 60=0.333`
  - `loss_mse` 与 `route_loss` 随 epoch 单调下降，说明主路径训练是正常收敛的
- shared backbone 差异检查：
  - 站间 shared backbone 已不再完全相同，但差异量仍很小
  - 当前 pairwise `max_abs diff` 约为 `4e-4`，`L2 diff` 约为 `0.105`
  - 更合理的解释是：在 CPU-only 的 `20/5` 中等预算下，3 个候选 shared states 仍过于接近，因此 softmax 路由尚未形成可见偏好
- 当前判断：
  - `Q-less pFedFSL-lite` 的代码通路、日志通路、评估通路都已验证通过
  - 但“路由偏好已经形成”这一点目前还不能宣称，需要更长预算、不同温度，或更敏感的路由打分记录来继续判断

### 2026-03-14 - pFedFSL-lite 温度敏感性与更高预算验证（tau=0.25, 20/5 与 50/5）
- 已完成两组隔离实验：
  - `PFED_ROUTE_TEMPERATURE=0.25`, `PRETRAIN_EPOCHS=20`, `FEW_SHOT_EPOCHS=5`
  - `PFED_ROUTE_TEMPERATURE=0.25`, `PRETRAIN_EPOCHS=50`, `FEW_SHOT_EPOCHS=5`
- 两组实验都已完成：
  - `DemoModelTraining.py` 训练全链路
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py` 评估链路
- route weights 精确值结论：
  - 通过 TensorBoard scalar 读取精确值后，平均 route weights 仍仅在 `0.3332~0.3334` 范围内波动
  - 将 `tau` 从 `0.5` 下调到 `0.25` 后，没有观察到可解释的显著放大效应
  - 将预算从 `20` 提高到 `50` 后，也没有让平均 route weights 明显偏离均匀态
- shared backbone 差异结论：
  - `tau=0.25, 20/5` 与 `tau=0.25, 50/5` 下，站间 shared backbone 仍只有很小差异
  - pairwise `max_abs diff` 依旧约为 `4e-4`
  - 说明当前更主要的瓶颈不是 softmax 温度过高，而是候选 shared states 在现有 WPF 预训练口径下仍然过于接近
- 当前更可信的下一步判断：
  - 不应继续优先花时间做 `tau` 微调
  - 若要逼出路由偏好，下一优先级应转向：
    - 更长预算（例如 `100+` epoch）
    - 更有区分度的 route split / route loss 记录
    - 或改为记录每客户端 `route_weights`，而不只看跨客户端平均值
- 评估侧附带观察：
  - `tau=0.25, 50/5` 下，评估脚本给出轻微 warning：
    - `Station 58 Frost_nMAE_%: Proposed > Pre_Training`
    - `Station 58 Frost_nRMSE_%: Proposed > Pre_Training`
  - 因此该配置下仍不能宣称 `Proposed` 已在全部 strict baseline 指标上稳定占优

### 2026-03-14 - per-client route weights 日志已落地 + tau=0.25, 100/5 验证
- 日志增强：
  - `DemoModelTraining.py` 现已同时记录：
    - 跨客户端平均 `route_weight_pre_pfedfsl_lite_{candidate}`
    - 每客户端细粒度 `route_weight_pre_pfedfsl_lite_client{client}_from_{candidate}`
  - 控制台预训练日志也已显示 `route_weights_by_client`
- `tau=0.25, PRETRAIN_EPOCHS=100, FEW_SHOT_EPOCHS=5` 隔离实验已完成，训练与评估链路都成功跑通。
- 每客户端 route weights 结论：
  - 到 `100` epoch 时，权重仍只在非常窄的范围波动，约为 `0.3330~0.3337`
  - 示例末轮：
    - `client58`: `58=0.333309, 59=0.333538, 60=0.333153`
    - `client59`: `58=0.333464, 59=0.333168, 60=0.333368`
    - `client60`: `58=0.333334, 59=0.333664, 60=0.333002`
  - 这说明即使观察每客户端而不是看平均值，也仍未出现真正有解释力的路由分化
- shared backbone 差异结论：
  - `100` epoch 后站间 shared backbone 差异仍维持在很小量级
  - pairwise `max_abs diff` 仍约为 `4e-4`
  - 因此当前更可信的解释仍是“候选 shared states 过于接近”，而不是“平均化掩盖了真实偏好”
- 评估侧结论：
  - `STRICT_PAPER_ORDER=0` 下评估脚本继续跑通
  - 但 `100` epoch 配置下仍出现多条 `Proposed > Pre_Training` warning（如 `58-HighTemperature`, `59-HighTemperature`, `60-Frost`），因此不能把更长预算简单解读为更优个性化联邦结果
- 当前主线判断：
  - `Q-less pFedFSL-lite` 代码通路、日志通路、评估通路已全部验证
  - 但截至 `tau=0.25, 100/5`，仍没有证据表明 routing preference 已经形成
  - 下一优先级不应再是继续加长同构训练预算，而应转向提高 route signal 可分性，例如：
    - 改 route split 的构造方式
    - 记录/分析 route loss matrix 而不只看 softmax weights
    - 或显式增强 per-station shared state 的异质性来源

### 2026-03-14 - route loss matrix 日志已落地 + 早期诊断结论
- `DemoModelTraining.py` 已新增 `route loss matrix` 记录：
  - 控制台预训练日志新增 `route_loss_matrix_mean` 与 `route_losses_by_client`
  - TensorBoard 新增：
    - 平均 `route_loss_pre_pfedfsl_lite_{candidate}`
    - 每客户端 `route_loss_pre_pfedfsl_lite_client{client}_from_{candidate}`
- `tau=0.25` 隔离诊断已在 `100/5` 配置上跑到至少 `30` 个 pre-train epoch，并读取到 `epoch 9/19/29` 的 route loss matrix。
- 关键现象：
  - 同一客户端对 3 个候选 shared backbones 的 `route_loss` 差异极小，通常只在 `1e-4` 量级；
  - 但不同客户端之间的绝对 `route_loss` 差异明显更大，例如：
    - `epoch 29` 时 `client58` 约 `0.1411`
    - `client59` 约 `0.2449`
    - `client60` 约 `0.2840`
  - 这说明当前问题不是 softmax 把显著偏好抹平，而是候选 shared states 在每个客户端看来本来就几乎等价。
- 当前判断更新：
  - `route split` 的构造暂时不是第一优先级；
  - 若后续要继续提高个性化路由可分性，更优先的方向应是：
    - 增强 per-station shared backbone 的异质性来源
    - 或继续分析/改造 route loss 定义本身（例如更敏感的相对差分记录）
  - 只有在确认 shared states 已拉开后仍无区分度时，才值得优先改 route split 采样策略

### 2026-03-14 - FedRep-lite 已切为新的 accuracy-first true-FL 主线
- 主线目标更新：
  - 在“数据不出本地、服务器只做 shared-backbone 协同训练”的前提下，优先追求各场站预测精度，而不是继续逼 `Q-less pFedFSL-lite` 学出弱路由偏好。
- `DemoModelTraining.py` 默认值已切换为：
  - `ENABLE_FEDREP_LITE=True`
  - `ENABLE_PFEDFSL_LITE=False`
  - `ENABLE_F2L_PHASE2=False`
- 新实现口径：
  - 新增 `client_local_fedrep_round(...)`
  - 每个客户端常规天气 pre-train 采用两阶段交替更新：
    - 先冻结 shared backbone，只更新本地 `LWP + fore_baselearner`
    - 再冻结本地个性化部分，只更新 shared backbone
  - 服务器仅聚合单一 shared backbone；`Q-less pFedFSL-lite` 继续保留为可选 ablation 分支
- 日志增强：
  - 新增 `head_stage_loss_pre_fedrep_lite`
  - 新增 `backbone_stage_loss_pre_fedrep_lite`
- 输出兼容性：
  - `model_fore_pre_station{station_id}_personalized.pth` 保持不变
  - `model_fore_shared_backbone_station{station_id}_federated.pth` 在 FedRep-lite 下仍会生成，但内容是同一全局 shared backbone 的兼容快照

### 2026-03-14 - FedRep-lite 超短程 smoke + 评估链路通过
- 已在隔离目录完成 ultrashort smoke：
  - `PRETRAIN_EPOCHS=1`
  - `FEW_SHOT_EPOCHS=1`
  - 目录：`/tmp/wpf_fedrep_ultrashort`
- 观测结果：
  - 首个 FedRep-lite 预训练 epoch 已打印：
    - `loss_mse`
    - `head_stage_loss`
    - `backbone_stage_loss`
  - 共享 backbone、每站个性化 pretrain、12 个 few-shot 模型、`station{station_id}_test_results.mat` 与 `all_stations_test_results.mat` 都已成功生成
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py` 已在同一隔离目录成功生成 `multi_station_performance.csv`
- 当前解释边界：
  - 这次 smoke 只说明 `FedRep-lite` 代码通路、产物通路和评估通路都已打通
  - 还不能说明它在真实预算下已经优于 strict baseline；下一步必须做中等预算 A/B

### 2026-03-14 - FedRep-lite 运行时 bug 修复 + 20/5 A/B 结果
- `FedRep-lite` 本地两阶段更新在中等预算初跑时暴露出一个真实运行时问题：
  - 为了降低内存峰值，曾尝试在 `client_local_fedrep_round(...)` 末尾用 `no_grad` 重新计算 `final penalty`
  - 但 `penalty(...)` 内部依赖 `autograd.grad(...)`，因此在 `no_grad` 下会抛出 `RuntimeError`
- 已修复口径：
  - 不再在末尾额外重建 `final penalty` 计算图
  - 直接复用最后一个 backbone-stage 的 `loss_penalty / loss_mse` 作为该轮返回统计
  - 同时显式释放 head/backbone stage 的临时张量并执行 `gc.collect()`，避免阶段图在同一客户端轮次内不必要叠加
- 中等预算 A/B 已完成：
  - `FedRep-lite`: `/tmp/wpf_fedrep_20_5`
  - `strict baseline`: `/tmp/wpf_strict_20_5`
  - 两组都使用：
    - `PRETRAIN_EPOCHS=20`
    - `FEW_SHOT_EPOCHS=5`
    - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py`
- `Overall_Average / Proposed` 结论：
  - `FedRep-lite` 在 12 个误差指标中仅有 2 个略优（`Frost_nMAE_%`, `Frost_WD_%`），其余 10 个都略差于 strict baseline
  - 代表性差值（`FedRep-lite - strict`，误差指标越小越好）：
    - `HighWind_nMAE_%`: `+0.0520`
    - `HighTemperature_nRMSE_%`: `+0.0700`
    - `ColdWave_nRMSE_%`: `+0.0685`
    - `Frost_nMAE_%`: `-0.0307`
  - `R_p<0.05_%` 也低于 strict baseline，约 `-2.40` 个百分点
- `Overall_Average / Pre_Training` 结论：
  - `FedRep-lite` 同样没有优于 strict baseline；12 个误差指标里也只在 `Frost_nMAE_%` 与 `Frost_WD_%` 上略好，其余多数略差
- 当前主线判断更新：
  - 现有 `FedRep-lite` 虽然代码通路已稳定，但在 `20/5` 中等预算下没有显示出优于 strict baseline 的证据
  - 因此它目前仍更适合作为“accuracy-first 个性化主线候选的第一轮否证结果”，而不是已确认替代 strict baseline 的默认最优方案

### 2026-03-14 - 运行环境与执行分工约定
- 当前 Codex 所在环境按“验证环境”理解：
  - 本地仅用于 CPU-only 的小规模验证、smoke run、bug 复现与几十步/几十轮级别的结构性检查
  - 不把当前 CPU-only 环境当作正式训练环境来决定算法主线优劣
- 正式训练口径：
  - 用户会在自己的终端中使用 `RTX 4090` 执行真正的中长程训练
  - 因此后续当需要正式实验时，Codex 的职责应是：
    - 先在当前环境做小规模可运行性验证
    - 再给出用户可直接在 4090 环境执行的完整命令
- 决策边界：
  - 当前环境里的短程结果可用于排除结构错误、验证日志/产物/评估通路
  - 但不应把 CPU-only 小预算结果直接当成正式性能结论的最终依据

### 2026-03-14 - SFML-lite 主线骨架已落地（strict FL + strict federated meta-train + local few-shot）
- 主线决策更新：
  - 不回退整个 `DemoModelTraining.py` 到 `3.8.17.32`
  - 当前文件继续作为 strict FL 骨架
  - 旧版 `3.8.17.32` 仅作为 meta-learning 语义 donor，不再整文件回退
- `DemoModelTraining.py` 当前默认值已切换为：
  - `ENABLE_STRICT_FED_META_TRAIN=True`
  - `ENABLE_FED_META_TRAIN=False`（legacy pseudo-FL global-task-pool meta only）
  - `ENABLE_FEDREP_LITE=False`
  - `ENABLE_PFEDFSL_LITE=False`
  - `ENABLE_F2L_PHASE2=False`
- 已落地结构：
  - 新增 env override 层：`env_flag / env_int / env_float`
  - 新增 strict meta 配置：
    - `STRICT_META_USE_SECOND_ORDER`
    - `STRICT_META_EPOCHS`
    - `STRICT_META_LOCAL_TASKS_PER_ROUND`
    - `STRICT_META_INNER_STEPS`
    - `STRICT_META_SUPPORT_SIZE`
    - `STRICT_META_QUERY_SIZE`
    - `STRICT_META_INNER_LR`
    - `STRICT_META_USE_CDRM`
  - 新增 strict federated meta helper：
    - `compute_meta_support_loss(...)`
    - `compute_meta_query_loss(...)`
    - `meta_inner_adapt(...)`
    - `client_local_meta_round(...)`
    - `run_strict_federated_meta_training(...)`
- strictness 边界：
  - episodic support/query task 继续只通过 `sample_local_meta_task(...)` 从本地场站常规天气数据采样
  - 服务器仍然只通过 `server_aggregate_shared_updates(...)` 聚合 shared backbone
  - 本地 `LWP + fore_baselearner + few-shot params` 不上传
- legacy 降级：
  - 旧 `sample_meta_batch(...)` 已改名为 `sample_legacy_global_meta_batch(...)`
  - 旧 `run_meta_training(...)` 已改名为 `run_legacy_global_meta_training(...)`
  - 旧 global task pool 伪联邦 meta 路径仍保留作对照，但不再承载主线语义
- checkpoint 语义：
  - pre-train-only 继续保存 `model_fore_pre_station{station_id}_personalized.pth`
  - strict federated meta-train 新增保存 `model_fore_meta_station{station_id}_personalized.pth`
  - 新增 `model_fore_shared_backbone_meta_federated.pth`
  - `Proposed` few-shot 在 strict meta 开启时，改为从每站 `model_fore_meta_station{station_id}_personalized.pth` 启动
- 二阶 MAML 边界：
  - `STRICT_META_USE_SECOND_ORDER=True` 已预留为未来 full MAML 开关
  - 当前实现仍是一阶 `SFML-lite`；若误开二阶，会显式抛出 `NotImplementedError`，避免静默跑成错误语义

### 2026-03-14 - SFML-lite 1/1/1 CPU smoke + 评估链路通过
- 已在隔离目录 `/tmp/wpf_sfml_smoke` 完成：
  - `PRETRAIN_EPOCHS=1`
  - `STRICT_META_EPOCHS=1`
  - `FEW_SHOT_EPOCHS=1`
  - `ENABLE_STRICT_FED_META_TRAIN=1`
  - 其余增强分支全部关闭
- 观测结果：
  - strict pre-train 已完成并输出每站个性化 pretrain checkpoint
  - strict federated meta-train 已打印首轮日志：
    - `support_loss`
    - `query_loss`
    - `meta_penalty`
    - `shared_update_norm`
  - strict meta shared backbone 与每站 meta 初始化 checkpoint 均已成功生成：
    - `model_fore_shared_backbone_meta_federated.pth`
    - `model_fore_meta_station58_personalized.pth`
    - `model_fore_meta_station59_personalized.pth`
    - `model_fore_meta_station60_personalized.pth`
  - 12 个 Proposed few-shot 模型、每站结果文件和 `all_stations_test_results.mat` 全部成功生成
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py` 已在同一隔离目录成功生成 `multi_station_performance.csv`
- 当前解释边界：
  - 这次 smoke 说明 `strict pre-train -> strict federated meta-train -> local few-shot -> evaluation` 的通路已经跑通
  - 但它仍只是 CPU-only 小预算结构验证，不能当作正式性能结论
- 后续正式实验约定：
  - 真正的 `20/5`、更长预算，以及未来二阶 MAML 对比，应由用户在自己的 `RTX 4090` 终端上执行
  - Codex 在当前环境只继续负责小规模验证、命令生成与结果分析

### 2026-03-15 - strict federated meta 评估提示语已对齐
- `generate_multi_station_results.py` 已新增 strict meta 检测：
  - 新增 `detect_strict_fed_meta_enabled_from_training_script()`
  - 当 `DemoModelTraining.py` 默认启用 `ENABLE_STRICT_FED_META_TRAIN` 时，评估脚本会明确打印：
    - `Proposed` 由 strict federated meta 初始化
    - `Meta_Learning` 行仍保留兼容占位
- 边界保持不变：
  - 本轮没有重新定义 `Meta_Learning` 行的表格语义
  - `Pre_Training` 仍读取每站 pretrain-only checkpoint
  - `Proposed` 仍通过最终 few-shot 模型体现 strict federated meta 带来的初始化收益

### 2026-03-15 - RTX 4090 正式结果：SFML-lite 一阶在 20/5 下优于 strict baseline
- 正式对比目录：
  - strict baseline：`/tmp/wpf_4090_strict_20_5`
  - SFML-lite 一阶：`/tmp/wpf_4090_sfml_20_5`
- 共同预算：
  - `PRETRAIN_EPOCHS=20`
  - `STRICT_META_EPOCHS=20`（仅 SFML-lite）
  - `FEW_SHOT_EPOCHS=5`
  - `STRICT_PAPER_ORDER=0`
- `Overall_Average / Proposed` 结论：
  - SFML-lite 相对 strict baseline 在 12 个误差指标上全部更优，且 `R_p<0.05_%` 更高
  - 代表性差值（`SFML-lite - strict baseline`，误差指标越小越好）：
    - `HighWind_nMAE_%: -0.6653`
    - `HighTemperature_nRMSE_%: -0.4515`
    - `ColdWave_nRMSE_%: -0.6299`
    - `Frost_WD_%: -0.3151`
    - `R_p<0.05_%: +3.1266`
- 站点层面结论：
  - `Proposed` 在 `58/59/60` 三个场站上都一致优于 strict baseline
  - 不是由单一场站拉动的偶然提升
- 机制解释边界：
  - `Meta_Learning` 与 `Pre_Training` 行几乎不变，符合当前评估口径：`Meta_Learning` 仍是兼容占位
  - 真正变好的部分是 `Proposed`，说明 strict federated meta 初始化确实改善了后续 local few-shot 适配
- 训练日志可见性：
  - strict baseline 由于 `PRETRAIN_EPOCHS=20 < 100`，当前预训练日志间隔设置下没有额外 TensorBoard 标量
  - SFML-lite 的 strict meta 标量有记录：
    - `loss_support_strict_meta` 从 `0.265650` 下降到 `0.248850`
    - `loss_query_strict_meta` 从 `0.240157` 升到 `0.254244`
    - `shared_update_norm_strict_meta` 维持在约 `1e-4`
  - 说明这一轮的收益已在最终指标上显现，但 strict meta 的 query-side 训练动态仍值得在更长预算下继续观察
- 主线判断更新：
  - 现有证据已支持把 `SFML-lite` 一阶从“候选主线”提升为“当前 true-FL 精度主线”
  - 下一步优先级不再是继续试 FedRep-lite / pFedFSL-lite，而是：
    - 先做更长预算确认收益是否稳定
    - 若稳定，再实现完整二阶 MAML

### 2026-03-15 - RTX 4090 更长预算结果：SFML-lite 一阶在 100/5 与 200/5 下未保持优势
- 对比目录：
  - strict baseline：`/tmp/wpf_4090_strict_100_5`
  - SFML-lite 一阶：`/tmp/wpf_4090_sfml_100_5`
  - SFML-lite 一阶：`/tmp/wpf_4090_sfml_200_5`
- 关键语义提醒：
  - `generate_multi_station_results.py` 里的 `Meta_Learning` 行在 strict federated meta 主线下仍是兼容占位，数值上等同于 `Pre_Training`
  - 因此这里真正有意义的比较是 `Proposed` 对 `Pre_Training`
- `100/5` 结论：
  - strict baseline 下，`Proposed` 仍略优于 `Pre_Training`
  - 但 SFML-lite 下，`Proposed` 相对 `Pre_Training` 在 `Overall_Average` 的 12 个误差指标中有 11 个更差，仅 `Frost_nMAE_%` 略好（约 `-0.0240`）
  - 说明 `20/5` 的增益没有稳定外推到 `100/5`
- `200/5` 结论：
  - SFML-lite 的 `Proposed` 相对 `Pre_Training` 出现严重结构性退化：
    - `HighWind` 三项误差约 `+20`
    - `ColdWave` 三项误差约 `+19`
  - 但同时在 `HighTemperature` 与 `Frost` 上反而更好，说明这不是纯随机噪声，而是“对部分极端类型的偏置性迁移”
- 训练日志诊断：
  - `100/5` 的 strict meta 日志显示：
    - `support_loss` 从 `0.258060` 降到 `0.160423`
    - `query_loss` 从 `0.228020` 降到 `0.173239`
    - 即常规天气本地 episodic 代理任务上看起来是“训练正常”的
  - `200/5` 的 strict meta 日志显示：
    - `support_loss/query_loss` 在 `epoch 100` 左右达到低点后，到 `epoch 200` 明显反弹到约 `0.306557/0.309453`
    - 这是更像 over-training / instability，而不是简单的继续收益
- 额外诊断（直接评估 checkpoint，而非表格中的兼容 `Meta_Learning` 行）：
  - `100/5`：
    - `pretrain checkpoint` 略优于 `strict_meta_init checkpoint`
    - `few-shot Proposed` 能从 `strict_meta_init` 回升一部分，但仍未追回 `pretrain checkpoint`
  - `200/5`：
    - `strict_meta_init checkpoint` 已明显差于 `pretrain checkpoint`
    - `few-shot Proposed` 同样只能部分回升，无法救回
- 当前更可信的机制解释：
  - 问题主因不只在 few-shot；strict meta 阶段本身已经开始把初始化推离对极端天气更有利的区域
  - `client_local_meta_round(...)` 当前同时更新并保留 shared backbone 与 local state，长期运行后，本地 `LWP + head` 很可能被常规天气 episodic task 过度塑形，削弱了后续 extreme-weather few-shot 迁移
  - 同时当前 strict meta 只保存“最后一轮 checkpoint”，没有 best-checkpoint 选择，这在 `200/5` 的反弹轨迹下尤其危险
- 主线判断再次更新：
  - 现阶段不能再直接把“一阶 SFML-lite”视为已稳定优于 strict baseline 的默认主线
  - 更合理的结论是：
    - `20/5` 显示出潜力
    - 但 `100/5` / `200/5` 证明当前实现存在明显的长预算稳定性问题
  - 下一优先级应先修训练策略，而不是直接升级到二阶 MAML：
    - 保存 best strict-meta checkpoint，而不是只取最后一轮
    - 重新考虑是否应把 meta 更新后的 local state 直接带入 few-shot 起点
    - 必要时收紧 `STRICT_META_EPOCHS / INNER_LR / LOCAL_TASKS_PER_ROUND`

### 2026-03-15 - Table IV 语义校正：论文中的 `Meta_Learning` 是真实消融，不是占位字段

- 已核对 [Wind_Power_Forecasting_Under_Extreme_Weather_a_Novel_Few-Shot_Learning_Architecture.pdf] 的 TABLE IV。
- 论文语义应明确区分：
  - `Pre_Training`
  - `Meta_Learning`
  - `Transfer learning`
  - `Proposed = Pre-train + Meta-learning + Transfer learning (fine-tune)`
- 当前工程里，训练脚本的 legacy `Meta-only` 路径实际上是按论文消融口径继续执行同口径 few-shot 的：
  - 见 `DemoModelTraining.py` 中 `Meta-only：同口径执行 step-11 few-shot，确保与论文消融对齐`
- 但当前 `generate_multi_station_results.py` 在 strict federated meta 主线下，仍把 `Meta_Learning` 行回退成“兼容占位 / 个性化预训练快照”：
  - 这与论文 TABLE IV 的真实语义不一致
- 因此必须明确：
  - 论文里的 `Meta_Learning` 不是占位
  - 当前 strict-SFML 结果表里的 `Meta_Learning` 行只是临时兼容字段，不能按论文 Table IV 的真实 `Meta_Learning` 方法来解读
- 后续若要恢复论文一致的表格语义，应新增 strict federated `meta-only` 消融路径，而不是继续让 `Meta_Learning` 行回退到 `Pre_Training`

### 2026-03-15 - 结果表语义修正：strict baseline 去掉 `Meta_Learning`，strict-SFML 未实现 meta-only 时导出 `N/A`

- `generate_multi_station_results.py` 已按“运行产物”而不是 `DemoModelTraining.py` 默认开关来判断当前目录属于哪种评估口径：
  - strict baseline: 仅输出 `Proposed` 与 `Pre_Training`
  - strict federated meta 且尚未实现 strict `meta-only`: 保留 `Meta_Learning` 行，但整行导出为 `N/A/NaN`
  - legacy meta-only 真存在时：`Meta_Learning` 才作为真实方法评估
- 这样修正后：
  - `/tmp/wpf_4090_strict_100_5/multi_station_performance.csv` 只包含 `Proposed` 和 `Pre_Training`，共 8 行
  - `/tmp/wpf_4090_sfml_100_5/multi_station_performance.csv` 包含 `Proposed` / `Meta_Learning` / `Pre_Training`，共 12 行，其中 `Meta_Learning` 全行为 `NaN`
- 这避免了之前两种语义错误：
  - strict baseline 结果表里误出现伪 `Meta_Learning`
  - strict-SFML 目录里因为 `pivot_table` 丢弃全 NaN 行，日志说是 `N/A`，CSV 却没有 `Meta_Learning` 行

### 2026-03-15 - strict-SFML 稳定性修复第一步：best shared checkpoint + pretrain local rollback

- 已新增纯工具模块 `sfml_meta_utils.py`：
  - `update_best_meta_checkpoint(...)`
  - `compose_personalized_meta_init_state(...)`
- `run_strict_federated_meta_training(...)` 现在不再只使用最后一轮 strict-meta shared state：
  - 按平均 `query_loss` 追踪并保存 best shared checkpoint
  - `STRICT_META_SHARED_BACKBONE_MODEL_PATH` 现在保存的是 best shared meta backbone，而不是最后一轮
- 每站 `model_fore_meta_station{station_id}_personalized.pth` 的组装方式已改为：
  - `best meta shared backbone`
  - `+ initial_local_states[station_id]`（即 pre-train local state rollback）
- 这意味着 Proposed few-shot 的 strict-meta 起点不再直接继承长期 meta 训练后被常规天气 episodic task 塑形的 local head/LWP
- CPU `1/1/1` smoke 已通过，并额外验证：
  - `model_fore_meta_station58_personalized.pth` 的 local 参数与 `model_fore_pre_station58_personalized.pth` 完全一致
  - 其 shared 参数与 `model_fore_shared_backbone_meta_federated.pth` 完全一致

### 2026-03-15 - 修正版 strict-SFML 4090 结果：已恢复 `Proposed > Pre_Training`，但对 strict baseline 的额外优势仍是混合的

- 4090 新目录：
  - `/tmp/wpf_4090_sfml_bestlocal_20_5`
  - `/tmp/wpf_4090_sfml_bestlocal_100_5`
  - `/tmp/wpf_4090_sfml_bestlocal_100_5_tight`
- 关键恢复点：
  - 相比各自 `Pre_Training`，修正版 strict-SFML 的 `Overall_Average / Proposed` 在三组实验里都重新实现了 12/12 个误差指标全优
  - 说明“best shared checkpoint + pretrain local rollback” 已经修复了旧版 `100/5`、`200/5` 出现的系统性退化问题
- 但若与 strict baseline 的 `Proposed` 直接横比，结论仍是混合的：
  - `20/5`：修正版 SFML 对 strict baseline `Proposed` 是 6 优 6 劣，基本接近打平
  - `100/5`：修正版 SFML 是 7 优 5 劣，略有改善但不是单边压制
  - `100/5 tight`：是 5 优 7 劣，说明把 strict-meta 收太紧也不稳
- 因此主线判断更新为：
  - 修正版 strict-SFML 已经从“不稳定/会伤害 Pre_Training”恢复到“可用”
  - 但它还没有稳定证明自己在正式 A/B 中显著优于 strict baseline `Proposed`
  - 下一步应继续做真实 strict `meta-only` 消融与 early-selection 策略，而不是立刻跳到二阶 MAML

### 2026-03-15 - strict-SFML 的 early-selection 已显式化为两个开关

- `DemoModelTraining.py` 已新增：
  - `STRICT_META_SAVE_BEST_ONLY`
  - `STRICT_META_EARLY_STOP_PATIENCE`
- 配套纯工具已补入 `sfml_meta_utils.py`：
  - `select_meta_shared_state(...)`
  - `should_stop_early(...)`
- 当前 strict meta 行为：
  - 始终跟踪 best shared checkpoint（按平均 `query_loss`）
  - `STRICT_META_SAVE_BEST_ONLY=True` 时，用 best shared 作为 strict-meta 输出
  - `STRICT_META_SAVE_BEST_ONLY=False` 时，改为使用 latest shared
  - `STRICT_META_EARLY_STOP_PATIENCE>0` 时，若连续若干轮未刷新 best，则提前结束 strict meta
- CPU smoke 已实测触发：
  - 在 `STRICT_META_EPOCHS=20, STRICT_META_EARLY_STOP_PATIENCE=1` 下，strict meta 于第 2 轮提前停止
  - 日志会输出 `no_improve_rounds`、`strict meta 提前停止`、以及最终 `completed_epochs / best_epoch / best_query_loss / save_best_only`

### 2026-03-15 - 真实 strict `meta-only` 消融已落地，`Meta_Learning` 行不再依赖占位回退

- 已新增 strict federated `meta-only` 开关：
  - `ENABLE_STRICT_FED_META_ONLY`
  - 激活条件为 `TRAIN_META_ONLY_BASELINE and ENABLE_STRICT_FED_META_TRAIN and ENABLE_STRICT_FED_META_ONLY`
- strict `meta-only` 复用了现有 strict meta 主循环，但改为：
  - 从 `meta_only_random_init_state` 拆出的 shared/local 初始状态出发
  - 输出到 `model_fore_meta_only_station{station_id}_personalized.pth`
  - 共享 backbone 输出到 `model_fore_shared_backbone_meta_only_federated.pth`
- few-shot 阶段现在会在 strict `meta-only` 激活时，额外生成每站每类：
  - `model_fore_station{station_id}_extreme{i_class}_meta_only.pth`
- `generate_multi_station_results.py` 已切到按真实产物识别 strict `meta-only`：
  - 发现 `model_fore_meta_only_station{station_id}_personalized.pth` 或 per-class `*_meta_only.pth` 时，`Meta_Learning` 行视为真实结果
  - 否则 strict-SFML 目录里的 `Meta_Learning` 继续导出为 `N/A`
- CPU `1/1/1` smoke 已验证闭环：
  - strict `meta` 与 strict `meta-only` 的 per-station personalized checkpoints 均成功保存
  - 12 个 per-class `meta_only` few-shot 模型成功生成
  - `multi_station_performance.csv` 成功输出 12 行，`Meta_Learning` 行读取的是真实 `model_fore_station*_meta_only.pth`
- 当前评估语义已与论文 TABLE IV 更接近：
  - `Pre_Training` = strict pre-train
  - `Meta_Learning` = strict federated meta-only + few-shot
  - `Proposed` = strict pre-train + strict federated meta + few-shot

### 2026-03-16 - apples-to-apples 对照后的主线判断：legacy 很强，但不是 strict 联邦语义

- 本轮关键对照目标不是复刻旧代码结果，而是做真正的 apples-to-apples：
  - `legacy semantics`：当前代码下尽量复现旧论文/旧主线语义
  - `strict semantics`：当前 strict federated 主线在相同大预算下的表现
- 已确认的关键事实：
  - legacy `35000/30000/50` 结果非常强，但其本质是 `global task-pool meta-style training`，不是 strict federated meta；
  - 旧版 `c385634 (3.8.17.32)` 不是标准 full MAML，也不是标准 FOMAML，更接近 support/query 两阶段顺序优化；
  - 旧版 few-shot 不能再表述为论文原生 `CDRM + MSE`，经论文核对，fine-tuning 应按 `MSE-only` 理解。
- 当前对 legacy vs strict 的总判断：
  - legacy 更像 raw-performance-first 的宽松/伪联邦路线；
  - strict 更像 data-silo-first 的真联邦路线；
  - 两者不能再被表述成“同一种算法的新旧实现只差一个 bug”。
- 因此，后续任何对外表述都必须避免：
  - 把 legacy 的高指标直接当作 strict 主线理应达到的同语义上界；
  - 把 strict 落后于 legacy 简化成“当前 strict 代码一定写坏了”。

### 2026-03-16 - strict 主线目标重置：真联邦边界、完整训练主线、以及最低验收标准

- 当前项目目标已重新固定为：
  - 做一条 **真联邦** 的极端天气风电小样本预测主线；
  - 保留论文语义：`Proposed = Pre-Training + Meta-Training + Fine-Tune`；
  - 但其中 `Meta-Training` 必须是 strict federated 版本，而不是 legacy global task pool。
- strict 联邦边界固定为：
  - server 不看原始数据；
  - server 不接收 local state；
  - server 只聚合 shared 参数；
  - client 保留 local 参数并在本地执行 few-shot 适配。
- 主线训练语义固定为：
  - 固定轮数；
  - final checkpoint；
  - `best-only / early-stop` 只可作为诊断工具，不能再作为最终算法定义的一部分。
- 当前最低效果验收标准固定为：
  - `Proposed` 至少要稳定优于 `Pre_Training`；
  - 在此之前，不再讨论 “strict 主线已经成立，只是还没逼近 legacy”。

### 2026-03-16 - strict 路线的重要原则更新：暂不删除 pre-training；server 不收 local state

- 当前证据不足以支持“删除 pre-training”：
  - 在连续值时序预测里，pre-training 仍是当前最稳定、最强的基线；
  - strict meta 目前尚未强到足以证明可以替代 pre-training；
  - 分类类 FFSL/F2L 常见的不显式 pre-training 不能直接迁移到本项目的连续时序预测场景。
- local state 口径也已固定：
  - client 可以保留自己的 local meta/few-shot 状态；
  - 但 server 不接收、不汇总、不广播 local state；
  - 任何依赖 server-visible local state 的方案都不应被表述为当前 strict 主线。

### 2026-03-16 - strict decoupled meta 已演化为 shared-only aggregation + final shared+local 的主线候选

- strict meta 的主线设计已明确从“同时动 shared 和 local”收敛到更清晰的职责拆分：
  - `support`：优先承担本地适配；
  - `query`：承担 shared 侧更新信号；
  - server：仅聚合 shared。
- 当前主线不再采用 “best shared + pretrain local rollback” 作为最终定义，而是回到：
  - fixed-epoch；
  - final shared；
  - final local；
  - 最终 few-shot 从 `final shared + final local` 启动。
- 但必须明确：
  - 这只说明 strict 主线的边界与流程已更合理；
  - 不代表 shared objective 已经对齐最终 extreme-weather few-shot 的真正需求。

### 2026-03-17 - strict 参数侧与 task 侧的诊断结论：问题已不再是“参数太多”或“task 太容易”这么简单

- 已完成的参数侧尝试包括：
  - `support local-only / query shared-only`
  - 只更新 shared block 子集
  - 引入 `shared_adapter`，并让 server 仅聚合 `shared_adapter`
- 已完成的 task 侧尝试包括：
  - `same-cluster`
  - `cross-cluster`
- 当前诊断结论：
  - task 太容易确实是问题之一，因为 `cross-cluster` 能将 `best_epoch` 从约 `17` 推迟到约 `38`；
  - 但 task 改难后，最终 `Proposed` 仍未稳定优于 `Pre_Training`，说明 task 不是一级矛盾；
  - 将 shared 更新对象从整块主干收缩到 shared 子集也未解决问题，因此问题也不再是“参数范围太大”本身；
  - 更可信的判断是：当前 strict meta 的 shared objective 学到的不是“对最终 few-shot 真有用的共享知识”，而只是某种常规天气 residual proxy。

### 2026-03-17 - shared adapter 方向成立，但不能再被误解成 LoRA/LLM 复刻

- 已检索并参考的相关文献/代码脉络包括：
  - `FedAdapter`
  - `FedPETuning`
  - `Dual-Personalizing Adapter for Federated Foundation Models (FedDPA)`
- 这些工作对本项目的真正启发不是直接搬运 LoRA/LLM 代码，而是：
  - global/shared 小模块负责知识共享；
  - local 模块负责个性化；
  - server 只聚合 global/shared 模块；
  - local 模块始终留在客户端。
- 因此本项目的 `shared_adapter` 应被理解为：
  - 一个 TCN 场景下的 shared lightweight meta module；
  - 它借鉴的是 FedDPA 类方法的 global/local adapter 分工思想；
  - 不是对其 PEFT/LoRA 代码的直接复刻。

### 2026-03-17 - downstream 适配作用域已修正：few-shot 现在更新 `local + shared_adapter`

- 先前一个关键错位是：
  - strict meta 阶段在学习 `shared_adapter`；
  - 但最终 few-shot 阶段只更新 local，不更新 `shared_adapter`。
- 这一点现已修正：
  - downstream few-shot 的训练作用域改为 `finetune = local + shared_adapter`；
  - 从而使 strict meta 学到的 `shared_adapter` 真正进入后续 few-shot 适配算子。
- 这一步是必要修复，但实验已表明：
  - 它只能带来小幅改善；
  - 不能单独解决 `Proposed` 无法稳定优于 `Pre_Training` 的主问题。

### 2026-03-17 - 最新中预算实验结论：不要上 full-budget；当前一级问题是 shared meta objective 错位

- 关键中预算目录与结论：
  - `/tmp/wpf_4090_strict_shared_adapter_100_100_5`：整体不达标；
  - `/tmp/wpf_4090_strict_shared_adapter_crosscluster_100_100_5`：训练动态更像在学，但最终精度仍未过线；
  - `/tmp/wpf_4090_strict_shared_adapter_crosscluster_align_100_100_5`：加入 downstream `local + shared_adapter` 对齐后，相比前一版略有改善，但 `Overall_Average` 仍为 `2` 优 `10` 劣。
- 最新 `crosscluster_align` 结果应记为：
  - `HighWind_nMAE`: `60.9762 -> 61.6749`
  - `HighTemperature_nMAE`: `20.3451 -> 20.4659`
  - `ColdWave_nMAE`: `62.3147 -> 62.8628`
  - `Frost_nMAE`: `20.1014 -> 20.0221`
  - `R_p<0.05_%` 基本不变
- 因此当前明确禁止的下一步：
  - 不要直接上 full-budget；
  - 不要继续在 `same / cross / hybrid task mode` 上做局部微调并期待根治；
  - 不要再把问题解释成“只要拉长训练就会收敛好”。
- 当前最核心的一级判断固定为：
  - strict meta 的 `query` 目标还没有真正对齐 “support 之后、最终 few-shot 是否更有利” 这一需求；
  - 下一步应优先重写 strict meta 的 shared objective，而不是继续调 sampling 或参数范围。

### 2026-03-19 - 主线切换为 federated pre-training + local meta-training
- `DemoModelTraining.py` 当前主线不再继续 strict / decoupled federated meta-learning；第二阶段改为各场站独立的 local meta-training。
- Phase 1 保留联邦预训练，但实现口径改为标准 client-server `FedAvg`：
  - server 广播全局 pretrain 模型；
  - 每个场站仅用本地常规天气数据更新；
  - 客户端上传更新后的模型参数与样本数；
  - server 按样本数加权聚合得到新的全局 pretrain 模型。
- Phase 2 仅使用各场站自己的常规天气 `task clustering` 数据做 `support -> query` 本地元训练；不再聚合 meta update。
- Phase 3 保持论文协议：极端天气数据只在 local fine-tune 出现；few-shot 初始化改为加载“本场站自己的 meta-model”。
- 同轮修复：few-shot 纯 `MSE` 日志路径不再引用未定义的 `loss1`。
- smoke run 结果（运行时临时降轮数，未改源码超参）：`PRETRAIN_EPOCHS=2`、`PROPOSED_META_EPOCHS=2`、`FEW_SHOT_EPOCHS=2`、`TRAIN_META_ONLY_BASELINE=False`；在 CPU 回退环境下已完整跑通 `FedAvg pretrain -> 3站 local meta -> 12个 extreme few-shot -> 全站预测导出`。

### 2026-03-19 - 结果生成脚本已对齐本地主线产物布局
- `generate_multi_station_results.py` 已从旧的 strict federated meta / shared-adapter 假设切回当前 `3.11.15.3 + local meta` 主线：
  - 评估模型结构与当前 `DemoModelTraining.py` 一致，不再带 `shared_adapter`；
  - `Pre_Training` 行默认读取 `model_fore_pre_federated.pth`（必要时回退 `model_fore_pre.pth`）；
  - `Meta_Learning` 行只按真实产物识别 `model_fore_station{station_id}_extreme{class_idx}_meta_only*.pth`，不再依赖旧的 strict federated meta 开关推断。
- 新增 AST 校验 `tests/test_generate_results_local_meta_ast.py`，覆盖：
  - 评估脚本模型结构必须匹配当前训练模型；
  - 评估脚本必须识别 `model_fore_pre_federated.pth` 与 `model_fore_train_task_query_meta_only_station{station_id}.pth` 这套新产物布局。
- 已在当前 CPU 环境下实跑：
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py`
  - 成功生成 `multi_station_performance.csv`；
  - 当前目录下 `Meta_Learning` 真实 few-shot 产物可被正确识别。
- 本轮实跑仍出现论文排序告警，但这是当前模型结果本身的问题，不是评估脚本路径或结构不匹配导致的问题。

### 2026-03-19 - 训练主线已支持环境变量驱动的 smoke / formal run 切换
- `DemoModelTraining.py` 顶部关键运行开关与轮数已外置为环境变量，默认值保持当前论文口径不变：
  - `USE_FEDERATION`
  - `TRAIN_META_ONLY_BASELINE`
  - `FEW_SHOT_EPOCHS`
  - `FEW_SHOT_USE_CDRM`
  - `META_TASKS_PER_EPOCH`
  - `PRETRAIN_EPOCHS`
  - `PROPOSED_META_EPOCHS`
  - `META_ONLY_META_EPOCHS`
  - `META_ONLY_USE_CDRM`
  - `META_ONLY_TRAIN_ALL_PARAMS`
  - `META_ONLY_DISABLE_LWP`
- 目的：
  - 同一份代码即可区分 `CPU smoke / debug validation` 与 `4090 formal run`；
  - 不再需要为短轮次验证临时改源码或做运行时字符串替换。
- 隔离目录 `/tmp/wpf_env_smoke` 已实测：
  - `TRAIN_META_ONLY_BASELINE=0 PRETRAIN_EPOCHS=1 PROPOSED_META_EPOCHS=1 FEW_SHOT_EPOCHS=1 python DemoModelTraining.py`
  - 成功跑通 `FedAvg pretrain -> 3站 local meta -> 12个 proposed few-shot -> all_stations_test_results.mat`
  - 随后 `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py` 成功生成 8 行结果表，并自动省略 `Meta_Learning` 行。
- 当前 CPU smoke 的意义仅是验证：
  - 环境变量入口生效；
  - 训练链路和结果生成链路在隔离目录下可运行；
  - 正式训练和正式指标结论仍必须回到 `RTX 4090` 环境。
- 补充验证：
  - 隔离目录 `/tmp/wpf_env_smoke_meta` 已实测 `TRAIN_META_ONLY_BASELINE=1 PRETRAIN_EPOCHS=1 PROPOSED_META_EPOCHS=1 META_ONLY_META_EPOCHS=1 FEW_SHOT_EPOCHS=1 python DemoModelTraining.py`
  - 成功跑通 `FedAvg pretrain -> 3站 proposed local meta -> 3站 meta-only local meta -> 24个 few-shot 模型 -> all_stations_test_results.mat`
  - 随后 `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py` 成功生成 12 行结果表，并正确保留 `Meta_Learning` 行。

### 2026-03-19 - 5组主表消融已落地并完成 CPU smoke
- 结合根基论文 `Table IV`，联邦多场站主消融收敛为 5 组主表：
  - `Proposed = FedPretrain + LocalMeta + FewShot`
  - `Local_Meta_Transfer = LocalPretrain + LocalMeta + FewShot`
  - `Transfer_Learning = LocalPretrain + FewShot`
  - `Meta_Learning = RandomInit + LocalMeta + FewShot`
  - `Local_PreTraining = LocalPretrain only`
- `DemoModelTraining.py` 已补齐：
  - 每场站 `local conventional pretrain`
  - `Local_Meta_Transfer` 本地元训练分支
  - `Transfer_Learning` 的本地 pretrain 后直接 few-shot 分支
  - `Local_PreTraining` 的按场站预测输出
- `generate_multi_station_results.py` 已改为主表固定输出上述 5 组；`Fed_PreTraining` 若后续需要，仅作为辅助结果而非主表。
- 针对本轮结构新增 AST 校验 `tests/test_local_ablation_matrix_ast.py`，并已通过：
  - `python -m unittest WPF-under-extreme-weather-main.tests.test_local_ablation_matrix_ast -v`
  - `python -m py_compile WPF-under-extreme-weather-main/DemoModelTraining.py WPF-under-extreme-weather-main/generate_multi_station_results.py`
- CPU smoke 已在隔离目录 `/tmp/wpf_local_ablation_smoke` 实测：
  - `USE_FEDERATION=1 TRAIN_META_ONLY_BASELINE=1 PRETRAIN_EPOCHS=1 PROPOSED_META_EPOCHS=1 META_ONLY_META_EPOCHS=1 FEW_SHOT_EPOCHS=1 python DemoModelTraining.py`
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py`
  - 结构链路已跑通，并生成 5 组主表 `multi_station_performance.csv`。
- 当前 1 epoch smoke 下，`Proposed` 未稳定优于 `Local_Meta_Transfer`，因此这次 smoke 只证明代码路径正确，不证明排序结论；下一步必须进入 `RTX 4090` 上的 reduced-budget pilot，而不是直接上 full-budget。

### 2026-03-19 - 4090 pilot 结果诊断：问题更像在 phase interaction，而不是代码串站
- 已检查 `DemoModelTraining.py` 的数据边界：所有含 `meta-training` 的对照组在第二阶段都只调用 `sample_station_meta_batch(station_id)`，仅使用各自场站的 `p_conven_class / nwp_conven_class`；不存在在 meta 阶段聚合所有场站 task 或参数的实现问题。
- 当前 `multi_station_performance.csv` 的 5 组主表显示：
  - `Proposed` 相对 `Local_Meta_Transfer` 在 `Overall_Average` 的 `nMAE/nRMSE` 上仅 `1胜7负`；
  - `Local_Meta_Transfer` 相对 `Meta_Learning` 明显更好，但相对 `Transfer_Learning` 仅 `4胜4负`；
  - `Transfer_Learning` 相对 `Local_PreTraining` 在 `Overall_Average` 的全部 `nMAE/nRMSE` 指标上都更好，说明 phase-3 few-shot fine-tune 本身是有效的。
- 额外辅助诊断（脚本外单独计算）表明：`Fed_PreTraining` 相对 `Local_PreTraining` 在总体 `nMAE/nRMSE` 上是更优的，但优势主要集中在 `HighWind / ColdWave`，而 `HighTemperature / Frost` 反而更差。
- 因此当前最合理的一级判断不是“代码把多场站知识错误混入 meta-training”，而是：
  - `fed pretrain` 和 `local meta` 各自学到的偏好不同；
  - `fed pretrain` 对 `HighWind / ColdWave` 有利，`local meta` 对 `HighTemperature / Frost` 更有利；
  - 两阶段直接串联后，`Proposed` 没能同时保留这两类优势，说明问题更像是 `phase-1 / phase-2 interaction`，而不是简单的 epoch 不够或实现串站。
- 下一步 pilot 不应直接上 full-budget；更合理的是继续做 `4090` 中预算验证，并优先观察：
  - 提高 `PRETRAIN_EPOCHS` 是否能增强 `Proposed`；
  - 增加或继续拉长 `PROPOSED_META_EPOCHS` 是否会进一步抹掉 fed pretrain 在 `HighWind / ColdWave` 上的优势。

### 2026-03-19 - Pilot B 结果：主排序已大幅改善，但 `Proposed > Local_Meta_Transfer` 仍未完全成立
- 当前 `multi_station_performance.csv` 显示，`Overall_Average` 的主指标 `nMAE/nRMSE` 上：
  - `Local_Meta_Transfer` 已经对 `Transfer_Learning / Meta_Learning / Local_PreTraining` 全部实现 `8胜0负`；
  - `Proposed` 也已经对 `Transfer_Learning` 实现 `8胜0负`；
  - 但 `Proposed` 相对 `Local_Meta_Transfer` 仍是 `4胜4负`，尚未形成稳定压制。
- 这说明与上一轮更小预算 pilot 相比，增加训练预算确实明显改善了排序；“训练轮数太少”是合理因素之一。
- 但当前还不能下结论说“继续拉长就一定会反超”，因为：
  - `Proposed - Local_Meta_Transfer` 的差距虽然整体已缩小，但 `station59` 仍是主要瓶颈；
  - 在 `station59` 上，`Proposed` 对 `Local_Meta_Transfer` 仅 `2胜6负`，主要劣势集中在 `HighWind / ColdWave / Frost`。
- 当前最稳妥的判断是：
  - 预算增加已经把问题从“排序明显错误”推进到了“只剩 final edge 未打通”；
  - 下一步若继续试验，应优先围绕 `Proposed` 与 `Local_Meta_Transfer` 的最后差距做验证，而不是重复检查其他较弱基线。

### 2026-03-19 - Pilot A vs Pilot B：较短 meta-training 更有利于 `Proposed > Local_Meta_Transfer`
- `Pilot A`（`PRETRAIN_EPOCHS=2000, PROPOSED_META_EPOCHS=500, META_ONLY_META_EPOCHS=500, FEW_SHOT_EPOCHS=20`）相对 `Pilot B`（同 pretrain/few-shot、但 meta 为 1000）显示：
  - `Proposed` 相对 `Local_Meta_Transfer` 在 `Overall_Average` 的 `nMAE/nRMSE` 上由 `4胜4负` 提升为 `5胜3负`；
  - `Local_Meta_Transfer` 相对 `Transfer_Learning / Meta_Learning / Local_PreTraining` 仍然保持明显优势；
  - `Proposed` 相对 `Transfer_Learning` 仍明显更强。
- `Pilot A` 的 `Overall_Average` 上，`Proposed - Local_Meta_Transfer` 的主差值为：
  - 改善：`HighWind_nRMSE -1.8892`, `HighTemperature_nMAE -0.3809`, `HighTemperature_nRMSE -0.4953`, `ColdWave_nMAE -1.2143`, `ColdWave_nRMSE -2.8065`
  - 劣势：`HighWind_nMAE +0.2236`, `Frost_nMAE +0.8050`, `Frost_nRMSE +0.5033`
- 这说明：
  - 增加 pretrain 预算是有效的；
  - 但将 phase-2 local meta 从 500 继续拉长到 1000，并没有进一步帮助 `Proposed`，反而更像在部分场景下冲掉了 federated prior 的优势。
- 当前最合理的二级判断更新为：
  - 相比 `Pilot B`，`Pilot A` 更支持“问题主要不在 pretrain 不足，而在 local meta 过长导致 phase interaction 变差”；
  - 如果后续继续向 full-budget 推进，不应默认 `PROPOSED_META_EPOCHS` 和 `META_ONLY_META_EPOCHS` 必须与 pretrain 同比例继续增大。

### 2026-03-20 - Pilot C 结果：`Proposed` 已在总体主指标上反超 `Local_Meta_Transfer`
- `Pilot C` 设置：`PRETRAIN_EPOCHS=4000, PROPOSED_META_EPOCHS=500, META_ONLY_META_EPOCHS=500, FEW_SHOT_EPOCHS=20`。
- 当前 `multi_station_performance.csv` 显示，`Overall_Average` 的 `nMAE/nRMSE` 上：
  - `Proposed` 对 `Local_Meta_Transfer` 已达到 `6胜2负`；
  - `Proposed` 对 `Transfer_Learning` 为 `8胜0负`；
  - `Local_Meta_Transfer` 对 `Meta_Learning / Local_PreTraining` 仍为 `8胜0负`，对 `Transfer_Learning` 为 `7胜1负`。
- 在 8 个主指标的均值上：
  - `Proposed = 30.2506`
  - `Local_Meta_Transfer = 31.8002`
  - `Transfer_Learning = 33.3667`
  - `Meta_Learning = 36.8421`
  - `Local_PreTraining = 38.0802`
  说明当前 `Proposed` 已成为主表中绝对指标最优的方法。
- 与 `Pilot A/B` 对照：
  - `Proposed` 的 8 指标均值在 `A/B/C` 中以 `Pilot C` 最优；
  - `Proposed - Local_Meta_Transfer` 的平均边差由 `Pilot A=-0.6568`, `Pilot B=+0.0701` 改善为 `Pilot C=-1.5496`。
- 当前剩余短板主要集中在 `Frost`：
  - `Overall_Average` 上 `Frost_nMAE` 与 `Frost_nRMSE` 仍落后于 `Local_Meta_Transfer`；
  - 分站看 `station58/59/60` 的 `Frost` 仍是 `Proposed` 的一致弱项。
- 现阶段可以确认：
  - 增加 federated pretrain 预算、同时保持 local meta 较短（500）是有效方向；
  - 相比 `Pilot B`，继续拉长 local meta 并不如继续强化 pretrain 更有利于 `Proposed > Local_Meta_Transfer`。

### 2026-03-20 - Pilot D 结果：`PRETRAIN_EPOCHS=8000` 并未继续改善 `Proposed` 的综合排序
- `Pilot D` 设置：`PRETRAIN_EPOCHS=8000, PROPOSED_META_EPOCHS=500, META_ONLY_META_EPOCHS=500, FEW_SHOT_EPOCHS=20`。
- 与 `Pilot C` 相比，`Proposed` 的 8 个主指标均值由 `30.2506` 变为 `30.3756`，略有退化；但 `Local_Meta_Transfer` 由 `31.8002` 改善到 `31.3504`，导致 `Proposed - Local_Meta_Transfer` 的平均边差从 `-1.5496` 收窄到 `-0.9748`。
- 口径上出现“均值仍优、逐项胜负变差”的分化：
  - `Overall_Average` 的 `nMAE/nRMSE` 上，`Proposed` 对 `Local_Meta_Transfer` 从 `Pilot C` 的 `6胜2负` 退化为 `2胜6负`；
  - 但由于 `ColdWave_nMAE / ColdWave_nRMSE` 上的领先幅度明显扩大，`Proposed` 的 8 指标均值仍然优于 `Local_Meta_Transfer`。
- 因此当前更合理的判断是：
  - `PRETRAIN_EPOCHS=4000` 比 `8000` 更像当前结构下的甜点区；
  - 继续单纯拉长 federated pretrain 并不是单调有益，反而会把优势进一步集中到 `ColdWave`，同时恶化 `HighWind / HighTemperature / Frost` 的多项指标。
- 若后续继续推进正式实验，当前优先候选应仍是 `Pilot C` 配置，而不是 `Pilot D`。

### 2026-03-20 - 两主创新点第一版骨架已落地：工况感知联邦预训练 + 先验保持型本地元训练
- `DemoModelTraining.py` 已新增 `Phase 1` 的工况感知联邦预训练骨架：
  - 新增 `env_float(...)` 与 4 个关键超参：`FED_PRETRAIN_REGIME_ALPHA`、`FED_PRETRAIN_AGGREGATION_GAMMA`、`PROPOSED_META_SHARED_ANCHOR_BETA`、`PROPOSED_META_SHARED_LR_SCALE`；
  - 新增 `compute_regime_sample_weights(...)` 与 `weighted_mse_loss(...)`；
  - `client_local_pretrain_update(...)` 现在会对 federated pretrain 客户端样本按 conventional-weather 的波动/稀有/边界程度加权，并输出 `regime_factor` 与 `aggregation_weight`；
  - `server_aggregate_client_states(...)` 已从纯样本数加权切到 `aggregation_weight` 聚合。
- `DemoModelTraining.py` 已新增 `Phase 2` 的先验保持型本地元训练骨架：
  - 新增 `build_meta_optimizer(...)`，将 `fore_baselearner` 视为共享参数、`LWP` 视为本地元适配参数；
  - 新增 `compute_shared_anchor_loss(...)`；
  - `run_local_meta_training(...)` 已支持 `shared_anchor_beta` 与 `shared_lr_scale`；
  - 仅 `Proposed` 分支启用 `shared_anchor_beta=PROPOSED_META_SHARED_ANCHOR_BETA` 与 `shared_lr_scale=PROPOSED_META_SHARED_LR_SCALE`，`Local_Meta_Transfer / Meta_Learning` 保持原语义。
- 已补新 AST 测试 `tests/test_regime_prior_coupling_ast.py`，并通过：
  - `python -m unittest WPF-under-extreme-weather-main.tests.test_regime_prior_coupling_ast -v`
- 已通过静态验证：
  - `python -m unittest WPF-under-extreme-weather-main.tests.test_regime_prior_coupling_ast WPF-under-extreme-weather-main.tests.test_local_ablation_matrix_ast WPF-under-extreme-weather-main.tests.test_runtime_env_config_ast -v`
  - `python -m py_compile WPF-under-extreme-weather-main/DemoModelTraining.py WPF-under-extreme-weather-main/generate_multi_station_results.py`
- 已在 `/tmp/wpf_regime_smoke` 做 `1/1/1/1` CPU smoke：
  - `USE_FEDERATION=1 TRAIN_META_ONLY_BASELINE=1 PRETRAIN_EPOCHS=1 PROPOSED_META_EPOCHS=1 META_ONLY_META_EPOCHS=1 FEW_SHOT_EPOCHS=1 python DemoModelTraining.py`
  - `STRICT_PAPER_ORDER=0 python generate_multi_station_results.py`
  - 训练链路与 5 组主表生成均跑通；但 smoke 结果上 `Proposed` 仍未压过 `Local_Meta_Transfer`，这只说明结构正确，不代表中预算/正式预算性能结论。

### 2026-03-20 - 两主创新点首轮 4090 结果：`Proposed` 继续压过 `Local_Meta_Transfer`，平均边差优于旧 `Pilot C`
- 在 `Pilot C` 预算上引入两主创新点（工况感知联邦预训练 + 先验保持型本地元训练）后，当前 `multi_station_performance.csv` 的 `Overall_Average` 主指标显示：
  - `Proposed` 对 `Local_Meta_Transfer` 维持 `6胜2负`；
  - `Proposed` 对 `Transfer_Learning` 为 `8胜0负`；
  - `Local_Meta_Transfer` 对 `Transfer_Learning / Meta_Learning / Local_PreTraining` 仍分别为 `7胜1负 / 8胜0负 / 8胜0负`。
- 8 个主指标均值上：
  - `Proposed = 30.2474`
  - `Local_Meta_Transfer = 32.1132`
  - `Transfer_Learning = 33.9817`
  - `Meta_Learning = 36.8420`
  - `Local_PreTraining = 38.4174`
- 相比旧 `Pilot C`：
  - `Proposed` 的均值由 `30.2506` 轻微改善到 `30.2474`；
  - `Proposed - Local_Meta_Transfer` 的平均边差由 `-1.5496` 扩大到 `-1.8658`；
  - 说明新两主创新点至少在总体均值上是正向的。
- 结构性结论：
  - 优势仍主要集中在 `HighWind / HighTemperature / ColdWave`；
  - `Frost_nMAE / Frost_nRMSE` 仍落后于 `Local_Meta_Transfer`，说明新机制尚未补齐 `Frost` 这一顽固短板；
  - 分站看 `station58=6胜2负, station59=5胜3负, station60=4胜4负`，`station60` 仍是最难站点。
- 当前更合理的下一步不是继续大改 epoch，而是围绕 4 个新超参做有针对性的 4090 中预算调参，优先观察 `Frost` 与 `station60` 是否改善。

### 2026-03-20 - `T3 -> T5` 调参结果：应继续围绕 `Phase 2` 调，而不是软化 `Phase 1`
- 在固定主预算 `PRETRAIN=4000, PROPOSED_META=500, META_ONLY_META=500, FEW_SHOT=20` 下：
  - `T3 = (alpha=1.0, gamma=0.5, anchor_beta=0.005, shared_lr_scale=0.5)`；
  - `T5 = (alpha=0.5, gamma=0.25, anchor_beta=0.005, shared_lr_scale=0.5)`。
- 当前结果显示：
  - `T3` 的 `Overall_Average` 上，`Proposed vs Local_Meta_Transfer` 仍为 `6胜2负`，但 8 指标均值边差扩大到 `-1.9873`，优于此前首轮 4090 结果的 `-1.8658`；
  - `T3` 的 `Frost_nMAE / Frost_nRMSE` 差值从此前的 `+2.1090 / +1.7011` 缩小到 `+1.6956 / +1.1942`；
  - `station59` 的 `Proposed vs Local_Meta_Transfer` 由 `5胜3负` 提升到 `6胜2负`，`station60` 仍为 `4胜4负`。
- `T5` 虽然同样维持 `6胜2负`，但均值边差仅为 `-1.6026`，明显劣于 `T3`；说明在当前阶段同时软化 `Phase 1`（降低 `alpha/gamma`）并没有带来额外收益，反而削弱了 `Proposed` 对 `Local_Meta_Transfer` 的整体优势。
- 因而下一步更合理的调参方向是：
  - 固定 `Phase 1` 在 `alpha=1.0, gamma=0.5`；
  - 继续围绕 `anchor_beta` 和 `shared_lr_scale` 做 `Phase 2` 的细调；
  - 重点继续观察 `Frost` 与 `station60` 是否改善。
