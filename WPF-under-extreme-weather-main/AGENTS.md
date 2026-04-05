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
  - 长轮次或正式版本训练（例如上万轮、完整预算、最终汇报口径）默认**不**由代理在当前会话里代跑；代理只提供可直接执行的终端命令，由用户在远程的 `RTX 4090` 终端中运行。
- **设备口径**：回答用户或记录实验时，必须显式区分：
  1. `CPU smoke / debug validation`
  2. `4090 formal run`
- **默认行为**：当前会话若检测到无 CUDA，可继续做小步验证；但进入正式训练前，必须提醒用户切回 4090 环境。
- **算力优先级约束**：
  - 算力等价于时间和金钱；实验设计默认优先复用已有结果，避免重复跑已存在的等价配置。
  - 对 subsampling 类实验，若未显式设置 `CONVENTIONAL_SUBSAMPLE_SEED_OFFSET`，默认视为 `seed0`；此前已跑过的 `R70 / R50 / R30` 就是各自的 `seed0`，不应重复执行。
  - `R100` 在 `CONVENTIONAL_RATIO=1.0` 下不发生 subsampling，因此不需要做 subsampling multi-seed 复验；multi-seed 仅用于 `R70 / R50 / R30` 这类实际发生抽样的设定。
  - 在未证明“更多实验能改变决策”的前提下，默认先跑最小区分度矩阵，而不是全因子穷举。

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

### 2026-03-21 - conventional-ratio 必要性实验基础设施已接通
- 新增 `CONVENTIONAL_RATIO` 开关，当前只支持 `1.0 / 0.7 / 0.5 / 0.3`，目的不是改主方法，而是验证“当单站 conventional source knowledge 不足时，联邦共享先验是否变得必要”。
- 这次实现只缩减 `conventional` 数据，不动 `extreme few-shot` 与 `test`：
  - `pretrain` 侧缩减 `clients_train_data`；
  - `meta-train` 侧缩减 `all_stations_full_data[*]['p_conven_class'/'nwp_conven_class']`；
  - `p_extre / nwp_extre / test_input / test_target` 保持完整。
- pretrain 缩减策略不是“所有场站统一缺同一段时间”，而是：
  - 每个场站把 conventional 样本按时间顺序分成 `10` 个连续 bin；
  - 每个 bin 内按相同比例独立随机抽样；
  - 各站遵循同一规则、同一 ratio，但具体缺失片段不同，从而保留“跨站互补”的必要性检验。
- meta-train 缩减策略是按 class 分层抽样，并强制每类至少保留 `20` 个样本段，以兼容当前 `10 support + 10 query` 的 episodic protocol；因此首轮必要性实验建议只做 `100% / 70% / 50% / 30%`，不直接做 `20% / 10%`。
- 已完成验证：
  - AST 测试通过：`test_conventional_ratio_necessity_ast.py`
  - `py_compile` 通过
  - `CONVENTIONAL_RATIO=0.3` 的 `1/1/1/1` CPU smoke 跑通，训练与 `generate_multi_station_results.py` 链路均未被破坏。
- smoke 结果本身不用于性能判断；它只确认：后续可以在 4090 上系统跑 `ratio ∈ {1.0, 0.7, 0.5, 0.3}` 的必要性实验矩阵，重点比较 `Proposed` 与 `Local_Meta_Transfer`。

### 2026-03-21 - conventional-ratio 必要性实验首轮结果：联邦必要性得到正向支持
- 以 `T3` 作为 `R100` 基线后，`R70 / R50 / R30` 的 `Overall_Average` 上，`Proposed` 对 `Local_Meta_Transfer` 的 8 主指标平均边差分别为：
  - `R100 = -1.9873`
  - `R70 = -2.6114`
  - `R50 = -2.3560`
  - `R30 = -2.7728`
- 也就是说，三个低资源设定都比全数据设定更有利于 `Proposed > Local_Meta_Transfer`；其中 `R30` 最强，说明当单站 conventional source knowledge 明显缩减时，跨站联邦共享先验的价值在增强。
- 逐项胜负在 `Overall_Average` 上始终维持 `6胜2负`，尚未翻成 `7胜1负/8胜0负`；但结构性短板 `Frost` 在 `R50/R30` 下明显缩小，且 `station60` 从 `R100` 的 `4胜4负` 改善到 `R30` 的 `5胜3负`，说明必要性信号已经开始从总体均值扩展到更难站点。
- 需要谨慎表述的点：
  - 当前每个 ratio 只有单次 subsampling，不足以作为最终论文级证据；
  - `R70/R30` 的绝对误差优于 `R100`，说明缩减不仅制造了“低资源”，也可能剔除了部分负迁移的 conventional segments；因此目前更适合表述为“联邦共享先验在 local conventional knowledge 不完整时更有价值”，而不是简单写成“数据越少联邦越必要”的严格单调定律。
- 当前更合理的下一步不是继续加 ratio，而是对 `R100 / R50 / R30` 做 2-3 个不同 subsampling seed 的复验，确认该必要性趋势不是单次抽样偶然。

### 2026-03-21 - conventional-ratio 多 seed 复验开关已接通
- 新增 `CONVENTIONAL_SUBSAMPLE_SEED_OFFSET` 环境变量，用于控制 conventional-ratio 实验中的 subsampling 随机种子偏移。
- 该开关只影响 `CONVENTIONAL_RATIO < 1.0` 的缩减实验：
  - `R100` 不做 subsampling，因此不需要 multi-seed；
  - `R70 / R50 / R30` 可通过改变 `CONVENTIONAL_SUBSAMPLE_SEED_OFFSET` 做复验。
- 当前实现中：
  - pretrain 的时间分箱抽样和 meta 的 class 分层抽样都共享这一 seed offset；
  - 训练主随机种子未改动，因此该开关专门用于验证“subsampling 偶然性”，不混入额外训练随机性。

### 2026-03-21 - R30 多 seed 结果：必要性趋势基本稳住，但仍不是“全面稳压”
- `R30` 的 `seed0 / seed1 / seed2` 在 `Overall_Average` 上，`Proposed - Local_Meta_Transfer` 的 8 主指标平均边差分别为：
  - `seed0 = -2.7728`
  - `seed1 = -2.1473`
  - `seed2 = -3.9652`
  - 均值约 `-2.9618`，三个 seed 全部优于 `R100` 的 `-1.9873`。
- 三个 seed 的共同结构：
  - `Overall_Average` 上均维持 `6胜2负`；
  - `Frost_nMAE / Frost_nRMSE` 依然没有翻正，但都明显优于 `R100`；
  - `station59` 始终强势，`station60` 从 `seed1` 的 `4胜4负` 到 `seed2` 的 `6胜2负`，说明困难站点的联邦受益方向是正的，但强度存在抽样敏感性。
- 这组结果已经足以支撑更稳的必要性表述：
  - 当 local conventional knowledge 明显不完整时，联邦共享先验的价值会稳定增强；
  - 但该增强目前更多体现为总体边差扩大，而不是所有天气/所有站点都全面翻盘，因此不宜夸大为“全面稳压 LMT”。
- 若继续节省算力，当前比起把 `R50` 也做满多 seed，更优先的做法是：
  - 先把 `R30` 作为低资源必要性的核心证据；
  - 视需要再补 `R50` 的 `seed1/seed2`，作为中度低资源的辅助稳健性验证。

### 2026-03-21 - R30-seed2 的 budget reopen（M4000/P3000）：Frost 明显改善，但整体边差缩小
- 在 `R30-seed2` 下将 `PRETRAIN_EPOCHS=4000` 保持不变，并把 `PROPOSED_META_EPOCHS/META_ONLY_META_EPOCHS` 从 `500` 拉到 `3000` 后：
  - `Overall_Average` 上 `Proposed - Local_Meta_Transfer` 的 8 主指标平均边差从 `-3.9652` 收缩到 `-2.5146`；
  - 逐项胜负仍为 `6胜2负`，但两项 `Frost` 差值从 `+1.3319 / +0.6075` 显著缩小到 `+0.0582 / +0.1537`，几乎翻正；
  - `station58` 与 `station59` 都从原来的 `6胜2负` 提升为 `8胜0负`，`station60` 为 `6胜2负`。
- 关键解释：这次 reopen 并不能直接证明“Proposed 单边 under-budget”，因为当前代码里 `Local_Meta_Transfer` 也共享 `PROPOSED_META_EPOCHS`（见 `DemoModelTraining.py:1054-1060`），所以把 meta 预算从 `500` 拉到 `3000` 会同时强化 `Proposed` 和 `LMT`。
- 因此，该实验更准确的结论是：增加 meta budget 会明显改善剩余短板（尤其是 `Frost`），但同时也会显著增强 `LMT`，导致总体平均边差缩小。若下一步要判断 `Proposed` 是否单边 under-budget，需要将 `Proposed` 与 `LMT` 的 meta epoch 控制解耦，再做针对性实验。
- 节省算力原则下，不建议继续在当前“耦合 epoch”设定上盲目拉长；更优先的改动是：增加独立的 `LOCAL_META_TRANSFER_EPOCHS` 配置，再用最小矩阵验证 `Proposed` 专属 budget 是否还能把 `Frost` 翻正而不同时放大 `LMT`。

### 2026-03-21 - R30-seed2 下的“合成最小矩阵”已经基本回答了 Proposed 单边 meta budget 问题
- 由于当前代码固定了全局训练随机种子 `1029`，且 `R30-seed2` 的 subsampling 条件也固定（`CONVENTIONAL_RATIO=0.3, CONVENTIONAL_SUBSAMPLE_SEED_OFFSET=2`），因此可用两次运行构造一个近似受控的“合成最小矩阵”：
  - `Proposed(meta=3000)` 取自 `budget_reopen_runs/R30_seed2_M4000_P3000/multi_station_performance.csv`
  - `Local_Meta_Transfer(meta=500)` 取自 `necessity_runs/R30_seed2/multi_station_performance.csv`
- 该合成比较在 `Overall_Average` 上得到：
  - 8 主指标平均边差 `-4.0237`（优于原始 `R30-seed2` 的 `-3.9652`），但逐项胜负仍为 `6胜2负`，并没有变成 `7胜1负/8胜0负`；
  - 两项剩余短板仍是 `Frost_nMAE=+0.9348`、`Frost_nRMSE=+0.1398`。
- 这说明：
  - “上次 reopen 之所以没扩大总体优势，仅仅因为 LMT 也被一起强化”并不是全部解释；
  - 即使只看 `Proposed` 单边把 meta budget 从 `500` 拉到 `3000`，在当前算法结构下也仍然不能实现对 `LMT` 的全面翻盘。
- 因此，若目标是 `Proposed vs LMT` 达到全赢状态，继续单纯增加 `Proposed` 的 meta 轮数已经不够有把握；更合理的后续方向应是：
  - 继续提升 pre-train 阶段对 `Frost`/困难工况的针对性，或
  - 修改 local meta 的 task/更新设计，而不是仅依赖当前 budget 扩张。

### 2026-03-22 - 修正：当前 3 场站数据中的 Frost 并不像根基论文 Table II 那样占 7.30%
- 先前基于根基论文 Table II（单场站）曾推断 `Frost` 在当前数据中“没那么 few-shot”，这一外推不严谨，现已纠正。
- 直接按当前项目的 `.mat` 数据统计 12-step segment 占比：
  - `station58`: `Frost = 50 / 1460 = 3.4247%`
  - `station59`: `Frost = 32 / 1460 = 2.1918%`
  - `station60`: `Frost = 38 / 1460 = 2.6027%`
- 这说明在当前 3 场站设定下，`Frost` 仍然属于低占比事件，并不能简单用根基论文中的 `7.30%` 来解释“为什么 Frost 上 LMT 更强”。
- 因而当前更合理的解释应转向：
  - `station60` 的 Frost 构造/分布特性更特殊，或
  - 当前联邦先验/元训练对该站 `Frost` 的迁移仍不足，或
  - 预算不足仍在起作用；
  而不能再用“Frost 在当前数据里并不稀缺”作为主要论据。

### 2026-03-22 - Frost 诊断：当前剩余问题更像 station60 特异性，而不是 Frost 类别本身
- 直接按当前项目 `.mat` 数据统计的 12-step segment 占比：
  - `station58`: `Frost = 50/1460 = 3.4247%`
  - `station59`: `Frost = 32/1460 = 2.1918%`
  - `station60`: `Frost = 38/1460 = 2.6027%`
- 这说明在当前 3 场站设定下，`Frost` 仍然是低占比事件；同时 `station59` 的 Frost 比 `station60` 更少，却没有成为剩余瓶颈，因此“Frost 样本不够多/不够 few-shot”并不能解释当前问题。
- 进一步的分布诊断表明：
  - `station60` 的 Frost 相对本站 conventional 的整体偏移（基于扁平化 NWP+power 的平均绝对 z-score）略高于 `58/59`：`1.0603` vs `1.0382 / 1.0077`，但差异不算大；
  - `station60` 的 Frost 到其最近 conventional class 的距离并不最差（best distance `1.1249`），但该最近类的样本池较小（`82` segments），可能削弱 local meta 对 Frost 邻域的建模稳定性。
- 更关键的是，在 `R30-seed2 -> R30-seed2_M4000_P3000` 的 budget reopen 中：
  - `station58/59` 的 Proposed-Frost 明显改善；
  - `station60` 的 Proposed-Frost 反而轻微恶化（`nMAE +0.3284`, `nRMSE +0.1901`）。
- 因而当前更合理的判断是：剩余难点主要集中在 `station60-Frost` 的站点特异性/数据构造/分布问题，而不是 `Frost` 这一天气类别本身必须做专门算法。预算不足仍可能存在，但已不再是唯一或最直接的解释。

### 2026-03-22 - Proposed 专用 balanced meta sampler 已接通
- 按最小正确方案，不修改根基论文的 `k=10` 与 `k*=5` 协议，只修改 `Proposed` 在 `Phase 2` 中“这 5 个 conventional tasks 怎么抽”。
- 新实现要点：
  - 新增 `PROPOSED_META_SAMPLER_MODE`，默认 `balanced`；
  - `Local_Meta_Transfer` 与 `Meta-only` 保持 `uniform` 抽样，不改变基线定义；
  - `balanced` 抽样使用固定化规则：`weight = size_bonus * coverage_bonus`，其中
    - `size_bonus = sqrt(mean_class_size / class_size)`，避免小类长期被大类淹没；
    - `coverage_bonus = 1 / (1 + exposure_c)`，其中 `exposure_c` 统计最近 `4` 个 meta epoch 是否已抽中过该类，避免短窗口覆盖不足。
- 这样做的目的不是针对 `Frost` 特化，而是在不改变 task 数与 episodic budget 的前提下，提高 `station60` 这类“小邻域 class”在 Proposed 元训练中的覆盖效率。
- 验证结果：
  - 新增 AST 测试 `tests/test_balanced_meta_sampler_ast.py`，先红后绿；
  - 相关 AST 集合与 `py_compile` 均通过；
  - 在 `/tmp/wpf_sampler_smoke` 做 `1/1/1/1 + R30-seed2 + balanced sampler` smoke 后，训练与结果生成链路均跑通。

### 2026-03-22 - R30-seed2 + balanced Proposed meta sampler + meta=3000 达成 Overall_Average 上 Proposed 对 LMT 的 8胜0负
- 运行配置：
  - `CONVENTIONAL_RATIO=0.3`
  - `CONVENTIONAL_SUBSAMPLE_SEED_OFFSET=2`
  - `PRETRAIN_EPOCHS=4000`
  - `PROPOSED_META_EPOCHS=3000`
  - `META_ONLY_META_EPOCHS=3000`
  - `FED_PRETRAIN_REGIME_ALPHA=1.0`
  - `FED_PRETRAIN_AGGREGATION_GAMMA=0.5`
  - `PROPOSED_META_SHARED_ANCHOR_BETA=0.005`
  - `PROPOSED_META_SHARED_LR_SCALE=0.5`
  - `PROPOSED_META_SAMPLER_MODE=balanced`
- 结果文件：`sampler_runs/R30_seed2_balanced_M3000/multi_station_performance.csv`
- 与同 budget 的 uniform sampler 对照 `budget_reopen_runs/R30_seed2_M4000_P3000/multi_station_performance.csv` 相比：
  - `Overall_Average` 8 主指标均值边差（`Proposed - LMT`）从 `-2.5146` 扩大到 `-4.2729`；
  - 逐项胜负从 `6胜2负` 提升为 `8胜0负`；
  - `Frost_nMAE / Frost_nRMSE` 从 `+0.0582 / +0.1537` 翻为 `-0.2237 / -0.3405`。
- 分站结果：
  - `station58 = 8胜0负`
  - `station59 = 7胜1负`
  - `station60 = 8胜0负`
- 关键解释：这强烈支持“剩余瓶颈主要在 Proposed 的 Phase-2 task coverage / neighborhood quality，而不是 Frost 专用算法缺失”。在保持 `k=10, k*=5` 不变的前提下，仅通过 Proposed 专用的 balanced meta sampler（`size_bonus * coverage_bonus`）就把 low-resource `R30-seed2` 场景下的 `Overall_Average` 推到了对 LMT 的全面占优。
- 后续判断：
  - 对 low-resource 论文主叙事，当前最强配置应优先采用 `balanced sampler + meta=3000`；
  - 不再优先考虑继续盲目增加 epoch；如需追加算力，优先做该配置的复验而不是先改大结构。

### 2026-03-22 - R30-seed2 + balanced sampler 在 M8000/P8000 下发生全面反转，不宜直接把 high-budget 当最终版
- 运行配置：`CONVENTIONAL_RATIO=0.3, CONVENTIONAL_SUBSAMPLE_SEED_OFFSET=2, PRETRAIN_EPOCHS=8000, PROPOSED_META_EPOCHS=8000, META_ONLY_META_EPOCHS=8000, FED_PRETRAIN_REGIME_ALPHA=1.0, FED_PRETRAIN_AGGREGATION_GAMMA=0.5, PROPOSED_META_SHARED_ANCHOR_BETA=0.005, PROPOSED_META_SHARED_LR_SCALE=0.5, PROPOSED_META_SAMPLER_MODE=balanced`。
- 结果文件：`final_runs/R30_seed2_balanced_M8000_P8000/multi_station_performance.csv`。
- 与当前最强的 `sampler_runs/R30_seed2_balanced_M3000/multi_station_performance.csv` 相比：
  - `Overall_Average` 上 `Proposed - LMT` 8 主指标均值边差从 `-4.2729` 反转为 `+4.3742`；
  - 逐项胜负从 `8胜0负` 反转为 `0胜8负`；
  - `station58 = 0胜8负`，`station59 = 0胜8负`，`station60 = 2胜6负`。
- 根因拆解：
  - `Proposed` 自身相对 `M3000` 只中等幅度变差（8 主指标均值 `+1.0854`，越小越好）；
  - `Local_Meta_Transfer` 却大幅变强（8 主指标均值 `-7.5617`）。
- 当前最合理解释不是“balanced sampler 失效”，而是：
  - 在当前耦合设定下，同时把 `PRETRAIN` 和两条 meta 分支都拉到 `8000`，会让 `LMT` 获得更强的本地常规知识+本地元训练收益；
  - `Proposed` 则可能受到更强 federated prior、anchor 约束以及 balanced task exposure 的共同影响，出现非单调退化。
- 结论：
  - 不能再假设“高预算一定更好”；
  - 当前 low-resource 最强点仍是 `balanced sampler + PRETRAIN=4000 + META=3000`；
  - 后续若要继续诊断 high-budget 失效，应优先做“单变量 reopen”（只增 pretrain 或只增 meta），不要再一次性同步拉高两者。

### 2026-03-22 - 已实现 full-conventional-data 下的 regime-missing / class-dropout stress protocol
- 目标：在不改主算法的前提下，用更对症的 necessity stress test 替代单纯 `CONVENTIONAL_RATIO` 压样本密度的协议。
- 新协议要点：
  - `CONVENTIONAL_RATIO=1.0` 恢复全 conventional 数据；
  - 新增 `REGIME_MISSING_MODE=class_dropout`；
  - 固定互补 drop map：
    - `station58`: drop `{1,2,3,4}`
    - `station59`: drop `{4,5,6,7}`
    - `station60`: drop `{7,8,9,10}`
  - 约束：不存在任何一个 class 在三个场站同时缺失；
  - regime-missing 同时作用于：
    - `Phase 1` 的本地 conventional pretrain pool（通过保留类重建 pretrain pool）
    - `Phase 2` 的本地 conventional meta task pool；
  - extreme few-shot 和 test 数据保持不变。
- 采样协议：
  - 4-class dropout 后每站剩 `6` 类；
  - 新增 `resolve_local_meta_tasks_per_epoch(...)`，在 regime-missing 模式下采用“每轮半覆盖”逻辑，因此本地 `k*=3`；
  - `Proposed` 仍走 balanced sampler，`LMT`/`Meta-only` 仍走 uniform sampler。
- 实现文件：`DemoModelTraining.py`，设计文档：`docs/plans/2026-03-22-regime-missing-design.md`。
- 测试与验证：
  - 新增 `tests/test_regime_missing_stress_ast.py`；
  - 相关 AST 套件与 `py_compile` 全部通过；
  - 在 `/tmp/wpf_regime_missing_smoke` 完成 `CONVENTIONAL_RATIO=1.0 + REGIME_MISSING_MODE=class_dropout + 1/1/1/1` CPU smoke，训练与结果生成链路均跑通。

### 2026-03-22 - full conventional + complementary 4-class dropout 并未支持联邦必要性，反而进一步削弱 Proposed 必要性
- 运行配置：`CONVENTIONAL_RATIO=1.0, REGIME_MISSING_MODE=class_dropout, PRETRAIN_EPOCHS=4000, PROPOSED_META_EPOCHS=3000, META_ONLY_META_EPOCHS=3000, PROPOSED_META_SAMPLER_MODE=balanced`。
- 结果文件：`regime_missing_runs/fullconv_classdrop_m3000/multi_station_performance.csv`。
- `Overall_Average` 上：
  - `Proposed = 23.2814`
  - `Local_Meta_Transfer = 21.0303`
  - 边差 `Proposed - LMT = +2.2512`，即 `Proposed vs LMT = 2胜6负`。
- 分站：
  - `station58 = 1胜7负`
  - `station59 = 2胜6负`
  - `station60 = 4胜4负`
- 与全数据无 dropout 的当前强基线 `tune_runs/T3/multi_station_performance.csv` 相比，二者绝对值都显著改善，但 `LMT` 改善更大；与 `R30-seed2 balanced M3000` 相比，更是从 `8胜0负` 反转为 `2胜6负`。
- 当前最合理解释：
  - 这组固定互补 4-class dropout 协议并没有激发出“联邦补齐缺失 regime”的优势；
  - 相反，它更像是同时简化了本地 conventional task pool（每站 10 类降到 6 类，k*=3），而 `LMT` 比 `Proposed` 更能从这种更干净的本地任务空间中获益；
  - 因此，按当前这版 stress test，联邦必要性没有被支持，反而被进一步削弱。
- 结论：
  - 不能把这组 regime-missing 结果作为联邦必要性证据；
  - 若继续推进 necessity 叙事，需要重新审视协议是否真正制造了“单站缺失、跨站可补”的条件，而不是先简化了本地任务空间。

### 2026-03-23 - 长会话总括索引
- 本次长会话的完整整理文档见：`docs/plans/2026-03-23-session-summary.md`。
- 需要优先记住的三件事：
  1. 两个主创新点已形成并实现：
     - `Phase 1`: regime-aware federated pre-training；
     - `Phase 2`: prior-preserving local meta-training。
  2. Proposed-only balanced Phase-2 sampler 是本次最关键的结构性改进，其公式为：
     - `coverage_bonus(c) = 1 / (1 + exposure_c)`
     - `size_bonus(c) = sqrt(mean_class_size / class_size_c)`
     - `weight_c = coverage_bonus(c) * size_bonus(c)`
     - 固定短窗 `W=4`。
  3. 当前结论是分裂的：
     - 在 `R30-seed2 + balanced sampler + PRETRAIN=4000 + META=3000` 下，`Proposed vs LMT = 8胜0负`，这是最强正结果；
     - 但 full-conventional complementary 4-class dropout stress test 给出 `2胜6负`，削弱了“联邦必要性”的总体主张。
- 新窗口续接时，应先读 `docs/plans/2026-03-23-session-summary.md`，再读本文件最近几段实验记录。

### 2026-03-23 - 方法核心公式与理论解释（优先记忆）
- 这次会话真正需要优先保留的不是某一组跑分，而是 Proposed 的两条主创新线及其数学动机。

#### 主创新点 1：Regime-Aware Federated Pre-Training（Phase 1）
- 目标：让 federated prior 不再是对所有 conventional segments 一视同仁的平均结果，而更偏向 hard / rare / boundary conventional regimes。
- 样本级权重：
  - 对每个 conventional segment `i`，先构造 regime score：
    - `ramp_score_i`：功率序列相邻步长平均绝对变化；
    - `volatility_score_i`：功率序列标准差；
    - `rarity_score_i`：输入特征相对站内均值方差归一化后的平均绝对偏离。
  - 原始分数：`s_i = ramp_score_i + volatility_score_i + rarity_score_i`
  - 归一化后：`s_i_tilde`
  - 最终 sample weight：`w_i = 1 + alpha * s_i_tilde`
  - 其中 `alpha = FED_PRETRAIN_REGIME_ALPHA`。
- 客户端预训练损失可写成加权 MSE：
  - `L_pre_client = sum_i w_i * ||f_theta(x_i) - y_i||^2 / sum_i w_i`
- 服务器聚合不再是 plain FedAvg，而是 regime-aware client weighting：
  - 先从客户端得到 `regime_factor_s`；
  - 聚合权重：`rho_s ∝ n_s * clip(1 + gamma * (regime_factor_s - 1), 0.5, 2.0)`
  - 其中 `gamma = FED_PRETRAIN_AGGREGATION_GAMMA`。
- 理论解释：
  - 如果直接 FedAvg，就等于默认所有 station 的所有 conventional segments 对 future extreme few-shot 同等重要；
  - 这和之前 Pilot A/B/C/D 的非单调现象不符；
  - 因而需要让联邦 prior 更偏向“对 downstream adaptation 更有价值”的 conventional regimes。

#### 主创新点 2：Prior-Preserving Local Meta-Training（Phase 2）
- 目标：解决 local meta-training 覆盖/洗掉 federated prior 的问题。
- 诊断来源：早期 pilot 已表明 meta 轮数拉长时，旧 Proposed 容易从相对优势退化，说明本地元训练会冲掉联邦共享先验。
- 新目标函数：
  - `L_meta = L_query + beta * ||theta_shared - theta_fed||_2^2`
  - 其中：
    - `theta_fed`：Phase 1 输出的 federated shared prior；
    - `theta_shared`：当前 local meta 中共享参数；
    - `beta = PROPOSED_META_SHARED_ANCHOR_BETA`。
- 优化器层面的配套约束：
  - 共享参数学习率缩放：`lr_shared = shared_lr_scale * lr_base`
  - 其中 `shared_lr_scale = PROPOSED_META_SHARED_LR_SCALE`。
- 理论解释：
  - Proposed 与 LMT 的差别不应只停留在“初始化来源不同”；
  - 如果 Phase 2 不显式保护 federated prior，那么 local meta 最终会把 Proposed 拉回接近 LMT 的本地解；
  - 因此，Phase 2 必须让“联邦先验被消费且不被轻易覆盖”。

#### 次级但关键的结构改进：Proposed-only balanced Phase-2 sampler
- 这是后续把 low-resource `R30-seed2` 推到 `Overall_Average 8胜0负` 的直接原因。
- 对每个 local conventional class `c` 定义：
  - `coverage_bonus(c) = 1 / (1 + exposure_c)`
  - `size_bonus(c) = sqrt(mean_class_size / class_size_c)`
  - `weight_c = coverage_bonus(c) * size_bonus(c)`
- 其中 `exposure_c` 统计最近 `W=4` 个 meta epochs 中该类出现的次数。
- `W=4` 的解释：原协议 `k=10, k*=5`，均匀抽样下某类连续 4 轮不被抽中的概率约为 `(1-0.5)^4 = 0.0625`，因此可把 `W=4` 看成短期持续欠覆盖的检测窗口，而不是对一轮随机波动的过度反应。
- 理论解释：
  - 该 sampler 不是 Frost 特化；
  - 它解决的是 local task coverage / neighborhood quality 问题，尤其是在 station60 这类局部邻域不稳的场景下。

#### 当前最诚实的总体判断
- 上述三部分共同构成了本次会话的算法主线；
- 其中 Phase 1 和 Phase 2 是主创新点，balanced sampler 是后续关键改进；
- 低资源 `R30` 协议下，这套设计能把 Proposed 推到对 LMT 的强优势；
- 但 broader federated necessity claim 仍未被最终证明，因为 full-conventional complementary 4-class dropout stress test 结果为 `2胜6负`。

### 2026-03-23 - CDRM 与两主创新点的结合方式（重要）
- 根基论文明确指出：`CDRM` 应同时作用于 pre-train 和 meta-train，见 `(19)(20)(21)`；因此在当前多场站方法中，更合理的理解不是“CDRM 被替代了”，而是“两个主创新点建立在 CDRM 底座之上”。
- 更干净的公式化写法应为：
  - `Phase 1`: `L_s^pre = L_s^{weighted-pred} + lambda_pre * L_{CDRM,s}^{pre}`
    - 其中 `L_s^{weighted-pred} = (1 / sum_t w_{s,t}) * sum_t w_{s,t} * l(f_theta(x_{s,t}), y_{s,t})`
    - `w_{s,t} = 1 + alpha * r_{s,t}`
    - `L_{CDRM,s}^{pre}` 对应根基论文 (18) 的跨 task / minibatch gradient consistency 正则；当前代码实现上由 `penalty(...)` 近似承载。
  - `Phase 2 support`: `L_s^{SS} = L_s^{support} + lambda_meta * L_{CDRM,s}^{SS}`
  - `Phase 2 query`: `L_s^{SQ} = L_s^{query} + lambda_meta * L_{CDRM,s}^{SQ} + beta * ||theta_sh - theta_fed||_2^2`
- 理论解释：
  - `CDRM` 负责跨 task 不变性 / cross-task generalization；
  - `regime-aware pretrain` 负责强调哪些 conventional segments 对 downstream extreme adaptation 更重要；
  - `prior-preserving meta` 负责避免本地元训练洗掉 federated prior；
  - 三者是正交的，不应互相替代。
- 重要实现现状：
  - 当前代码里 `Phase 2` 已经在 `use_cdrm=True` 时把 `penalty(...)` 加进 support/query 损失；
  - 但 `Phase 1` 的 CDRM 系数来自 `get_pretrain_penalty_weight(...)`，其 warmup 要到 `epoch >= 10000` 才非零；
  - 因此，在本次会话大多数实验预算（`PRETRAIN_EPOCHS=4000` 或 `8000`）下，`Phase 1 CDRM` 实际上是关闭的。
- 这意味着：
  - 当前很多关于“regime-aware federated pretraining”的实验，并没有真正测试“Regime-aware + CDRM”联合设计；
  - 若后续要重新评估方法本体，应优先重构 `Phase 1` 的 `CDRM` 权重调度，而不是简单再加一个新损失名词。

### 2026-03-23 - 最早期 git 版本中的 CDRM 实现是否与论文公式完全一致（重要）
- 已核查最早期 `git` 历史：从 `first commit 80e6266` 开始，代码中的 `CDRM` 就不是对论文 `(18)(19)(20)(21)` 的逐字实现，而是一个代理实现。
- 最早期实现与当前核心结构一致：
  - `penalty(logits, y)` 定义为：
    - 先引入辅助标量 `scale = 1.0`；
    - 再把一个 batch 划分为奇偶两半，分别计算 `loss1` 和 `loss2`；
    - 对 `scale` 求两半 loss 的梯度，并返回二者点积：`sum(grad_batch1 * grad_batch2)`。
  - 预训练和元训练都采用 `loss_en = k * penalty(...) + mse(...)` 的写法。
- 因此，代码从一开始就是“梯度一致性 penalty 的工程代理”，而不是论文符号层面的原式翻译。
- 为什么只能说“作用机理对应”，而不能说“公式写法完全一样”：
  - 论文 `(18)` 的对象是抽象的 base learner `eta`，并写成 `\nabla_{eta | eta = 1.0}`；代码并没有显式单独建模论文式 `eta`，而是用辅助标量 `scale` 作为代理变量。
  - 论文 `(18)` 写的是 sampled tasks 中两个 mini-batches `m,n` 的梯度点积；代码则是在当前张量 batch 内直接按奇偶索引拆成两半。
  - 论文 `(19)(20)(21)` 写的是参数更新规则；代码实现时落成了一个可反传的代理损失 `loss_en = k * penalty + mse`。
- 因此，更准确的表述应为：
  - 代码与论文 `CDRM` 在“通过二阶/梯度一致性促进 invariant features 与 cross-task generalization”的优化机理上对齐；
  - 但在变量定义、task/minibatch 组织方式、以及最终损失写法上，都不是论文原式的严格同构实现。
- 另一个已确认事实：`pre-train` 阶段 `k` 的 warm-up（`<10000 -> 0, <20000 -> 1, <30000 -> 5, else 10`）在最早 `first commit` 就已经存在，并不是后期引入的改动。
- 关于“为什么代码不严格按论文公式写”：目前只能给出合理推断，不能表述为已确认事实：
  - 论文本身已经从原始双层约束化简到 `(17)` 与 `(18)`；
  - Appendix A 又进一步通过固定 `eta=1.0`、线性假设和无偏估计解释，为“可计算代理化”留出了空间；
  - 代码选择 `scale`-gradient 点积，大概率是作者在可计算性、训练稳定性和实现复杂度之间做的工程折中。

### 2026-04-04 - 六客户端季节稀缺协议与动态元任务设计（已确认）
- 已确认弃用“`2022` 全年训练 / `2023` 全年测试”的三场站主协议，改为六客户端季节稀缺协议，用于更直接检验“target 数据不足时，跨客户端 federated prior 是否优于纯本地 prior”。
- 六个实验客户端定义固定为：
  - `WT1 (58)`: train `2022-03-01`~`2022-05-31`, test `2023-03-01`~`2023-05-31`
  - `WT2 (59)`: train `2022-03-01`~`2022-05-31`, test `2023-03-01`~`2023-05-31`
  - `WT3 (60)`: train `2022-06-01`~`2022-08-31`, test `2023-06-01`~`2023-08-31`
  - `WT4 (61)`: train `2022-06-01`~`2022-08-31`, test `2023-06-01`~`2023-08-31`（源 workbook 仍来自 `058`）
  - `WT5 (62)`: train `2022-11-01`~`2023-01-31`, test `2023-11-01`~`2024-01-31`（需拼接 `2024` workbook）
  - `WT6 (63)`: train `2022-11-01`~`2023-01-31`, test `2023-11-01`~`2024-01-31`（需拼接 `2024` workbook）
- 评估口径改为逐 `(client, extreme_class)` 报告，不强制再做总平均；当前预期主任务为：
  - `(58, high_wind)`, `(59, high_wind)`, `(60, high_temp)`, `(61, high_temp)`, `(62, cold_wave)`, `(62, frost)`, `(63, cold_wave)`, `(63, frost)`。
- `Proposed` / `LOCAL_META_TRANSFER` 的对齐口径固定为：
  - `Phase 1/2` 常规天气只用训练块前 `1` 个月 `normal_weather`
  - few-shot support 用目标客户端整个训练块中的 `extreme weather`
  - test 用后一年的匹配季节块 `extreme weather`
- 元训练任务设计已确认改为动态，而不再复用历史全年 `10` 类 + `10/10` episode：
  - 保留 `len_realp = 12`
  - 每个 meta episode 改为 `5 support + 5 query`
  - 聚类只在训练块前 `1` 个月 `normal_weather` 上进行
  - `K` 不设固定上限，而由窗口预算动态推导：`K_max = floor(N_windows / (support + query))`
  - 肘部法仅在 `2..K_max` 内搜索，并要求最小簇窗口数不少于 `support + query`
  - balanced sampler 的每轮采样类数固定为 `max(2, ceil(K / 2))`
- `2024` 数据新增约束：
  - `24jilin_058_processed_4classes.xlsx` / `059` / `060` 已确认覆盖完整 `2024-01-01`~`2024-12-31`
  - 其中 `Power2` 标幺化容量必须按 `58/59/60 = 50/100/300` 处理，不能沿用旧的统一容量假设。

#### 2026-04-04 - 六客户端季节协议已落地到代码与数据资产（已执行）
- 已新增 `build_six_client_seasonal_protocol.py`，可直接从 `2223/24` 原始 `xlsx` 构建六客户端季节协议 `.mat` 资产与 `seasonal_protocol_data/seasonal_protocol_metadata.json`。
- 该构建脚本已完成两项关键兼容修复：
  - 不再依赖 `sklearn`；若环境缺少该依赖，则自动回退到内置 KMeans 实现；
  - `xlsx` 解析已改为按单元格列号对齐，而不是按出现顺序 `zip`，避免因缺失单元格导致 `Power2` 等列被错位丢失。
- 已实际生成 6 个客户端资产，当前动态聚类结果为：
  - `WT1 (58): K=4, sampler_task_count=2`
  - `WT2 (59): K=3, sampler_task_count=2`
  - `WT3 (60): K=3, sampler_task_count=2`
  - `WT4 (61): K=3, sampler_task_count=2`
  - `WT5 (62): K=2, sampler_task_count=2`
  - `WT6 (63): K=3, sampler_task_count=2`
- `DemoModelTraining.py` 已接入 seasonal protocol 开关与 metadata 驱动逻辑：
  - 支持 `SEASONAL_PROTOCOL_ENABLED=1`
  - 支持按 metadata 加载 `58~63` 客户端资产
  - 支持动态 `META_SUPPORT_SHOTS / META_QUERY_SHOTS`
  - 支持按客户端 `valid_class_indices` 和 `sampler_task_count` 运行
- `generate_multi_station_results.py` 已新增 seasonal protocol 专用结果路径：
  - 读取 protocol metadata
  - 逐 `(client_id, extreme_class)` 导出任务级结果
  - 不再强制输出 `Overall_Average` 作为主结果
- 已完成的验证：
  - `tests.test_six_client_seasonal_protocol_ast`：通过
  - `tests.test_generate_results_local_meta_ast`：通过
  - `tests.test_local_ablation_matrix_ast`：通过
  - `SEASONAL_PROTOCOL_ENABLED=1 python generate_multi_station_results.py`：已跑通并成功生成任务级 `multi_station_performance.csv`
- 当前 smoke 结果中的 `61/62/63` 多个任务仍为 `NaN`，原因不是 seasonal protocol 路径错误，而是当前目录下尚无这些新客户端对应的完整 few-shot 模型产物；因此下一步真正需要做的是按新协议重训，而不是继续修结果脚本。

#### 2026-04-04 - seasonal 训练启动器已补齐（已执行）
- 已新增统一入口 `run_six_client_seasonal_protocol.py`，用于替代手工拼接环境变量。
- 启动器支持四个 stage：
  - `build`
  - `train`
  - `eval`
  - `all`
- 启动器默认注入的 seasonal preset：
  - `SEASONAL_PROTOCOL_ENABLED=1`
  - `SEASONAL_PROTOCOL_METADATA_PATH=seasonal_protocol_data/seasonal_protocol_metadata.json`
  - `META_SUPPORT_SHOTS=5`
  - `META_QUERY_SHOTS=5`
- 启动器还支持：
  - `--smoke`：注入 `PRETRAIN_EPOCHS=1`, `PROPOSED_META_EPOCHS=1`, `META_ONLY_META_EPOCHS=1`, `FEW_SHOT_EPOCHS=1`, `STRICT_PAPER_ORDER=0`
  - `--dry-run`：只打印 build/train/eval 命令，不实际执行
- 已新增 `tests/test_seasonal_protocol_launcher_ast.py`，并通过：
  - launcher surface AST
  - launcher stage command AST
- 已完成 dry-run 验证：
  - `python run_six_client_seasonal_protocol.py all --smoke --dry-run`
  - 能正确打印 `build_six_client_seasonal_protocol.py -> DemoModelTraining.py -> generate_multi_station_results.py` 三阶段命令
- 这意味着当前仓库已经具备：
  - seasonal 数据构建脚本
  - seasonal 训练/评估代码路径
  - seasonal 统一启动入口
  - 后续可直接在此入口上补充正式实验预算，而无需再手工维护命令行环境变量。

#### 2026-04-04 - six-client seasonal protocol 冒烟全流程已跑通（已执行）
- 已实际运行：
  - `python run_six_client_seasonal_protocol.py all --smoke`
- 运行结果确认了三点：
  1. seasonal `build -> train -> eval` 三阶段链路能完整跑通；
  2. `58~63` 六客户端的本地 pretrain、Proposed/LMT/Meta-only 元训练、few-shot 适应和最终评估都已真正产生产物；
  3. `generate_multi_station_results.py` 在训练产物齐全后不再输出大量 `NaN`，说明前面的 seasonal 结果路径接线正确。
- 本次 smoke 的任务级主表已写入 `multi_station_performance.csv`，当前按 `nMAE_%` 统计：
  - `Proposed vs Local_Meta_Transfer = 2胜6负`
  - 胜的任务为：
    - `(59, high_wind)`
    - `(63, cold_wave)`
  - 其余任务均为 `LMT` 更优。
- 因此，这次 `1/1/1/1` smoke run 的意义仅是：
  - 验证 seasonal 协议实现闭环可运行；
  - 初步暴露在六客户端稀缺协议下，当前默认超参并未自动带来 `Proposed > LMT`。
- 不能把这组 smoke 分数当作正式结论，原因：
  - `PRETRAIN_EPOCHS=1`
  - `PROPOSED_META_EPOCHS=1`
  - `META_ONLY_META_EPOCHS=1`
  - `FEW_SHOT_EPOCHS=1`
  - 这只是链路健康检查，不是有效性能实验。
- 下一步若要继续做方法结论，应优先进入：
  - 设定正式 seasonal 预算
  - 跑至少一个可解释的 seed / budget 组合
  - 再比较 `Proposed` 与 `LMT` 的任务级结果

## 收敛检测（2026-04-04 起）
- 已把收敛检测正式接入 `DemoModelTraining.py`，并默认开启：
  - `ENABLE_CONVERGENCE_MONITOR=True`
  - `CONVERGENCE_REPORT_PATH=training_convergence_report.json`
- 设计原则固定为：
  - 只检测，不 early stop；
  - 不改变现有训练预算与优化路径；
  - 仅额外记录“是否收敛，以及若收敛是在何时收敛”。
- 当前收敛判定采用 `patience + min_delta` 的 plateau 检测，并要求至少达到 `min_epochs` 后才允许判定。
- 若训练后续又出现显著下降，则撤销先前 plateau 判断，避免假阳性。
- 当前默认超参：
  - `CONVERGENCE_MIN_DELTA = 1e-4`
  - `CONVERGENCE_MIN_EPOCHS = 5`
  - `CONVERGENCE_PATIENCE_PRETRAIN = 200`
  - `CONVERGENCE_PATIENCE_META = 100`
  - `CONVERGENCE_PATIENCE_FEW_SHOT = 5`
- 覆盖的训练过程固定为：
  - `federated_pretrain`
  - `local_pretrain`
  - `local_meta`
  - `few_shot`
- 每条收敛记录至少包含：
  - `stage_type`
  - `stage_id`
  - `total_epochs`
  - `converged`
  - `convergence_epoch`
  - `best_epoch`
  - `best_loss`
  - `final_loss`
- 冒烟验证已确认：
  - `python run_six_client_seasonal_protocol.py all --smoke` 会真实打印各阶段收敛摘要；
  - 并落盘 `training_convergence_report.json`；
  - 当前 smoke 报告共 `57` 条记录：
    - `federated_pretrain = 1`
    - `local_pretrain = 6`
    - `local_meta = 18`
    - `few_shot = 32`
  - 由于 smoke 预算仅 `1 epoch`，当前全部记录都为“未收敛”，这符合预期，不能误读为方法本身不收敛。

## 训练日志可见性（2026-04-04 起）
- six-client seasonal launcher 已固定使用非缓冲输出：
  - `run_six_client_seasonal_protocol.py` 会注入 `PYTHONUNBUFFERED=1`
  - 各阶段子进程统一按 `python -u ...` 启动
- 目的固定为：保证 `screen + tee` 场景下，终端能够实时看到阶段切换与 epoch 进度，而不是等缓冲区累积后才一次性刷出。
- `DemoModelTraining.py` 已新增统一日志辅助：
  - `progress_log(...)`：关键日志一律 `flush=True`
  - `should_log_epoch(...)`：统一控制长阶段打印节奏
- 当前默认日志策略：
  - `federated_pretrain` / `standalone_pretrain` / `local_pretrain`：前 `10` 个 epoch 每轮打印，之后每 `100` 轮打印一次，最后一轮必打
  - `local_meta`：前 `10` 个 epoch 每轮打印，之后每 `100` 轮打印一次，最后一轮必打；单行汇总 `support_mse / query_mse / anchor`
  - `few_shot`：预算较小，默认每轮都打印
- 该改动只影响日志可见性，不改变训练算法、训练预算、收敛检测或结果口径。
- CPU smoke 验证已确认：
  - `python run_six_client_seasonal_protocol.py all --smoke` 在当前无 CUDA 环境下，`build -> train -> eval` 各阶段日志均可实时看到；
  - 终端中已能直接看到 `local_pretrain`、`local_meta`、`few_shot` 的阶段头、`Epoch x/y`、以及收敛检测摘要。
- 正式实验口径仍不变：
  - 本地只用于 `CPU smoke / debug validation`
  - 完整预算仍由用户在远程 `RTX 4090` 上运行。

## Seasonal 数据完整性修复（2026-04-04）
- 问题现象：
  - `proposed_station60~63` 在元训练阶段出现 `support_mse / query_mse` 从首轮起接近或显示为 `0.000000`；
  - 进一步核查发现不仅 `63`，`58~63` 六个 seasonal `.mat` 都存在不同程度的功率尺度错误。
- 根因已确认在 `build_six_client_seasonal_protocol.py`：
  - `workbook_cache` 中的 `SheetRecord` 是可变对象；
  - `merge_workbooks_by_sheet(...)` 之前直接复用了这些对象；
  - `serialize_client_assets(...)` 内部对 `merged_train` / `merged_test` 执行 `normalize_power(...)` 时是原地修改；
  - 因此同一批记录会在 train/test 合并与多个 client 序列化过程中被重复归一化，导致后续 client 数值被越除越小。
- 修复策略：
  - 在 `merge_workbooks_by_sheet(...)` 中对每个 `SheetRecord` 进行深度脱钩复制（至少复制 `values` dict），保证每个 client 的 train/test sheet 只修改自己的副本；
  - 保留 `normalize_power(...)` 的单次归一化逻辑，不再共享可变记录对象。
- 回归测试已加入：
  - `tests/test_seasonal_data_integrity.py`
  - 覆盖两点：
    1. merge 后记录对象与缓存对象必须脱钩；
    2. 顺序构建所有 6 个 client 后，`63wf_seasonal_protocol.mat` 的 conventional target 尺度必须与从 `xlsx` 单次归一化得到的期望值一致。
- 重建后已逐 client 对账，以下字段均与从原始 `xlsx` 单次归一化、按协议切窗重算的期望值一致：
  - `p_conven_class`
  - `p_1h`
  - `p_test`
  - 各极端天气类的训练 / 测试功率序列
- 重建后的 6 个 client conventional 均值如下：
  - `58`: `0.37397753`
  - `59`: `0.24847406`
  - `60`: `0.11088715`
  - `61`: `0.22597806`
  - `62`: `0.23172681`
  - `63`: `0.16396683`
- CPU smoke 复核已确认：
  - `proposed_station60`: `support_mse=0.027336`, `query_mse=0.034695`
  - `proposed_station61`: `support_mse=0.089465`, `query_mse=0.150499`
  - `proposed_station62`: `support_mse=0.085218`, `query_mse=0.056790`
  - `proposed_station63`: `support_mse=0.028353`, `query_mse=0.033794`
  - 不再存在从首轮起恒为 `0.000000` 的异常。
- 运行口径提醒：
  - 在本地当前环境完成的仅是 `CPU smoke / debug validation`；
  - 因 seasonal 资产已重建，之前基于旧资产启动的 `4090 formal run` 日志与结果均作废，必须在远程 `RTX 4090` 上使用新资产重新启动正式实验。

## Seasonal 年份容量口径修复（2026-04-04）
- 在修复“重复归一化”之后，继续核对 seasonal `.mat` 与源 `xlsx` 的归一化 power，发现 `59/60/62/63` 的 `p_conven_class` 与 `p_test` 均值仍偏小。
- 进一步从源 `xlsx` 直接读取 `Power2`（以及能读到的 `Radio`）后确认：
  - `2223jilin_058_processed_4classes.xlsx`：容量口径为 `50`，且 `Radio` 列与 `Power2 / 50` 对应；
  - `2223jilin_059_processed_4classes.xlsx`：原始 `Power2` 最大值约 `48.16`，应按 `50` 归一化；
  - `2223jilin_060_processed_4classes.xlsx`：原始 `Power2` 最大值约 `98.22`，应按 `100` 归一化；
  - `24jilin_058/059/060_processed_4classes.xlsx`：容量口径分别为 `50 / 100 / 300`（与用户说明一致，且原始最大值约 `48.11 / 97.34 / 299.67`）。
- 因此 seasonal builder 不能再按 `source_station_id` 绑定单一容量，而必须按 `workbook filename` 区分年份容量：
  - `2223jilin_058_processed_4classes.xlsx -> 50`
  - `2223jilin_059_processed_4classes.xlsx -> 50`
  - `2223jilin_060_processed_4classes.xlsx -> 100`
  - `24jilin_058_processed_4classes.xlsx -> 50`
  - `24jilin_059_processed_4classes.xlsx -> 100`
  - `24jilin_060_processed_4classes.xlsx -> 300`
- 修复实现：
  - `build_six_client_seasonal_protocol.py` 改为 `CAPACITY_BY_WORKBOOK`；
  - `serialize_client_assets(...)` 先按工作簿克隆 sheet records，再按各自工作簿容量归一化，最后再 merge train/test；
  - 这样对于 `WT5/WT6` 这类跨 `2223 + 2024` 的测试窗口，也能正确处理不同年份的不同容量。
- 回归测试已扩展：
  - `tests/test_seasonal_data_integrity.py` 现在不仅检查 merge 脱钩与 `client63` conventional 尺度，还检查 `client63` 跨 `2223 + 2024` 测试窗口的 mixed-workbook 容量归一化是否正确。
- 修复后，以下 `seasonal .mat` 导出值已与源 `xlsx` 的归一化 power 对齐：
  - `58`: `p_conven_class mean = 0.37397753`, `p_test mean = 0.43009737`
  - `59`: `p_conven_class mean = 0.49694812`, `p_test mean = 0.50733650`
  - `60`: `p_conven_class mean = 0.33266146`, `p_test mean = 0.28806461`
  - `61`: `p_conven_class mean = 0.22597806`, `p_test mean = 0.19160303`
  - `62`: `p_conven_class mean = 0.46345361`, `p_test mean = 0.38140211`
  - `63`: `p_conven_class mean = 0.49190050`, `p_test mean = 0.33278873`
- CPU smoke 复核后，`proposed_station60~63` 的元训练损失也已恢复正常非零：
  - `60`: `support_mse=0.245595`, `query_mse=0.313472`
  - `61`: `support_mse=0.089461`, `query_mse=0.150485`
  - `62`: `support_mse=0.343715`, `query_mse=0.227622`
  - `63`: `support_mse=0.249552`, `query_mse=0.304768`
- 结论：当前 `seasonal_protocol_data/*.mat` 已和源 `xlsx` 中按正确年份容量归一化后的 power 对齐；任何在此修复前启动的 formal run 均需作废并重跑。
