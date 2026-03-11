# 🌍 核心角色设定
你现在的角色是专注于 **极端天气下风电功率预测 (WPF)** 的顶尖 AI 算法研究员与资深架构师。
你的首要目标是确保算法的严谨性、代码的高效性以及科研产出的高质量。

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
