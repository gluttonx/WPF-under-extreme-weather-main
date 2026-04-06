# Extreme-Stage FL Redesign

## Goal
在保持 `local_pretrain -> local_meta -> extreme adaptation` 主骨架不变的前提下，重设计 extreme 阶段的跨场站协作机制，使其真正服务于：

> 在仅有 `58 / 59 / 60` 三场站的背景下，如何利用另外两个场站同类极端天气的有效样本，提高目标场站的极端天气风机出力预测精度？

本次重设计不再把核心放在“站级场景相似性代理”上，而是把核心放在：

- 两个 source 场站都参与；
- 但每个 source 场站只贡献对当前 `target station + extreme class` 真正有效的 extreme windows；
- 在共享初始化下完成 extreme-stage local adaptation，再进行 target-conditioned aggregation。

## Why Redesign Is Needed
`pilot-medium: 2000 / 2000 / 20` 的 clean ablation 表明，当前 `Proposed-A` 没有优于 `Extreme-FedAvg`。根因不是单纯训练轮数不足，而是 extreme-stage 机制本身存在偏差：

1. 当前 prototype 在站内标准化后再求均值，导致各站点/类别的 prototype 接近零向量，`sim_{k->s}^c` 退化到接近常数。
2. 当前 `q_{k,c}` 并不来自真正的 held-out validation，而是来自 local adaptation 终值 loss，无法反映迁移可靠性。
3. 当前 extreme-stage 不是从共享初始化出发的联邦适配，而更像“从各自 `local_meta` 出发的一次性参数平均”。
4. 在仅有三个场站时，站级 similarity 本身就缺乏足够的排序空间，不适合作为主判据。

因此，新的 extreme-stage 设计应直接围绕“哪个 source windows 对 target 真正有帮助”来定义。

## Terminology
为避免将 extreme-stage 与元学习术语混淆，统一采用：

- `adapt`: local adaptation split
- `val`: local validation split

不再在 extreme-stage 使用 `support/query`。

这里的 `adapt / val` 含义是普通的本地拟合集与本地验证集，而不是 episodic meta-learning 中的任务划分。

## Core Shift
旧设计的核心是：

- same-class extreme updates
- target-conditioned reliability-aware aggregation
- station-level similarity weighting

新设计的核心改为：

- same-class extreme updates from a shared target-conditioned initialization
- window-level effective-sample screening
- two-source participation with controlled borrowing
- target-conditioned usefulness weighting

换句话说，新设计不再问：

> 哪个 source station 更像 target station？

而是问：

> 两个 source stations 的该类 extreme windows 中，哪些窗口对当前 target station 真正有帮助？

## Problem Setup
对目标场站 `s` 和极端类别 `c`，记：

- `D_{s,c}`: target station 在训练年中该类 extreme windows 的全集
- `P_{k,c}`: source station `k` 在训练年中该类 extreme windows 的候选池，`k != s`

其中 `k in {58, 59, 60} \ {s}`。

### Target Split
目标场站的该类样本按 extreme-stage 使用需求划分为：

- `D_{s,c}^{adapt}`
- `D_{s,c}^{val}`

当 `|D_{s,c}| >= 2` 时，优先按窗口划分 `adapt / val`。

当 `|D_{s,c}| = 1` 时，采用单窗口的 horizon-level fallback：

- 将单个 `12-step` future horizon 划分为 `H^{adapt}` 与 `H^{val}`
- local adaptation loss 在 `H^{adapt}` 上计算
- usefulness / aggregation scoring 在 `H^{val}` 上计算

这一步只用于 training-side model selection，不涉及测试年泄漏。

### Source Split
source 侧不再直接使用 `P_{k,c}` 全量数据，而是经过两级筛选后得到有效窗口集合：

- `Q_{k,c}`: source-quality gate 之后的候选池
- `E_{k->s,c}`: target-conditioned usefulness screening 之后的有效窗口集合

若 `|E_{k->s,c}| >= 2`，再划分为：

- `E_{k->s,c}^{adapt}`
- `E_{k->s,c}^{val}`

若 `|E_{k->s,c}| < 2`，则不再强制划分 `val`，并在权重中令 source-side reliability 项退化为 `1`。

## Shared Initialization
对目标场站 `s`、类别 `c`，新的 extreme-stage shared initialization 定义为：

`theta_{s,c}^{(0)} = theta_s^{meta}`

其中 `theta_s^{meta}` 为目标场站 `s` 的 `local_meta` 模型参数。

这是整个重设计的关键约束：

- target self update 从 `theta_{s,c}^{(0)}` 出发
- 两个 source updates 也从同一个 `theta_{s,c}^{(0)}` 出发

不再允许每个 source 从各自不同的 `local_meta` 起点独立更新后再做参数平均。

## Stage 1: Source-Quality Gate
对 source station `k` 的候选窗口 `w in P_{k,c}`，先计算 source-internal quality score。

令 `f_k^{meta}` 为 source station `k` 的 `local_meta` 模型，对单个窗口 `w = (x_w, y_w)` 定义：

`e_{k,c}^{self}(w) = L(f_k^{meta}(x_w), y_w)`

其中 `L` 为 window-level forecast loss，可取 MSE。

令 `tau_{k,c}^{self}` 为 `P_{k,c}` 上该误差分布的分位阈值，例如 `rho_self = 0.8` 分位。则 quality gate 后的窗口集合定义为：

`Q_{k,c} = { w in P_{k,c} | e_{k,c}^{self}(w) <= tau_{k,c}^{self} }`

这一步的作用仅为去掉明显异常、极不稳定、source 自身都难以解释的窗口。

它不是 target-conditioned selection，不参与最终迁移权重的主排序。

## Stage 2: Target-Conditioned Usefulness Screening
对每个 `w in Q_{k,c}`，从共享初始化 `theta_{s,c}^{(0)}` 出发，做一次轻量单窗口或小批量更新：

`theta_{s,c}^{(0)}(w) = U(theta_{s,c}^{(0)} ; w)`

然后直接在目标场站的 validation split 上评估该窗口对 target 的实际帮助：

`u_{k->s,c}(w) = L(theta_{s,c}^{(0)} ; D_{s,c}^{val}) - L(theta_{s,c}^{(0)}(w) ; D_{s,c}^{val})`

解释如下：

- 若 `u_{k->s,c}(w) > 0`，说明使用该窗口更新后，target validation loss 下降，该窗口对 target 有帮助；
- 若 `u_{k->s,c}(w) <= 0`，说明该窗口对 target 无帮助或有害。

于是，有效窗口集合定义为：

`E_{k->s,c} = TopB_{k->s,c}({ w in Q_{k,c} | u_{k->s,c}(w) > 0 })`

其中 `TopB_{k->s,c}` 表示按 `u_{k->s,c}(w)` 从大到小选取前 `B_{k->s,c}` 个窗口。

为避免 source 侧样本量淹没 target，借用预算设置为：

`B_{k->s,c} = min(|Q_{k,c}|, max(B_min, ceil(gamma * |D_{s,c}^{adapt}|)))`

默认设计意图是：

- `gamma` 控制 source borrowing budget 与 target shots 同量级；
- `B_min` 防止 target 样本极少时 source borrowing 完全为空。

因此，本设计不是“source station 二选一”，而是：

- 两个 source stations 都参与；
- 但各自只贡献 top useful windows。

## Stage 3: Local Extreme Updates
### Target Self Update
从共享初始化 `theta_{s,c}^{(0)}` 出发，在目标场站的 adaptation split 上更新：

`theta_{s->s,c}^{(1)} = U(theta_{s,c}^{(0)} ; D_{s,c}^{adapt})`

### Source Updates
对每个 source station `k != s`，从同一个共享初始化出发，在筛选后的有效窗口集合上更新：

`theta_{k->s,c}^{(1)} = U(theta_{s,c}^{(0)} ; E_{k->s,c}^{adapt})`

注意：

- 所有 source update 的起点相同；
- 差别只来自 `E_{k->s,c}`；
- 这才构成严格意义上的 target-conditioned same-class FL。

## Stage 4: Reliability and Transferability Scoring
对每个 source update `theta_{k->s,c}^{(1)}`，定义三类量：

### Effective Sample Sufficiency
`m_{k->s,c} = log(1 + |E_{k->s,c}^{adapt}|)`

### Source-Side Reliability
若 `|E_{k->s,c}^{val}| > 0`，定义：

`q_{k->s,c} = exp(-tau_q * L(theta_{k->s,c}^{(1)} ; E_{k->s,c}^{val}))`

否则：

`q_{k->s,c} = 1`

### Target-Side Transferability
直接在 target validation split 上评估该 source update 的迁移有效性：

`t_{k->s,c} = exp(-tau_t * L(theta_{k->s,c}^{(1)} ; D_{s,c}^{val}))`

这里 `t_{k->s,c}` 是新设计中的主判据。

在仅有三个场站时，相比 station-level similarity，target validation loss 更能直接反映“该 source update 对 target 是否有帮助”。

## Stage 5: Aggregation
### Extreme-FedAvg
为保证 fair comparison，`Extreme-FedAvg` 与 `Proposed-A` 共享同一套：

- shared initialization
- source-quality gate
- target-conditioned usefulness screening

二者唯一差别在 aggregation rule。

`Extreme-FedAvg` 的聚合方式定义为：

`theta_{s,c}^{agg-fedavg} = (1 / (1 + |K_s^c|)) * (theta_{s->s,c}^{(1)} + sum_{k in K_s^c} theta_{k->s,c}^{(1)})`

其中：

- `K_s^c = { k != s | |E_{k->s,c}^{adapt}| > 0 }`

### Proposed-A
`Proposed-A` 保留一个目标场站自更新的先验权重 `beta_self`，并将剩余权重分配给两个 source updates。

source 未归一化权重定义为：

`a_{k->s,c} = (m_{k->s,c})^{lambda} * (q_{k->s,c})^{mu} * (t_{k->s,c})^{nu}`

source 归一化权重为：

`tilde_alpha_{k->s,c} = a_{k->s,c} / sum_{j in K_s^c} a_{j->s,c}`

最终聚合为：

`theta_{s,c}^{agg-prop} = beta_self * theta_{s->s,c}^{(1)} + (1 - beta_self) * sum_{k in K_s^c} tilde_alpha_{k->s,c} * theta_{k->s,c}^{(1)}`

其中：

- `beta_self` 控制 target 自身个性化更新的保底占比；
- `lambda` 控制有效样本量；
- `mu` 控制 source-side reliability；
- `nu` 控制 target-side transferability。

与旧设计相比，新设计完全移除了 `sim_{k->s}^c` 主项。

## Stage 6: Target Refinement
无论是 `Extreme-FedAvg` 还是 `Proposed-A`，聚合后都增加一次短的 target-only refinement：

`theta_{s,c}^{final} = U_refine(theta_{s,c}^{agg} ; D_{s,c}^{adapt})`

理由是：

- 聚合态是“跨站共享模型”；
- 论文最终需要的是 target 个性化模型；
- 增加一次短 refinement 能提高个性化一致性。

## Revised Baselines
在共同 backbone `local_pretrain -> local_meta` 下，三条方法重新定义如下。

### LMT-new
`theta_{s,c}^{final-lmt} = U(theta_s^{meta} ; D_{s,c}^{adapt})`

即目标场站只利用自己的该类 extreme data，不引入任何跨站协作。

### Extreme-FedAvg
- 使用 shared initialization
- 使用 source-quality gate
- 使用 target-conditioned usefulness screening
- 使用 uniform aggregation
- 使用 target refinement

### Proposed-A
- 使用 shared initialization
- 使用 source-quality gate
- 使用 target-conditioned usefulness screening
- 使用 reliability-aware weighted aggregation
- 使用 target refinement

这样可以保证：

- `Extreme-FedAvg` 与 `Proposed-A` 的差别只在 aggregation rule；
- 有效窗口筛选成为 extreme-stage 协作的共同前处理，而不是只属于 Proposed 的独享技巧。

## Default Hyperparameters
建议第一版默认：

- `rho_self = 0.8`
- `B_min = 1`
- `gamma = 1.0`
- `tau_q = 1`
- `tau_t = 1`
- `lambda = 1`
- `mu = 1`
- `nu = 2`
- `beta_self = 0.5`

其中：

- `nu = 2` 体现 target transferability 的主导地位；
- `beta_self = 0.5` 体现 target self update 的保底重要性；
- `gamma = 1.0` 保证 source borrowing budget 与 target shot 数同量级。

## Logging Requirements
新设计必须显式记录以下日志，否则无法诊断方法是否真正工作：

1. `target_station`, `extreme_class`, `|D_{s,c}^{adapt}|`, `|D_{s,c}^{val}|`
2. 每个 source 的 `|P_{k,c}|`, `|Q_{k,c}|`, `|E_{k->s,c}|`
3. 每个 source 的 `m_{k->s,c}`, `q_{k->s,c}`, `t_{k->s,c}`
4. `Extreme-FedAvg` 的 uniform weights
5. `Proposed-A` 的 `beta_self`, `tilde_alpha_{k->s,c}`
6. target refinement 前后的 validation loss

## Validation Scope
优先验证以下问题：

1. `Proposed-A` 是否稳定优于 `Extreme-FedAvg`
2. `Extreme-FedAvg` 是否优于 `LMT-new`
3. `effective-window screening` 是否确实减少了无效 source borrowing
4. `beta_self` 是否防止跨站更新压过 target self personalization

评估仍按：

- `(station, extreme_class, model)` task-level 明细
- `Station x Model` 的 `TABLE IV` 风格宽表

指标保持：

- `nMAE_%`
- `nRMSE_%`
- `WD_%`
- `R_p<0.05_%`

## Non-Goals
本次重设计不引入：

- secure aggregation
- differential privacy
- 图结构 GNN 式场站关系建模
- 大规模多轮联邦通信协议

当前目标是先让 `3` 场站的 same-class extreme cooperation 在方法上成立，并在 pilot 阶段打出稳定正信号。
