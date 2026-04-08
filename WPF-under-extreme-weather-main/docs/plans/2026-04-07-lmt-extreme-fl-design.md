# LMT-Based Extreme FL Design

## Goal
在 `9699391` 的 `few-shot = MSE-only` 口径上，保留已经恢复稳定的 `LMT` 基线，并仅在 extreme few-shot 阶段新增两条方法：

- `Extreme-FedAvg`
- `Proposed-A`

结果主表保持 `fb2c67a` 风格，只输出三条方法：

- `LMT`
- `Extreme-FedAvg`
- `Proposed-A`

不再单独输出 `Meta_Learning` 与 `Pre_Training` 行。

## Fixed Backbone
三条方法共享同一 backbone：

1. local pre-training
2. local meta-training
3. class-specific extreme adaptation

记目标场站为 `s`，极端天气类别为 `c`，则共享初始化定义为：

`theta_{s,c}^{(0)} = theta_s^{meta}`

其中 `theta_s^{meta}` 为场站 `s` 的 `local_meta` checkpoint。

## LMT
`LMT` 保持 target-only 极端适配：

`theta_{s,c}^{LMT} = U(theta_{s,c}^{(0)} ; D_{s,c}^{adapt})`

其中 `D_{s,c}^{adapt}` 为目标场站 `s` 在极端类别 `c` 上的本地适配集。

## Extreme Data Split
对目标场站和 source 场站的 extreme windows 都采用 `adapt / val` 命名，不再使用 `support / query`。

若某类窗口数 `N >= 2`，则按窗口切分：

- `D^{adapt}`：前 `N_val` 之外的窗口
- `D^{val}`：保留的验证窗口

若 `N = 1`，则对单个 `12-step` horizon 做时间轴切分：

- `H^{adapt}`：前半段
- `H^{val}`：后半段

## Effective Source Windows
对 source 场站 `k != s` 的同类 extreme windows，先做两级筛选。

### Stage 1: source-quality gate
用 source 场站自己的 `local_meta` 模型计算每个窗口的本地误差：

`e_{k,c}^{self}(w) = L(f_k^{meta}(x_w), y_w)`

按误差从小到大保留前 `K_q` 个窗口，得到：

`Q_{k,c} subset P_{k,c}`

### Stage 2: target-conditioned usefulness
对每个 `w in Q_{k,c}`，从共享初始化 `theta_{s,c}^{(0)}` 做一次轻量单窗口更新，并在目标场站 `D_{s,c}^{val}` 上评估迁移效果：

`u_{k->s,c}(w) = L(theta_{s,c}^{(0)} ; D_{s,c}^{val}) - L(U(theta_{s,c}^{(0)} ; w) ; D_{s,c}^{val})`

只保留 `u_{k->s,c}(w) > 0` 的窗口，并按 usefulness 取前 `B_{k->s,c}` 个，得到有效窗口集：

`E_{k->s,c}`

这里的 borrowing budget 由目标场站 shots 控制，而不是把两个 source 场站的全年同类窗口全部吞入。

## Extreme-FedAvg
对目标场站 `s`、类别 `c`：

1. target self update：
   `theta_{s->s,c}^{(1)} = U(theta_{s,c}^{(0)} ; D_{s,c}^{adapt})`
2. source updates：
   `theta_{k->s,c}^{(1)} = U(theta_{s,c}^{(0)} ; E_{k->s,c}^{adapt})`
3. uniform aggregation：

`theta_{s,c}^{fedavg} = (theta_{s->s,c}^{(1)} + sum_k theta_{k->s,c}^{(1)}) / (1 + |K_s^c|)`

4. target refinement：

`theta_{s,c}^{final-fedavg} = U_refine(theta_{s,c}^{fedavg} ; D_{s,c}^{adapt})`

## Proposed-A
`Proposed-A` 与 `Extreme-FedAvg` 共享：

- shared init
- source-quality gate
- target-conditioned usefulness screening

唯一差异在 aggregation。

对每个 source update 定义：

- sample sufficiency：
  `m_{k->s,c} = log(1 + |E_{k->s,c}^{adapt}|)`
- source-side reliability：
  `q_{k->s,c} = exp(-tau_q * L(theta_{k->s,c}^{(1)} ; E_{k->s,c}^{val}))`
- target-side transferability：
  `t_{k->s,c} = exp(-tau_t * L(theta_{k->s,c}^{(1)} ; D_{s,c}^{val}))`

未归一化权重：

`a_{k->s,c} = (m_{k->s,c})^lambda * (q_{k->s,c})^mu * (t_{k->s,c})^nu`

归一化 source 权重：

`tilde_alpha_{k->s,c} = a_{k->s,c} / sum_j a_{j->s,c}`

最终聚合：

`theta_{s,c}^{prop} = beta_self * theta_{s->s,c}^{(1)} + (1 - beta_self) * sum_k tilde_alpha_{k->s,c} theta_{k->s,c}^{(1)}`

随后同样追加 target refinement：

`theta_{s,c}^{final-prop} = U_refine(theta_{s,c}^{prop} ; D_{s,c}^{adapt})`

## Evaluation Contract
主表继续使用 `fb2c67a` 风格：

- 每个场站 `3` 行：`LMT / Extreme-FedAvg / Proposed-A`
- `Overall_Average` 再给出 `3` 行
- 列顺序保持四类天气横向展开后接 `Training_duration_s` 和 `R_p<0.05_%`

当前阶段不再单独输出 `Meta_Learning` 与 `Pre_Training` 行，因为它们已经不再是论文主比较对象。
