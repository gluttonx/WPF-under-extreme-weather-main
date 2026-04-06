# 2026-04-05 Session Summary

## 1. 本次会话的最终结论
- `2026-04-04` 的 six-client seasonal protocol 已经完整实现、修过数据完整性与年份容量问题，并在 `4090 formal run` 上跑完。
- 但 formal 结果不支持“当前 `Proposed` 在各 client 上稳定优于 `LMT`”这一主张。
- 这次会话的关键收敛不是继续调 seasonal 协议，而是**重写研究问题**：
  - 从“normal-only federated pretrain 是否必要”
  - 转向
  - **“在极端天气小样本预测中，跨 client 的共享应该放在哪个阶段、以什么目标进行，才真正有助于 extreme few-shot？”**

## 2. 对 seasonal 六客户端方案的最终判断
- six-client seasonal 路径可以保留为已执行的 archive 路径，但**不再作为主方法路线**。
- 主要原因不是单一 bug，而是方法层面错位：
  - 联邦共享发生在 `normal/conventional` pretrain；
  - 最终目标却是 `extreme few-shot`；
  - 共享目标与下游目标不够对齐。
- 这次会话已明确：
  - seasonal formal result 不能支撑“Proposed 对 LMT 全面稳压”的主结论；
  - 继续只在 seasonal 路径上补 budget / sampler / 窗口，不是当前最优方向。

## 3. 关于旧三场站全年版本的关键事实
- 重新核查后已确认：
  - 旧代码中，全年连续序列的 generic `test` 路径确实是 `2022 train -> 2023 test`；
  - 但**按 extreme class 的 few-shot 路径不是**，因为旧 `.mat` 中 `p_extre_class*` 是把 `xlsx` 的整张 extreme sheet 直接写进去，并没有按 `2022/2023` 分开。
- 因而如果要做新的 extreme-stage FL，不能直接复用旧 `.mat + 旧 few-shot`，必须从 `xlsx` 重建：
  - `2022 extreme support`
  - `2023 extreme test`

## 4. 全年三场站 extreme-task scarcity 的核查结论
- 已按 `xlsx` 重新统计 `2022` 和 `2023` 的全年极端天气小时数与 `12` 小时窗口数。
- 结论不是“所有类都很多”，而是：
  - `high_temp`：仍然稀缺
  - `cold_wave`：仍然稀缺
  - `high_wind`：中等稀缺
  - `frost`：相对不再那么稀缺，单站本地 `LMT` 可能已经不弱
- 因此新的 scarcity 定义被固定为：
  - **target 在某一类 extreme task 上的数据有限**
  - 而不是“target 全年 normal 数据有限”

## 5. 关于共享月份窗口的结论
- 曾尝试在三站上寻找“相同月份、单个连续时间块”的共享协议。
- 系统扫过候选窗口后，若坚持：
  - 三站相同月份
  - 单个连续窗口
  - 四类 extreme 都保留
  - `2022 train / 2023 test`
  则最平衡的窗口是：
  - `2022-02-01 ~ 2022-12-31`
  - `2023-02-01 ~ 2023-12-31`
- 但用户最终决定：**这个问题先放下**，后面若新方法效果仍不佳，再回来优化月份窗口。
- 因此当前主路线固定为：
  - **先回到三场站全年数据**
  - 再做新的 extreme-stage FL

## 6. 新主方法已冻结
- 这次会话已经把新主方法固化进文档和 `AGENTS.md`。
- 新方法建立在 `d5030ab (3.24.23: RAPP-original data)` 骨架上，不是 seasonal 路径的补丁。

### 主创新点 1
- **Extreme-Type-Conditioned Reliability-Aware Federated Adaptation（Extreme Stage）**
- 共享从 `normal-only federated pretrain` 转移到 **same-class extreme adaptation**。
- 每个 extreme class 单独形成联邦组：
  - `G_c = {k | N_k^c > 0}`
- 对目标 client `s`、类别 `c`，目标条件化聚合为：
  - `theta_{s,c}^{(t+1)} = sum_{k in G_c} alpha_{k->s}^{c,(t)} theta_{k,c}^{(t+1)}`
- 不再使用 naive `FedAvg`。

### 主创新点 2
- **Scenario Prototype Construction and Reliability Weight Calibration**
- 聚合权重由三部分共同决定：
  - 样本量可靠性 `m_{k,c}`
  - 本地 query 质量 `q_{k,c}`
  - 目标场景相似度 `sim_{k->s}^c`
- 默认组合形式固定为：
  - `a_{k->s}^{c,(t)} = (m_{k,c})^lambda (q_{k,c}^{(t)})^mu (sim_{k->s}^c)^nu`
  - `alpha_{k->s}^{c,(t)} = a_{k->s}^{c,(t)} / sum_j a_{j->s}^{c,(t)}`
- 第一版默认：
  - `tau = 1`
  - `lambda = 1`
  - `mu = 1`
  - `nu = 2`

## 7. 隐私口径已固定
- 当前方法如果只做 A，不应声称“强加密”或密码学强保护。
- 允许、且应当使用的准确表述是：
  - 原始 extreme-weather 数据不离开本地
  - 客户端之间不交换原始样本
  - 跨 client 共享通过模型更新、样本数统计和场景原型/质量分数实现
- 这足以支持 federated/data-local privacy 口径，但不等同于 secure aggregation / DP / HE。

## 8. 已落盘的关键文档
- 新方法设计：
  - `docs/plans/2026-04-05-extreme-stage-weighted-fl-design.md`
- 新方法实现计划：
  - `docs/plans/2026-04-05-extreme-stage-weighted-fl-implementation.md`
- 本次长会话总结：
  - `docs/plans/2026-04-05-session-summary.md`

## 9. 新会话应优先读取的材料
1. `AGENTS.md` 中 `2026-04-05` 的新方法记录
2. `docs/plans/2026-04-05-session-summary.md`
3. `docs/plans/2026-04-05-extreme-stage-weighted-fl-design.md`
4. `docs/plans/2026-04-05-extreme-stage-weighted-fl-implementation.md`

## 10. 下一步（新会话中再做）
- 按实现计划开始真正编码，第一任务是：
  - 从原始 `xlsx` 构建三场站全年 `2022 support / 2023 test` 的 extreme 协议资产
- 严格遵守 runtime contract：
  - 本地仅做 `CPU smoke / debug validation`
  - `4090 formal run` 仍由用户在远程环境执行
