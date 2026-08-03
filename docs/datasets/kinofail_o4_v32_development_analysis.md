# Kino-Fail realistic O4 v32 开发结果与评价合同诊断

更新时间：2026-07-23

## 结论

v32 在 v33 预先冻结的五个新 development scene 上一次性执行，覆盖 Bedroom、DiningRoom、
Kitchen、LivingRoom、Office 五个 EmbodiedGen room family。v32 相对 v27 只改变三个运行参数：
首次 attachment 后 1 step 决策、最短/最长决策等待均为 1 step、Backstep 上限从 120 扩到完整
300-step horizon；其余物理、policy、控制、速度、force cap、peel 与 clearance 阈值保持不变。

- 证据完整性：`5/5`，`integrity_passed=true`。
- 预注册严格 outcome：`4/5`，`outcome_passed=false`。
- 主要物理对比：`5/5` Continue 在分支后跌倒，`5/5` Backstep 不跌倒、完成 clearance、后退
  距离超过 0.15 m，并发生真实 post-decision peel。
- 封存状态：`sealed=true`；五个场景均不得重跑、调参或转作 held-out confirmation。
- 该结果不属于 O4 held-out confirmation，也不属于 A0–A7；realistic readiness 仍为 `0/8`。

封存入口及 SHA-256：

- config：`configs/data/kinofail_o4_action_consequence_v32_development_freeze_v1.json`，
  `1637f3856591881def98bef491551de442fde08ca35c1d4d78e91348b411aece`
- preflight：`outputs/kinofail_realistic/operator_development/o4_action_v32_preflight/preflight_audit.json`，
  `08ab00f93eb1950f1868378fb02c7e2edf0470f8cf1d3f00da83758ce3beacbc`
- runner：`outputs/kinofail_realistic/operator_development/o4_action_v32_early_clearance/runner_audit.json`，
  `32f35cb3934f07cba8588ae48f86827eb1a980caeb56b57b32722c799a6fb288`
- postrun：`outputs/kinofail_realistic/operator_development/o4_action_v32_early_clearance/postrun_audit.json`，
  `8d7f18ce12260eee212a928576b755f314b9f52290780d0ba7f0714720d4732e`

## 逐场景结果

| 场景 | 首次 attachment / decision | Continue 首次跌倒 | Backstep 后退 | Backstep 最大横向偏差 | 严格结果 |
|---|---:|---:|---:|---:|---:|
| DiningRoom113 | 50 / 51 | 172 | 0.224 m | 0.023 m | PASS |
| Office114 | 50 / 51 | 156 | 0.224 m | 0.023 m | PASS |
| LivingRoom115 | 79 / 80 | 167 | 0.275 m | 0.075 m | PASS |
| Bedroom118 | 51 / 52 | 178 | 0.208 m | 0.029 m | FAIL |
| Kitchen122 | 55 / 56 | 122 | 0.262 m | 0.020 m | PASS |

所有场景均由 `maximum_dwell` 在首次 attachment 后恰好 1 step 产生 decision，Continue 与
Backstep 的 decision 前 telemetry byte-identical。LivingRoom115 与 Kitchen122 的旧式通用
图像门只在已跌倒 Continue outcome 报低纹理；预注册的 state-aware visual QA 正确保留了这两例，
同时 initial、decision 与安全 Backstep outcome 仍必须通过严格图像门。

## Bedroom118 为什么按旧规则失败

Bedroom118 的 Backstep 完整满足目标：未跌倒、后退 0.2078 m、完成 active-feet clearance、
产生一次 peel，最大横向偏差只有 0.0293 m。失败来自 Continue 的全时域最大横向偏差
0.4026 m，超过冻结的 0.30 m 门；collector 与独立 postrun auditor 因而均按原合同判负。

逐帧分解表明该越界不是反事实混杂：

- matched prefix 截至 decision step 52 的最大偏差为 0.0166 m；
- 使用与 v27/v32 已冻结姿态阈值相同、且不读取路线偏差的失稳起点定义，Continue 首次同时满足
  `tilt >= 0.08 rad` 与相对 attachment 前参考高度下降 `>= 0.02 m` 是 step 132；此时偏差
  0.0295 m；
- step 171 才首次超过 0.30 m，此时高度已降到 0.206 m、倾角已达 0.588 rad；
- step 178 判定跌倒，偏差 0.4026 m、倾角 0.838 rad。

因此全时域 `both_lanes_within_route_budget` 把 Continue 的倾覆/侧滑后果误当成了决策前路线混杂。
不过该诊断是在观察 v32 后完成，不能据此回写 v32 结果；v32 必须永久保持严格 `4/5`。

## 前瞻性 v34 合同

下一批不得再使用 v32 的五个场景，也不得处理 v33 已冻结为 untouched 的 candidate 123–136。
从 candidate 137 以后先按 nominal-only gate 获取全新 operator-blind cohort，再在任何 O4 outcome
可见之前冻结以下合同：

1. **动作与物理保持 v32 不变。** 仍在首次 attachment 后 1 step 分叉，Backstep 最多贯穿
   300-step horizon；不得改 policy、速度、route controller、adhesion、peel、clearance 或阈值。
2. **因果前缀受限。** Continue 与 Backstep 在 decision 前必须 byte-identical，且 matched prefix
   全部位于 0.30 m route budget 内。
3. **恢复分支全程受限。** Backstep 在完整 horizon 内必须位于 0.30 m route budget 内，并满足
   不跌倒、真实 peel、后退距离与 clearance。
4. **异常分支采用 phase-aware containment。** Continue 只要求从初始到首次物理失稳起点仍在
   route budget 内；失稳起点沿用冻结的姿态合取条件，不使用 route deviation 计算。起点后的
   横向滑移作为异常后果完整报告，不再作为因果混杂否决项。
5. **原始旧指标不删除。** 全时域 Continue 最大偏差、首次越界 step、失稳起点与跌倒 step 全部
   保留，避免通过隐藏困难后果取得通过。
6. **状态感知图像门保持 v32 不变。** 只允许已跌倒 Continue outcome 使用传感器有效性门；
   initial、decision 和 Backstep outcome 仍过严格视觉门。
7. 新 cohort 每场只执行一次，全部结果封存；只有前瞻性 v34 全 cohort 通过，才能把 O4 架构
   标记为新 realistic scene 上确认。该 operator gate 仍不能把 A0–A7 readiness 从 `0/8` 改为完成。

v34 的目标不是放宽成功定义，而是让 nuisance-control 门只约束因果识别所需的阶段，同时继续
以更严格的全时域标准约束安全恢复分支。完成 O4 后仍需扩展 O5–O11 realistic realization、绑定
canonical corpus、重新训练/推理并独立复现 A0–A7。
