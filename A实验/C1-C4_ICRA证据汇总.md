# C1–C4 ICRA 证据汇总

> 范围仅限新版 realistic Kino-Fail 与 A0–A7，不包含已放弃的外部数据集。
>
> 截至 2026-07-24 的机器判决：
> `all_four_core_claims_supported_c2_strict_superiority_not_confirmed`。
> C1、C3、C4 通过各自冻结验收；C2 的核心“冲突可可靠消解”得到强支持，
> 但预注册的“ours 严格击败每个 fusion baseline”未通过，因此不得写成方法全面领先。

## 1. 投稿级结论

| Claim | 最新确认数据 | 判定 | 主文可写结论 |
|---|---|---|---|
| C1：视觉与本体存在互补的单模态盲区 | 21 matched cases、126 samples、3 个未见场景/3 域 | **强证实，PASS** | 在冻结的 O2/O4 matched T2 构造中，本体严格不可分，而前置 RTX RGB 可高置信区分 |
| C2：双向模态冲突可被可靠消解 | 6-scene LOSO 开发 + 3 个全新场景、180 samples 的一次性确认 | **核心强证实；严格 superiority 未证实** | structured post-interaction routing 在双向 T2/T3 电池上达到 0.994；不能声称严格优于 RBF/late fusion |
| C3：归因条件动作会因果且不对称地改变物理后果 | 630 个 Isaac-Go2 intervention episodes；另接 75 对 realistic direct outcomes | **强证实，PASS** | 动作选择是物理后果的因果决定因素，且代价结构依异常而异 |
| C4：支持域内的 consequence-aware selective recovery 优于 always-safe | 75 对、3 个未见场景/3 域、matched-prefix direct comparison | **强证实，PASS** | 在冻结支持域和 20% coverage 下，selective policy 降低代价并提高成功率 |

机器证据账本：
`outputs/eval/c1_c4_evidence_ledger_v2.json`。它同时保留
`all_frozen_acceptance_passed=false` 与 `all_core_claims_supported=true`，
避免把 C2 的核心结论和额外 superiority 门混为一谈。

## 2. C1：单模态互补盲区

### 冻结设计

- 21 个 fresh matched cases，126 个样本，覆盖 life/production/wild 三域和三个
  confirmation scene clusters。
- 每个 cause-case 有 primary、swap-01、swap-02 三种 appearance view。
- O2 compliance 与 O4 adhesion 共享完全相同的 21×19 proprio bytes；
  只允许 Go2 前置 RTX RGB 中的物理可见线索改变。
- 推断以 scene cluster 为单位，主指标为 within-scene ROC-AUC 的宏平均及
  scene-cluster bootstrap 置信区间。

### 结果

| 指标 | 结果 |
|---|---:|
| RGB macro within-scene ROC-AUC | **0.99395** |
| scene-cluster bootstrap 95% CI | **[0.98639, 1.00000]** |
| 三场景 ROC-AUC | 0.98639 / 1.00000 / 0.99546 |
| RGB balanced accuracy | 0.91270 |
| texture-swap hard consistency | 0.83333 |
| proprio ROC-AUC | **0.50000** |
| proprio byte-identity rate | **1.00000** |
| 最大 paired proprio probability delta | **0.00000** |

这不是“视觉模型刷满”：balanced accuracy 仍为 0.913，texture consistency 仍为
0.833，保留了可继续改进的真实难点。强结论来自 paired construct validity、
scene-level uncertainty 和双通道方向相反的结果，而不是单一高准确率。

权威报告：
`outputs/eval/c1_causal_confirmation_v3/formal/report.json`。

## 3. C2：双向冲突消解

### v5 方法和无泄漏时序合同

最终方法是 `structured_post_interaction_router_v5`：

1. 用第一次 base-footprint 到 operator region 距离不超过 0.35 m 定义物理接触；
2. 只取接触后 0.30 s 处结束的 21 个 native-rate samples；
3. 将 19 个原始通道化为角速度、线加速度、里程计速度、slip、height、tilt、
   effort、support 八个旋转不变信号，再汇总为 80 维；
4. proprio proposal 若属于 T3 类则接受，否则调用 T2 CLIP+generic-HOG specialist。

窗口不读取 fall/outcome time、truth、operator ID、scene/domain ID 或 test cost。
因此它是部署时可计算的 post-interaction attribution，不是 outcome-aligned oracle。

### 开发与确认隔离

- 开发：6 个已消费场景、360 samples、60 matched cases，逐场景 LOSO。
- 确认：3 个全新场景
  `forest_trail_metric_g02`、`indoor_bedroom_190`、`indoor_office_156`，
  与开发场景严格不相交。
- 正式集：180 samples、30 matched cases、4 类、T2/T3 各 15 case、
  每个 cause 三种 appearance intervention。
- Amendment 1 只修复 swap-02 视觉变化不足的 collection QA；没有降低
  L1=0.015 的冻结门槛，没有改变模型、时序合同或 acceptance threshold，
  且在任何正式预测产生前从头重采全部 30 个 T3 pair。
- 联合 QA 核验 15 个 T2 case、60 个 T3 episode、30 个 counterfactual pair、
  30 个时序窗和 1,275 个 artifact hashes。最弱 swap L1=0.02849，
  最大时序端点偏差=0.020 s，均通过冻结门。

### 六场景 LOSO 开发结果

| 方法 | Balanced | T2 | T3 | Worst scene |
|---|---:|---:|---:|---:|
| vision-only | 0.6417 | 0.9944 | 0.2889 | 0.5000 |
| proprio-only | 0.7500 | 0.5000 | 1.0000 | 0.7500 |
| early fusion linear | 0.8250 | 0.9944 | 0.6556 | 0.5000 |
| early fusion RBF | 0.7722 | 0.9944 | 0.5500 | 0.5000 |
| late fusion | 0.9583 | 0.9944 | 0.9222 | 0.8667 |
| **ours v5** | **0.9972** | **0.9944** | **1.0000** | **0.9833** |

### 一次性正式确认结果

| 方法 | Balanced | T2 | T3 | Worst scene | Texture consistency |
|---|---:|---:|---:|---:|---:|
| vision-only | 0.7444 | 0.9889 | 0.5000 | 0.7333 | 0.7500 |
| proprio-only | 0.7500 | 0.5000 | 1.0000 | 0.7500 | 1.0000 |
| early fusion linear | 0.9889 | 0.9778 | 1.0000 | 0.9667 | 0.9667 |
| early fusion RBF | **0.9944** | 0.9889 | 1.0000 | 0.9833 | 0.9833 |
| late fusion | **0.9944** | 0.9889 | 1.0000 | 0.9833 | 0.9833 |
| **ours v5** | **0.9944** | **0.9889** | **1.0000** | **0.9833** | **0.9833** |

九个冻结门中八个通过。唯一失败项是
`beats_every_baseline_point=false`：ours、RBF early fusion 和 late fusion 的
180 条预测完全相同，只共同错了 bedroom 中一个 O4-primary sample。
相对 best baseline 的 30-case paired 结果为 0 wins / 30 ties / 0 losses，
Δ=0，95% CI=[0,0]。

因此：

- **强支持：** 双向 T2/T3 冲突在未见 realistic scenes 上可被可靠消解；
  单模态在错误方向恰为 0.5，排除了“任一单模态已经足够”的解释。
- **不支持：** v5 在该确认集严格优于所有 fusion baseline。确认集对三种融合方法
  已接近饱和，无法辨识细小架构差异。
- **论文定位：** C2 是 learnability/reliability claim，不是 universal SOTA claim。
  主表必须保留 tie；不得改阈值、删 baseline 或在同一正式集上重调。
- **benchmark headroom：** 该四类确认 slice 用于验证因果结构，不代表完整
  11-operator Kino-Fail leaderboard。完整 A0–A7、texture consistency、
  C3 boundary cells 和 C4 coverage 仍保留明显刷分空间。

权威产物：

- `outputs/eval/c2_bidirectional_v5/formal/report.json`
- `outputs/eval/c2_bidirectional_v5/confirmation_audit.json`
- `configs/eval/kinofail_realistic_c2_bidirectional_formal_v5_amendment1.json`

## 4. C3：动作—后果机制桥

### 受控因果干预

- 9 个 scenario clusters × 7 个强制动作标签 × 10 seeds = 630 个
  Isaac-Go2 physics episodes。
- 所有 9/9 场景的 terminal cost 随动作改变，8/9 的 success 随动作改变。
- 5 个 clean scenarios 有唯一 canonical winner，共 50 个 paired seed comparisons；
  two-sided exact sign `p=1.776×10⁻¹⁵`。
- T2 adhesion/compliant-terrain 的冻结 cost asymmetry ratio=4，
  posterior crossing `p*=0.25`。

### 边界

只有 5/9 场景具有唯一 winner；O7、O10 等 4 个 non-unique/boundary cells 原样保留。
C3 支持“动作选择会因果且不对称地改变后果”，不支持“每个正确语义标签都唯一决定
最佳动作”。这组负/弱单元正是 benchmark 可持续刷分的承重 headroom。

权威报告：
`outputs/eval/c3_mechanism_bridge_v1/report.json`。

## 5. C4：支持域内选择性恢复

- 75/75 matched cases，3 个 untouched test scenes，life/production/wild 三域。
- 每个 case 比较冻结 selective policy 与冻结 always-safe policy；
  pre-decision prefix、branch order、fresh-process isolation 和部署输入泄漏门全部通过。
- 只放行 actor 已支持的 15 个 O4 adhesion case；O2/O5/O8/O9 hard/boundary
  strata 全部回退，不删困难样本。

| 终点 | 结果 |
|---|---:|
| selective − always-safe terminal cost | **−5.9342** |
| scene-cluster hierarchical bootstrap 95% CI | **[−8.7041, −3.5568]** |
| selective − always-safe success | **+0.2000** |
| 95% CI | **[+0.1197, +0.2933]** |
| release coverage | **0.20** |
| released attribution precision | **1.00 (15/15)** |

C4 强证实的是冻结支持域中的 selective release，而不是通用恢复：80% coverage
仍由 always-safe 承担，O2/O5/O9 恢复、open-set 语义和实体 Go2 都不在结论内。
低 coverage 是安全边界，也是后续方法可继续提升的明确空间。

权威报告：
`outputs/eval/realistic_a0_a7_v6/a5_c4_direct.json`。

## 6. 主文与附录报告规则

主文只展示最新 v5 C2 结果，不展示“旧 specialist → 多轮冻结失败”的研发流程。
为保证科研审计，C2 v1–v4 的失败报告继续保存在 outputs 中，但只在 appendix/reproducibility
说明中简要登记，不能删除或倒写。

建议主文四个承重结果：

1. C1 paired RGB/proprio contrast 与 scene bootstrap CI；
2. C2 v5 正式表，突出单模态互补、ours 的高可靠性以及与强 fusion baseline 的 tie；
3. C3 9×7 consequence matrix，保留 4 个 non-unique boundary scenarios；
4. C4 paired Δcost/Δsuccess、coverage-risk 和 released precision。

禁止措辞：

- “ours beats all baselines”；
- “all eleven operators are solved”；
- “every attribution uniquely determines the optimal action”；
- “universal recovery”或“real-robot validated”。

推荐总贡献表述：

> Kino-Fail 提供 realistic、因果配对、物理—视觉可解耦且严格冻结的异常归因评测；
> C1 证明模态互补盲区，C2 证明双向冲突在未见场景上可可靠消解，C3 证明归因条件动作
> 会因果改变物理后果，C4 证明支持域内的 consequence-aware selective recovery
> 能以受控 coverage 改善安全结果。强 fusion baseline 的正式 tie、C3 boundary cells
> 和 C4 的 20% coverage 被保留为 benchmark headroom，而不是被隐藏。

