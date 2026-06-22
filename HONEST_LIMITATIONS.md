# KinoVLA 诚实局限审计报告 (Honest-Limitations Audit)

> **生成日期：** 2026-06-22
> **方法：** 10 个 agent 并行扫描全仓库 —— `kino_vla/` 全部子系统 + `tests/` / `scripts/` / `configs/` + `CLAUDE.md` §6 的 #1–#39 + `outputs/` 全部结果文档。共 211 条原始发现，去重并过滤掉"已彻底修复、不再是局限"的项后，得到 **182 条当前仍为真的诚实局限**。
> **范围：** 本文件只记录**当前仍为真**的诚实局限（包括那些标注为 RESOLVED 但其"解决方案本身就是一个永久范围限制"的项），不记录已彻底修复的环境/驱动类问题。

---

## 执行摘要 (Executive Summary)

KinoVLA 的**离线 / 同分布 (in-distribution) 归因结果是强且真实的**：在真实 Go2 + Isaac 上采集的数据上，Kino-SFT 在 Suite-Sem 歧义对上的归因准确率达 0.974，远超非语义多数类基线 (0.237)，且解析率 100%、可从 config+seed 复现。

但项目的几个**核心论点仍是诚实的局限，未被达成**：

1. **潜在路线 (latent/B5) 在归因准确率上与文本路线 (B4) 打平** (均 1.000 / 0.974)，其优势被诚实地重述为"必要性 + token 效率 (43×) + θ-grounding"，而非准确率；
2. **Embodied DPO 未稳健超越 SFT** (exit-3 未达成，里程碑框留空)——满数据 SFT 已到归因天花板，低数据下贪婪增益在采样下反转；
3. **闭环结果半边 (Recoverability Dichotomy 的 outcome 侧) 不成立**——在真实 Go2 闭环里因果盲的 FSM (B2) 在每个 outcome 指标上追平或击败完整 VLA (B5)，决定性的 O5 过载格还被反转；
4. **bang-bang→平滑巡航的 proprio shift** 使部署窗口对模型 OOD，主动探测探针只是 tradeoff (修了 O1/O2、破坏了 O5)；
5. **"视觉不可替代"的 O2↔O4 匹配本体感受对已被退役。**

此外，**CI / 快测 / demo 默认全部跑在 CPU surrogate + 代理 segmenter + ScriptedOracle 上**，真实组件只在 GPU 手动 gate 验证；CBF 零跌倒保证仅是降阶 LIP 模型属性 (在真实 Go2 上反保护)；数据集仅 413 条 (低于 1000 SFT 下限)。下面逐项列出当前为真的诚实局限。

---

## 1. 核心论点层面的诚实局限 (core-thesis limitations)

- **潜在路线 (B5) 与文本路线 (B4) 在归因准确率上打平** — `outputs/vla/ablation/M7_GAP1_RESULTS.md`, `outputs/vla/M7_RESULTS.md`, `suite_sem_{latent,text}.json`, CLAUDE.md #35。spec §3/§4 的中心主张"特权蒸馏潜在 Kino-Tokens 优于文本注入"在**贪婪解码归因准确率上零支持**：聚合 latent 0.974 == text 0.974；在 n=24 歧义子集上所有 proprio 保真度档 (binned/scalar/none) 全部打平于 1.000，因为持续均值已能分开各类。重述后的优势是必要性 (vs vision_only 0.458)、token 效率 (+6 vs +256，43×)、θ-grounding——**不是准确率**；准确率差距只会在 surrogate 低维 proprio 无法产生的 matched-mean / 真 1kHz regime 下打开 (未来工作)。

- **Embodied DPO 未稳健超越 SFT（exit-3 未达成，M7 框留空）** — `outputs/vla/M7_GAP2_RESULTS.md`, `outputs/vla/M7_RESULTS.md`, CLAUDE.md #34/#36, M7 checkbox。满数据 SFT 已在归因天花板 (0.974 temp-0 / 0.933 @0.8)，DPO 无 top-1 headroom 故打平 (0.917 vs 0.933 @0.8)；低数据下唯一的 DPO>SFT 增益 (贪婪 +8.4pt, 0.708→0.792) **在采样下反转** (temp-0.8 DPO 0.483/0.550 < SFT 0.650)，是 ~100 对的过锐化/过拟合。结论："DPO 只锐化已有能力，不能新增读 proprio 的能力。" §11 机制 (pref_acc→1.0) 已验证，但闭环 margin 推迟到 M8。

- **on-policy DPO 在无 action-span mask 时发散 (Thought 混淆)** — `outputs/vla/M7_GAP2_RESULTS.md`, `kino_vla/vla/model.py` build_inputs loss_span, `kino_vla/vla/dpo.py`。字面 §11 on-policy 过程在自由生成的 embodied CoT 上**发散** (pref_acc 0.156→0.188 倒退)，因为 completion_logprob 评分了 103 token 中 71% 的 Thought 散文而非 30 token 的 `<Action>` 决策。修复 (loss_span="action" 只评分决策跨度) 使其收敛 (pref 0.562→0.906)——**稳定性修复本身是真正贡献，但 naive spec 过程是坏的**；模板化的 canonical 对又因脱离模型自由生成分布而不迁移 (甚至把弱 SFT 从 0.708 拉到 0.500)。

- **闭环 Recoverability Dichotomy：正确归因并未转化为更好结果，因果盲 FSM 追平/击败 VLA** — `outputs/vla/dichotomy/{TABLE,M7_GAP3_RESULTS}.md`, CLAUDE.md §2 Gap-3。真实 Go2 闭环 (5 op × 3 seed)：B5 完整 VLA 的 no-fall 0.80 / reached 0.20 vs B2 因果盲 FSM 的 no-fall 1.00 / reached 0.33——baseline 在每个 outcome 上追平或更好，因为 VLA 的 push-through 恢复在危险上卡住而 FSM 盲绕。强证据仍**只是离线归因** (0.917 vs 0.250)；outcome 半边未在闭环证明。

- **决定性 O5 (16 kg 过载) 格被物理反转——正确的 Hold 反而把机器人弄倒** — `outputs/vla/dichotomy/M7_GAP3_RESULTS.md`。B5 正确归因 overload 2/3 并选正确的 Hold_and_Request，但机器人仍跌倒 2/3 (16 kg 过载把静止 Go2 压倒)，而 FSM 的缓慢运动保持动态直立。"Hold 是安全的过载恢复"在此严重度不成立——**恢复库 (而非归因) 才是瓶颈**。

- **train/deploy proprio shift：bang-bang 采集窗口对平滑部署 OOD** — `kino_vla/tokens/dataset.py`, `kino_vla/data/isaac_rollout.py`, `kino_vla/data/snapshot.py`, `kino_vla/vla/planner.py`, configs/recovery/fsm_isaac.yaml, CLAUDE.md #34c。失败快照在 bang-bang 激励下采集 (使 O5/O10 effort 可观测)，但部署平滑巡航，故运行时 proprio 对 projector 标准化器 OOD：无主动探针时归因坍塌为 overload (O1 ice 0/3)；主动探针是 tradeoff——修 O1/O2 表面危险却破坏全局 O5 payload (→low_friction 0/3)，**没有任何单一配置能正确归因所有 op**。Isaac 闭环 SFT temp-0 仅 1/3 成功。

- **"视觉不可替代"的 O2↔O4 匹配本体感受对已退役** — `kino_vla/eval/ambiguity.py`, `kino_vla/sim/operators/o2_compliance.py`, tests/test_ambiguity_pairs.py, CLAUDE.md #38。O2 由弹簧陷阱改为物理真实的有界拖曳场后，mud 与 tether 的阻力曲线发散 13.1 N，匹配脚本 verdict=FAIL，故头牌 P4 "纯本体感受基线必失败、视觉不可替代"的构造性证据**作废**；两个匹配断言被 skip (保留在册)。重设计的 vision-necessary 对是待定的人类 spec 决策。核心贡献 (matched-appearance 对 + O7) 不受影响。

---

## 2. 代理实现 (surrogate substitutions still in place)

- **CPU surrogate backend 是 CI/demo 的常驻交付路径** — `kino_vla/sim/surrogate.py`, `kino_vla/sim/backend.py`, `scripts/run_demo.py`, tests/conftest.py, CLAUDE.md #4。walking skeleton / run_demo / 全部快测跑在平面点机器人 (单车模型) 上，其物理常数"为合理四足尺度选取，非系统辨识"；所有 `@pytest.mark.sim` Isaac 测试在无 isaaclab 时自动 skip，**CI 从不触碰真实 Go2**。按 §0 它永远不是实验/结果。

- **默认 CI / 快测 / demo segmenter 是 class-hash 代理，而非真实感知** — `configs/map/traversability_v0.yaml` (`segmenter: surrogate`), `kino_vla/map/segmentation.py`, `appearance.py`。共享 config pin `surrogate`，其 appearance 是 `SHA256(class_name)→单位向量`，`segment` 直接取场景世界矩形、从不进入像素空间。所有常跑的 map 回归测试在标签哈希而非真实 CLIP/RGB-D 上验证 propagation/costmap。

- **标准 ClipSegmenter 用标签键控的程序化纹理而非相机像素** — `kino_vla/map/clip_segmentation.py`。可选的 `clip` 前端跑**真实 CLIP**，但输入是由 ground-truth `appearance_class` 键控的 `material_texture()` 图，几何是 ground-truth Rect 而非深度反投影。仅 LiveRtxSegmenter (仅 Isaac demo) 喂真实相机像素。

- **ScriptedOracle 是外部 Oracle LLM 的可控代理** — `kino_vla/data/oracle.py`, `scripts/build_hindsight_dataset.py`, configs/data/hindsight.yaml, CLAUDE.md #30b。CI/吞吐数据路径用从特权快照**推导**真类 (是生成数据不是推理) 并按固定率虚构的 ScriptedOracle；其 CI 虚构是按 filter 自身失败模式合成的 (wrong_attribution/wrong_primitive/unsafe_detour)，故 filter 主要在合成虚构上验证，仅 19 条小型 gpt-5.5 真跑验证真实 LLM 虚构。

- **真实数据集由 gpt-5.5 经网关标注，而非 spec 命名的 Gemini 3.1 Pro，且不可复现** — `kino_vla/data/dataset.py`, `oracle.py`, dataset_card.md。真实 CoT 由 gpt-5.5 经 OpenAI 兼容网关标注 (spec §10 命名 Gemini 3.1 Pro)；真实 LLM 数据集**非比特可复现** (外部非确定性 API)，只有 ScriptedOracle 路径在 CI 可复现。

- **CI Hindsight 流水线跑在 CPU surrogate 上，而非真实 Go2** — `kino_vla/data/pipeline.py`, CLAUDE.md #30。高吞吐 CI 数据流水线 (run_pipeline / _collect_snapshot) 经 SurrogateBackend 闭环驱动 maze 格，非 Isaac；spec 的 Isaac 数据路径是单独的解耦采集。

- **DPO 默认 rollout backend 是 surrogate；闭环/Suite-Sem eval 默认是 oracle stub** — `configs/vla/dpo.yaml`, `scripts/eval_vla_closed_loop.py`, `kino_vla/vla/planner.py` StubVlaPolicy。DPO 偏好对采样默认 surrogate (注释承认交付须 override 到 isaac)；闭环与 Suite-Sem eval 默认 StubVlaPolicy——一个读特权 operator/θ、按构造完美归因 (acc 1.0) 的 oracle，**CI 绿灯从不测真实学得的归因**。

- **O4 tether 用外部躯干 wrench 而非 spec 的 D6 弹簧-阻尼关节** — `kino_vla/sim/isaac_policy_backend.py`, CLAUDE.md #20。spec §8.2 命名 D6 弹簧关节；真实 Go2 上用等效 Hooke 律/拖曳外部 wrench (因运行时 D6 关节 prim 创建在 Isaac Lab 2.3.0 脆弱)，同物理效应但非 spec 关节。O2 compliance 用同一 wrench 路径。

- **Suite-Sem FSM "基线"是常数多数类预测器，非真实 FSM/调优 planner** — `kino_vla/eval/suite_sem.py` MajorityBaselinePolicy。头牌 exit-1 margin (VLA 0.974 vs FSM 0.237) 是对常数多数类 ('low_friction') 而非真实 FSM (字面 FSM stub 总是 Backstep、归因准确率 0)；调优的 B2 基线推迟到 M8。

- **M5 strict §7 gate 在程序化渲染上验证 (非实时相机馈送) + 颜色直方图编码器 (非 CLIP)** — `outputs/map/perception_strict.md`, dataset_card.md, CLAUDE.md #27/#18a。strict RGB-D 反投影几何为真，但所用 appearance 编码器是颜色直方图、深度是合成渲染。RgbdSegmenter (`kino_vla/map/rgbd.py`) 设计上无相机。

- **M5 appearance 编码器/ambiguity 匹配是 surrogate-only** — `kino_vla/map/appearance.py`, `kino_vla/eval/ambiguity.py`, CLAUDE.md #18。比特相同的 O4↔O2 匹配只在受控 surrogate 点机器人上"by construction"精确；真实 Isaac 轨迹接触噪声、非比特相同，故该构造性歧义证明是 surrogate-only 工件。

- **surrogate 闭环排除 O1 (A 类 ice 存活不可表征)** — `kino_vla/vla/scenarios.py`, tests/test_vla_dpo.py, CLAUDE.md #34e。surrogate 跌倒模型使任何 ice 穿越致命，故 O1 的 A 类"慢行继续即存活"不可表征、被排除 (surrogate scenario 集仅 O8+O3 两个 op)；DPO 配对测试因此把 real-stack success flag 手动盖到正确 rollout 上而非测量它。

---

## 3. 硬件/环境受限 (hardware/environment-blocked)

- **无 GPU CI runner——sim 验证永远是手动 gate** — CLAUDE.md #2。CI 仅 lint+unit，每个 `pytest -m sim` gate 都是在 GPU 箱手动跑的文档化 gate，**sim 验证从未自动化** (当前仍为真)。

- **闭环 map config 仍用程序化渲染而非实时 RTX 相机** — `configs/map/traversability_v0.yaml` camera block, `scripts/isaac_m5_perception_check.py`。注意：实时 RTX 相机硬件阻塞 (旧 595 驱动 segfault) **已在驱动 580 + CUDA 12.8 解决** (CLAUDE.md #28，独立 demo `clip_rtx.md`/`rtx_perception.md` 实时跑通)，但闭环 map config 此处仍走程序化 ground-plane 渲染路径，真实相机馈送描述为 drop-in。

- **`isaac_m5_perception_check.py` 排除于 sim gate + 陈旧硬件阻塞文档** — `scripts/isaac_m5_perception_check.py`, `kino_vla/map/pixel_appearance.py` docstring。该实时相机检查保留但排除于 sim gate 且非 CLIP (仅像素编码器)；其 docstring 仍声称"Real CLIP is blocked"并引用陈旧的硬件阻塞理由——而 CLAUDE.md 记录该箱实为 RTX 5090 / 驱动 580 且已解决，故**记录的硬件阻塞理由是陈旧的**，脚本本身仍未跑/被排除。

---

## 4. 物理/安全保证的范围限制 (scope of the physical/safety guarantees)

- **CBF 零跌倒是降阶 LIP 属性，在真实 Go2 上反保护** — `kino_vla/shield/adversarial.py`, `qp.py`, `scripts/isaac_cbf_pushfall.py`, CLAUDE.md #13/#23。头牌"shielded 零跌倒 / bypassed 非零"只在 CPU surrogate 点机器人的参数化 capture-speed 跌倒模型上成立。在真实全阶 Go2 上 reduced-LIP CBF **从不减少跌倒**，对前向/对角推力反保护 (bypassed 0/3、shielded 3/3)。零跌倒是降阶模型属性；真实 Go2 主张降为命令钳制，可证伪零跌倒测试留在 surrogate。

- **冰/持续滑移在 capture-point 保证之外** — `kino_vla/shield/cbf_shield.py`, `qp.py`, CLAUDE.md #13/#21。§6.6 CBF 仅保证 0 步可捕获 (无 capture-point 倾覆)；极低 μ 冰上的失败是持续滑移而非倾覆，CBF 不阻止、甚至反保护 (halt-then-creep)。M3 对抗 gate 仅干地；冰恢复交给 M7 planner (Set_Constraint/Hold_and_Request) + M4 μ̂ 头。

- **偏航命令 (wz) 不经 shield 过滤** — `kino_vla/shield/cbf_shield.py`, `lip.py`。CBF-QP 仅投影 2D 平移 ZMP/速度；偏航率在 feasible/intervening 路径直接透传，仅在 HALT 时归零。LIP 降阶模型不含角动力学，故敌意/幻觉偏航不被 barrier 裁决。

- **不可行 fallback 终点是 HALT (急停)，非恢复** — `kino_vla/shield/cbf_shield.py`。QP 与所有 fallback 模式都不可行时 shield 的终极动作是零速急停 (设计上安全、不软化约束)，但最后手段是把机器人停死、不是救回——恢复是 planner 的事。

- **摩擦锥是保守内多边形近似，μ 默认名义值** — `kino_vla/shield/qp.py`, `cbf_shield.py`。摩擦锥用 N 面内正多边形近似圆盘 (可行集 ⊆ 真盘)；shield 在 config 名义 μ (0.8) 上跑，除非 M4 μ̂ 头调用 set_mu_estimate，而该感知→安全耦合本身在 M4 仍是 surrogate-only。

- **"精确"投影中两个被接受的潜在缺陷** — `kino_vla/shield/qp.py`, CLAUDE.md #14。`feas_tol=1e-7` 可让空多面体在间隙 <100 nm 时读为可行；`‖a_j‖²≤1e-12` 的近零范数行被跳过垂足候选 ("可行 iff 非空"契约仅一般情形完美)。接受为潜在 (shield 归一化输入不可达)，但缺陷仍在代码中。

- **Reflex 在 Isaac/真实 Go2 路径上是 no-op 占位** — `kino_vla/sim/isaac_policy_backend.py` set_reflex。Reflex 层 (蹲伏/加宽延长存活) 在真实 Go2 上是显式 no-op；T_safe 存活延长结果只在 CPU surrogate 上产生 (仅缩放跌倒积分器阈值)。

- **Reflex / 存活 / 推恢复仅在 surrogate 点机器人上量化** — `kino_vla/monitor/reflex.py`。run_survival_episode / run_push_recovery_sweep 绑定 SurrogateBackend，"加宽/降低站姿抬高倾覆阈值"是 surrogate 的 set_reflex 开关，未在真实 Go2 上测量。

- **Monitor ROC AUC 在 surrogate 上计算 (非真实 Go2)** — `kino_vla/monitor/roc.py`。Kino-Monitor ROC (spec §12 指标、M2 exit、AUC≈1.0) 由 SurrogateBackend 前向驱动生成，反映 surrogate 干净 proprio 而非真实接触噪声；真实 Go2 需 backend 专属重标定。

---

## 5. 可观测性与数据局限 (observability + data-scale limitations)

- **O5 过载经 effort 通道真正不可观测** — `kino_vla/sim/operators/o5_payload.py`, `isaac_policy_backend.py`, `kino_vla/data/isaac_rollout.py`, configs/data/hindsight.yaml, CLAUDE.md #15/#31/#32。负载分散到四腿，平均扭矩从不达饱和 (eff_max=0.00 各质量)；≥6 kg 在饱和前就跌倒。O5 只在 ~16 kg 重载下作为躯干蹲伏出现 ("overload"签名是 crouch+grip 的 workaround 而非直接 effort 读数)，且从随机化 scale 采集 lane 中**排除**；轻过载在真实可观测性底。

- **bang-bang 激励才能观测 payload/effort θ，巡航不行** — `kino_vla/tokens/dataset.py`, CLAUDE.md #15。payload (O5) 和 effort-decay (O10) 仅在 actuator effort 预算 binding 时留 proprio 痕迹，需刻意 bang-bang 方波激励；平缓巡航从不达需求，故 O5/O10 不可辨识、θ 回归头只能回归先验均值。训练数据因此是人为激励而非名义运动。轻 payload 端 (≲4 kg) 在真实可观测性底，MAE 仅以 ~10% margin 过 1.5 kg 杠。

- **测得的"不可观测底"：中 μ 拐点 + 轻 effort 余量从训练**和**评测双排除** — `kino_vla/tokens/isaac_gate.py` filter_observable, configs/tokens/extractor_v0.yaml。摩擦鲁棒性拐点 (μ≈0.2–0.5，同 μ 既滑又不滑，pred 0.43±0.28) 和轻 effort 切 (慢 trot >2× 扭矩余量、与健康同) 从训练+严格 Isaac gate 双排除——extractor 在那里无法恢复 θ，这些档被丢弃从不 gate。

- **μ MAE 杠本身只反映"分辨冰/坚实"，非全程摩擦恢复** — `configs/tolerances.yaml`, `isaac_gate.py`。μ 容差显式定义为"分辨滑冰 vs 坚实地面"而非全范围恢复；中 μ 在摩擦鲁棒策略上不可观测被排除，故通过的 MAE 按构造只在受限可观测 regime 上。

- **413 条 SFT 是 proof-of-pipeline 规模 (低于 1000 下限)** — CLAUDE.md #34f, configs/data/hindsight.yaml (sft_scale_floor: 1000), `outputs/hindsight_isaac/dataset_card.md`, `outputs/vla/M7_RESULTS.md`。canonical 真 Go2 数据集 413 条 (split 315/49/49)，低于配置的 1000 SFT 下限，card 自动标记为 proof-of-pipeline；held-out test 仅 n=38 (n=24 歧义)，置信区间宽。M7 exit-1 结果强但集小。

- **每 op 偏斜；embodiment op (O5/O10) 在可观测性底稀缺** — `outputs/hindsight_isaac/dataset_card.md`。kept 计数 95/22 (max/min)：O7/region op 主导，O5 (22)/O10 (22) 因可观测性底稀缺，O2_compliance 也低保留率 (53/111)。

- **O5/O10 embodiment 数据由工程化 straddle-the-onset 采集而非自然拦截** — `kino_vla/data/isaac_rollout.py`。O5/O10 无空间 locus、无滑移/视觉 tell，故流水线驱动健康基线再装入故障、在固定/effort 触发延迟捕获使窗口跨故障起点——是让因果可读的工程化条件 regime；#31 记录：暴露 base_height 前每个 O10 都被误标 region_collapse。

- **O10 / O5↔O10 对在主动探针重采集中丢失** — CLAUDE.md §2 Gap-3 Path B, `outputs/vla/dichotomy/M7_GAP3_RESULTS.md`。499 探针快照→255 kept，O10 归 0 (探针窗口误标)，O5↔O10 歧义对在探针数据集中丢失；探针机动与 effort-decay 签名不兼容。探针窗口 SFT 归因 0.917 (略逊 bang-bang 0.974)。

- **O8 隐形碰撞体是 leave-one-out——模型未见、默认 overload** — `kino_vla/sim/operators/o8_invisible_collider.py`, `kino_vla/vla/scenarios.py`, configs/data/hindsight.yaml, CLAUDE.md #34d。O8 不在 M6 dataset_lanes，模型从未见、闭环默认归因 overload，从同分布归因指标和默认 `--ops` 集排除——对未见 operator 的泛化主张不被支持。

- **闭环整体 VLA 归因低 (0.40)，O3/O4 完全失败** — `outputs/vla/dichotomy/{TABLE,M7_GAP3_RESULTS}.md`, `suite_sem_probe_latent.json`。闭环 VLA 归因仅 0.40：O3 (collapse) 普遍读成 low_friction (O1↔O3 滑移歧义，0/3，探针 SFT 上 O3_collapse 准确率亦 0.0)；O4 (tether) 从不产生归因 (monitor 不在拽拉上触发，None 0/3)。

- **10 条快照丢于瞬态 Oracle 网关失败** — `kino_vla/data/schema.py` (ORACLE_ERROR), pipeline.py, dataset_card.md。10 条快照丢于瞬态网关 API/传输失败 (作为 oracle_error 从 reject-rate 排除，故引用的 21.0% 只反映 filter 判断)，可经 retry 脚本恢复但当前不在 kept 集。

- **rule monitor 在 O4 tether 上正确不触发 (覆盖缺口 9/10)** — `kino_vla/monitor/rule_monitor.py`, CLAUDE.md 2026-06-17 M2 log。短暂拽拉低于通道阈值，故 monitor 不在 O4 上触发 (报为正确不触发——强触发会损精度)；意味着 rule monitor 在 adhesion op 上有已知覆盖缺口，检测 9/10。

---

## 6. 被跳过的测试 / CI 中固定的代理 (skipped tests + CI-pinned surrogates)

- **O2↔O4 匹配本体感受 P4 对的 2 个断言被 skip** — `tests/test_ambiguity_pairs.py`。`_RETIRED` 原因 "O2 现为拖曳场非弹簧 (§6 #38)"，两个匹配本体感受断言 `@pytest.mark.skip` (保留在册非删除)；appearance 可分性测试仍过。

- **torch-gated 学得组件测试在 CPU/CI 箱 skip** — `tests/test_extractor.py`, `tests/test_vla_projector.py`。用 `pytest.importorskip("torch")`，故无 torch 的 CI 箱完全跳过——Kino-Tokens extractor 和 Kino-Projector 不被快测触碰。

- **真实 CLIP 测试在 CLIP 不可用时优雅 skip** — `tests/test_clip_map.py`, configs/map/traversability_v0.yaml。共享 map config pin `surrogate` 保 GPU/模型-free demo gate 绿；真实开放词表 CLIP 只在 Isaac demo 传 `{segmenter:'clip'}` override 时跑，真 CLIP 测试在 CLIP/transformers/权重不可用时 `pytest.skip`。

- **所有 `@pytest.mark.sim` Isaac 测试在 isaaclab 缺席时自动 skip** — `tests/conftest.py`。CPU dev/CI 箱上自动 skip，CI 从不行使真实 Go2。

- **ApiOracle.from_config 在 CI 未行使** — `kino_vla/data/oracle.py`。真实 LLM 客户端及其 OpenAI/Anthropic/Gemini 传输工厂在 CI 不行使 (无 key/网络)；测试注入 stub `complete`，只覆盖 prompt-build + 传输契约。

- **committed `dpo.yaml` 不设 loss_span (仅通过 --set 传)** — `configs/vla/dpo.yaml`, `scripts/run_m7_dpo_*.sh`。修复 on-policy 发散的关键 `loss_span=action` 只在 headroom/scale 脚本经 `--set` 传，committed dpo.yaml 无该键 (默认 whole-completion，即发散配置)。

---

## 7. 其他 / 轻微 (other / minor)

### 仿真物理近似 (surrogate & Isaac)
- **Traction 模型是刻意降阶的摩擦替身** — `kino_vla/sim/traction.py`。摩擦受限速度跟踪模型显式是降阶模型，真实 PhysX 接触仅在 GPU/Isaac 路径替代其角色。
- **Surrogate 跌倒模型是手调不稳定积分器** — `kino_vla/sim/surrogate.py`。跌倒建模为持续滑移/推后速度超越积分越阈值——0 步可捕获的可证伪代理，非真实倾覆动力学。
- **Surrogate 侧滑是种子高斯噪声** — `kino_vla/sim/surrogate.py`。冰漂移注入为按滑移缩放的种子高斯 (skid_noise_std·slip·√dt·randn)，是滑移侧移的现象学代理。
- **Surrogate base_height/tilt 通道平直** — `kino_vla/sim/surrogate.py`, `kino_vla/tokens/features.py`, CLAUDE.md #16。tilt 硬编 0.0，base_height 仅经 O2 sink 偏移变化；11 个特征通道中 base_height/tilt 在 surrogate 携零信息 (仅为可移植保留)。
- **O3 collapse 触发用 dwell-time 代理 spec 的接触载荷阈值 F_th** — `kino_vla/sim/operators/o3_collapse.py`。
- **O8 隐形碰撞体的渲染抑制只是装饰** — `kino_vla/sim/isaac_policy_backend.py`。Isaac 上是真实 PhysX cuboid 墙 opacity 0.15 (faint 非真隐形)。
- **Isaac effort_ratio 对当前 (O10 已缩放) cap 测量，非名义 cap** — `kino_vla/sim/isaac_policy_backend.py` _measure_effort (刻意代理选择)。
- **O2 几何 sink (d_sink) 在 Isaac wrench 路径未建模** — CLAUDE.md #18c；只施加切向阻力，base height 物理驱动 (sink 是 surrogate-only)。
- **O4 tether peel/penetration grip 是为"退避即逃逸"迭代调出的工程 hack (Bug-1)** — `kino_vla/sim/types.py`, surrogate.py, isaac_policy_backend.py；调出的机制而非第一性原理 adhesion 模型；body-frame 形式曾在转向时坏。`o4_tether.py` 与 `types.py` ResistanceRegion docstring 仍描述已退役的 F=k·s 弹簧/匹配对模型 (陈旧 docstring)。
- **O5 只发 static-overload，pendulum-slosh/sudden-detach 变体推迟** — `kino_vla/sim/operators/o5_payload.py`。
- **mass_kg 在 PhysX mass view 不可用时回退 config 常数** — `kino_vla/sim/isaac_policy_backend.py`。
- **推理关闭域随机化 (固定摩擦/质量、无 obs 噪声)** — `kino_vla/sim/isaac_policy_backend.py`；部署评测在固定无噪 embodiment 上跑，非随机化训练分布。
- **Isaac backend 单进程单 episode，destabilizing op 需独立横向 lane** — `kino_vla/sim/isaac_policy_backend.py` (clear_payload 因 reset 不还原质量而存在)。

### 感知 / 地图算法
- **pale-ice 同材料 cosine 0.722 < 0.9 propagation 杠 (network-free 像素编码器亮度脆弱)** — `outputs/map/rtx_perception.md`, `scripts/m5_rtx_camera_perception.py`, `kino_vla/map/pixel_appearance.py`；近白冰两视图不互相 propagate；spec §7 命名 CLIP 为光照不变补救 (注：默认前端已换真 CLIP，故此为旧像素前端局限)。
- **CLIP 在平色块上失败 (OOD)，需合成纹理才工作** — `kino_vla/map/clip_segmentation.py`, CLAUDE.md。平单色块所有 cosine 坍塌至 ~0.99 (ice→adhesive)；靠程序化纹理修复而非真实纹理地形。
- **真 CLIP 在实时 RTX 相机上同/跨 cosine margin 紧 (0.956 vs 0.918)** — `outputs/map/clip_rtx.md`；共享 RTX 光照致 margin 比程序化渲染 (0.97/0.86) 更紧；ice→ice CLIP 概率仅 0.38；开放词表标签鲁棒但 propagation 的 cosine 分离弱。
- **实时相机语义掩码是 ground-truth USD 标签，非学得/SAM 开放词表分割** — `kino_vla/map/live_rtx_segmenter.py`, `kino_vla/sim/isaac_policy_backend.py` add_textured_patch；"哪些像素是危险"来自 USD 类标签 (特权 ground truth)，只有标注 (CLIP) 和几何 (ray∩ground) 是感知的。
- **实时相机 footprint 用 ray∩z=0 地平面近似而非完整深度反投影** — `kino_vla/map/live_rtx_segmenter.py`；假设危险都平躺地面，对非平/抬升障碍 (O8/O9 类) 会错。
- **实时相机 footprint 误差 ~0.18 m 仅单次探针证明** — `kino_vla/map/live_rtx_segmenter.py`, `scripts/isaac_perception_probe.py`；proof-of-pipeline 点，非跨 op/位姿的标定精度。
- **SurrogateSegmenter O7 深度污染是 2D 中心位移，非真实深度几何** — `kino_vla/map/segmentation.py`；默认 surrogate 路径用 rgbd.py docstring 批评的 2D 捷径。
- **costmap 维度不匹配守卫静默丢 surrogate 特征 (无 propagation)** — `kino_vla/map/traversability_map.py`；64 维场景真值特征对 512 维 CLIP costmap 落零向量，标记落地但无相似度 propagation。
- **propagate_similar / untraversable_points 是 O(grid) 全扫描 (无空间索引)** — `kino_vla/map/costmap.py`；正确性优先参考实现，不断言每步预算、不 scale。
- **nav_hazards 聚类是粗贪婪单链桥到 circle detour** — `kino_vla/map/traversability_map.py` ("粗但可复现"；注：几何 avoid-disc/detour 已按用户指令从 VLA planner 移除，此为遗留桥)。

### VLA planner / 训练近似
- **名义 nav-pick 是 recovery-LoRA 零样本用 (未训 nav)** — `kino_vla/vla/prompt.py`, `planner.py`, CLAUDE.md #39a；在 demo 场景工作但 LoRA 从未训 nav，难布局可能需 nav 训练数据 (加 Turn 文本会移动零样本 pick 致脆弱 Go2 倾倒)。
- **co-trained nav 模型把 Turn 坍塌成 Waypoint——held-out nav kind-acc 0.679** — `outputs/vla/nav_eval2.log`, `kino_vla/vla/sft.py`；Turn 被预测为 Waypoint 7/8 次 (尽管 oversample 平衡 hack)。
- **委派恢复 (每次触发一次反思) 是漂移规避 workaround** — `kino_vla/vla/planner.py`, CLAUDE.md #39(2)；每 ~1Hz tick 重反思会漂移 (adhesion→Backstep 翻成 overload→HALT 把狗困住)，靠每次 monitor 触发只反思一次绕过底层不稳定。
- **名义巡航按设计走进危险 (无 see-and-avoid)** — `kino_vla/vla/scenarios.py`, CLAUDE.md #39b；核心论点是 step-in→feel→recover，视觉预避被显式撤回为损害论点。
- **latent projector θ-grounding 在 4 通道中 2 个噪声 (mu, support 超 M4 容差)** — `outputs/vla/ablation/M7_GAP1_RESULTS.md`, `kino_vla/vla/projector.py`, CLAUDE.md #35；315 样本辅助头解 effort (0.083)/payload (1.065) 在容差内，但 mu (0.242, 2.4×)/support (0.245, 2×) 超差。
- **FsmRecovery 是显式 M1 脚本 stub** — `kino_vla/vla/fsm_recovery.py`, rollout.py；因果盲 Backstep+detour (AvoidCircle "stub 拓扑记忆")，作 B2 基线，按字典是闭环上追平/击败 VLA 的恢复，只在 M8 升为调优基线。
- **DPO ambiguity-pair 构造刻意 sidestep (而非解决) proprio shift** — `kino_vla/vla/dpo.py`；同分布构造刻意绕开 bang-bang→平滑部署 shift，确认部署 shift 仍是离线指标回避的未解局限。
- **Embodied DPO 主指标 (闭环 nav 成功) 被 train/deploy proprio shift 阻塞** — `outputs/vla/M7_GAP2_RESULTS.md`, CLAUDE.md #34c；spec §11 主指标是闭环导航成功，被未解 shift (Gap-3) 阻塞，真正 DPO>SFT 演示推迟 M8；M7 只交付前提 (稳定 on-policy DPO) + 诚实表征为何归因代理饱和。

### 监控 / 标定 / 数据脆弱性
- **monitor 标定按 backend 和步态专属 (策略变更须重调)** — `kino_vla/monitor/rule_monitor.py`, configs/monitor/rule_v0_isaac.yaml, CLAUDE.md #10/#29；tilt 通道在平 surrogate 惰性 (tilt≡0)；策略重训 (摩擦-DR 底 0.3→0.08) 后新步态再滑、monitor/FSM 须重调 (cooldown 1.5→6 s 等)——脆弱手调操作点而非学得/鲁棒检测器。
- **摩擦-DR 底降至 0.08 使冰失败可观测 (M4↔demo 张力)** — `configs/locomotion/go2_flat_ppo.yaml`, CLAUDE.md #9/#29；单一底须同时服务 informative-ice-windows 和闭环冰穿越稳定——文档化张力，真实修复是 VLA planner 替换易抖 FSM stub。
- **采集 monitor 是高召回操作点，非部署 ROC 点** — `configs/monitor/collection.yaml`；M6 数据采集 monitor 刻意降阈、缩短 arm/debounce (高召回) 拦截每个异常，与部署精度/召回操作点不同。
- **JSON 解析器容忍 JSON 周围散文 (最后平衡对象胜)** — `kino_vla/data/schema.py`；启发式取最后解码为带 attribution/action dict 的平衡 `{...}` 跨度，是宽松启发式而非严格结构化输出强制。

### spec 偏离 (待人类决策)
- **spec backbone Qwen2-VL 升级到 Qwen3-VL-4B；Qwen-RobotNav 架构不匹配** — CLAUDE.md #33；M7 用 Qwen3-VL-4B+LoRA (spec 未编辑——须人类协调)；请求的 Qwen-RobotNav 是导航模型、不原生发 `<Thought>` CoT + 结构化 `<Action>`，作推理 backbone 架构不匹配。
- **O7 deception vs ambiguity-pair 框架——single-thesis 句未定，待人类** — CLAUDE.md #37；框为"无单一模态足够"的互补实例，但 spec 里的单论点句仍须人类敲定。

### Pass-through shield stub + monitor 文本 stub
- **Pass-through shield 是 M1 stub (仍在)** — `kino_vla/shield/passthrough.py`；透传每命令从不干预，被 CbfShield 取代但留在码库 (CbfShield 仍 import 其 ShieldDecision)。
- **Monitor 文本异常摘要是 stub (无物理归因)** — `kino_vla/monitor/rule_monitor.py`；MonitorEvent.summary 是模板 f-string，不做物理归因 (分不清冰/薄冰、泥/胶)，grounded 归因由 M4 extractor + M7 VLA 提供。

### OOD 检测器 / 辅助机制
- **OOD 检测器在重构残差外有 latent-Mahalanobis fallback** — `kino_vla/tokens/extractor.py`；spec §9 主用重构残差，另拟合 latent 高斯暴露 Mahalanobis 距离作 fallback (次级/备份机制而非单一验证 OOD 信号)。
- **Kino-Text 对比头是辅助-only、学得原型表，非真文本编码器** — `kino_vla/tokens/semantics.py`, extractor.py；经 InfoNCE 对齐小型每-regime 原型表"尚无重量级文本编码器"，真开放词表对齐 (换冻结 sentence/CLIP 编码器) 仅 M7 升级路径。
- **support-ratio 目标是尾随窗口均值非瞬时值** — `kino_vla/tokens/isaac_gate.py` smooth_support；真 trot 上瞬时 support 是步态相位噪声，监督目标用因果尾随移动均值近似。
- **anomaly-gated 注入：μ̂ 在 monitor 静默时弛豫回名义并 hold 防抖** — `kino_vla/tokens/coupler.py`, extractor.py；extractor 仅 monitor 唤起时跑，稳态 token 实质归零、μ̂ EMA 回 config 名义——μ̂ 非连续估计、被 gated 且 off-patch 弛豫。

### 证据文件缺失
- **文档化的"测得"不可观测底证据文件 (m4_diagnostic.txt) 不在盘上** — configs/tokens/extractor_v0.yaml, `kino_vla/tokens/isaac_gate.py`, outputs/gpu_audit/；config 与代码均引用该测得证据文件，但 outputs 目录无此文件 (仅 cam_probe / m6_hindsight)，故"测得而非假设"主张当前无法从仓库验证 (outputs gitignored)。

### 训练数据组成
- **训练数据仅单 operator——组合 (Suite-Comp) 物理仅测试不训练** — `kino_vla/tokens/dataset.py`, semantics.py；蒸馏数据集只含单 operator episode (干净五选一标签)，组合/多 operator 物理刻意从不训练只测试，故 extractor θ 回归和 regime 分类器不为共现失败训练。

---

## 附录：优先关注项 (最影响论文可信度)

若从 182 条里挑出最影响论文可信度的，建议排序：

1. **第 1 节全部** —— 直接关系论文核心贡献能否站住。尤其"latent 优于 text 在准确率上无支持"和"DPO 未超越 SFT"，目前是诚实地"未证明"，不是"已证明的正面结果"。
2. **闭环 outcome 半边失败 + O5 反转**（第 1 节）—— "correct attribution ≠ better recovery" 的硬证据，说明瓶颈在恢复库工程而非归因，需在论文里明确切分 claim。
3. **`m4_diagnostic.txt` 证据文件缺失**（第 7 节）—— 唯一一条"声称有但仓库里查不到"的项，建议重新生成落盘，否则 "measured not assumed" 的方法学论点无法复现。
4. **committed `dpo.yaml` 默认是发散配置**（第 6 节）—— 任何人 clone 后跑默认 DPO 都会复现发散，关键修复 `loss_span: action` 只在 shell 脚本里，建议固化进 committed config。
