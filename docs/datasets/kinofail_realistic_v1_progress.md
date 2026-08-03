# Kino-Fail realistic v1：实施与证据状态

更新时间：2026-07-23

## 当前结论

### 2026-07-23：v33 新开发 cohort 与 v32 O4 开发结果封存

v33 从 candidate 113–136 的冻结候选流按顺序处理，在首次满足配额的前缀 113–122 停止；
10 个已处理候选中准入 DiningRoom113、Office114、LivingRoom115、Bedroom118、Kitchen122，
共 5 scene/5 family，candidate 123–136 保持 untouched。v33 postrun 为 `passed=true`、
`sealed=true`，SHA-256 `88841b4ce8715845fd39086b72998cd63e159273dbef135e55c97a19a017df12`。

v32 随后在完整五场上一次性执行首次 attachment 后 1 step 的 action-agnostic 分支与完整 horizon
Backstep clearance。artifact/causal integrity 为 **5/5**，预注册严格结果为 **4/5**；五例 Continue
均在分支后跌倒，五例 Backstep 均安全、完成 clearance、发生真实 peel 并后退 0.208–0.275 m。
LivingRoom115/Kitchen122 的倒地 Continue outcome 已由冻结 state-aware visual gate 正确接纳。
唯一负例 Bedroom118 的 Backstep 完整成功，失败只来自 Continue 倾覆后横向偏差达到 0.403 m，
触发旧的双分支全时域 0.30 m route 门。

逐帧诊断没有事后改判：v32 仍永久封存为 4/5。Bedroom118 的 matched prefix 最大偏差仅
0.0166 m；物理失稳起点 step 132 的偏差为 0.0295 m，step 171 才在高度 0.206 m、倾角
0.588 rad 时越界，step 178 判倒。下一版保持 v32 动作/物理不变，只把 route nuisance gate
前瞻性改为“matched prefix + Backstep 全时域受限，Continue 截至物理失稳起点受限”，并在
candidate 137+ 的全新 operator-blind cohort 一次确认。完整分析见
`docs/datasets/kinofail_o4_v32_development_analysis.md`。这仍不是 O4 held-out confirmation，
也不计 A0–A7；readiness 保持 **0/8**。

### 2026-07-23：v30 新场景 cohort 与 v31 O4 held-out 负结论封存

v30 在完全不查看 O4 outcome 的前提下，按冻结候选顺序处理 candidate 89–105，并在首次达到
“至少 5 个 admitted scene、至少 4 个 room family”的前缀立即停流。17 个候选中准入 6 个：
Office90、LivingRoom91、Office96、Bedroom100、Office102、House105，覆盖 4 个 family；
106–112 后缀保持未触碰。所有 source、route、RTX、Go2-front 和 stack rejection 均留在分母。
v30 postrun 为 `passed=true`、`sealed=true`，SHA-256
`06860cf6d3f47b8b8dadc2b1dda80157809074185e8fecabbc5e0841cdd91e66`。

v31 随后把完全不变的 v27 collector、物理后端、policy 和参数一次性运行到全部六场。结果的
执行与 artifact 完整性为 6/6，但严格 outcome 只有 **1/6**；postrun 为
`integrity_passed=true`、`outcome_passed=false`、`sealed=true`，SHA-256
`9657f2a9ac97d59f42ad18232bfebe756a33c12ec890bb2d435d895fb6b9caa4`。Office96 全门通过；
LivingRoom91/House105 在 attachment 后 6 steps 才由 posture urgency 分支，此时 Backstep
仍跌倒；Bedroom100/Office102 安全后退 0.209/0.279 m，但 120-step 上限后仍有 active foot，
未完成 clearance；Office90 的物理恢复完整，但 Continue 倒地后的 body-fixed outcome 因低纹理
未过通用图像门。Office102 同时有 clearance 与倒地 outcome 图像问题。

逐帧复核纠正了一个重要归因：两例低纹理只发生在已跌倒 Continue outcome；initial、decision
以及安全 Backstep outcome 均合格，不能归因于 decision snapshot 或简单归咎 EmbodiedGen UV
warning。v32 因此冻结为：首次 attachment 后 1 step 的 action-agnostic 早分支；在剩余 horizon
内持续 Backstep，直至后退距离与 active-feet clearance 同时满足；视觉 QA 按 inference state 与
fallen consequence state 分层。v31 六场永久不重跑，v30 的 106–112 后缀也继续 untouched；
v32 只能从 candidate 113 以后的新 development scene 开发，随后还需另一批更晚的全新
operator-blind cohort 一次性确认。完整分析见
`docs/datasets/kinofail_o4_v31_failure_analysis.md`。这仍不计 A0–A7，readiness 保持 **0/8**。

### 2026-07-23：v27 phase-conditioned O4 开发封存

针对 v26 的 3/6 跨场景失败，新版 v27 只在三个已经暴露的 v16 development scene 上改动动作
架构，不修改 O4 物理、Go2 policy、route controller、速度、force cap、peel 阈值或 0.15 m
成功门。决策从固定 `first attachment + 25 steps` 改为有界触发：至少等待 5 步；若
`tilt >= 0.08 rad` 且相对近期站立高度下降 `>= 0.02 m` 连续 2 步，则提前分叉；否则最多等待
25 步。Backstep 从固定 75 步改为闭环 clearance：最多 120 步，目标后退 0.20 m，并要求活跃
adhesion 连续清零 5 步后退出，正式验收距离仍保持 0.15 m。

冻结前检绑定了 collector、物理后端、policy、三条场景栈和空输出目录；LivingRoom45、
Kitchen46、Bathroom49 随后各在独立 Isaac 进程中只执行一次。封存结果为 **3/3**，每场的
Continue/Backstep 在决策前 telemetry 逐行相等，Continue 均在分叉后跌倒，Backstep 均完成
300-step horizon、物理解黏且不跌倒。后退距离分别为 0.2464、0.2613、0.2495 m，post-decision
peel 为 3、7、4 次，最大路线偏差均不超过 0.30 m。postrun 位于
`outputs/kinofail_realistic/operator_development/o4_action_v27_phase_conditioned/postrun_audit.json`
（SHA-256 `8b7670c83ab918140832d4780759c513470279b77de25eddd28af2c0e98ba952`），
`integrity_passed=true`、`outcome_passed=true`、`sealed=true`。

这批结果仍有明确边界：三场都在最大 25 步等待处触发，未实际覆盖紧急提前触发分支；且三场
均是已暴露 development scene。因此 v27 只证明闭环 clearance 在这组三场上可用，**不等于
held-out O4 confirmation，更不计 A0–A7**。下一门必须先用 nominal-only QA 冻结一批从未执行
O4 的新 realistic scene，再把完全不变的 v27 合同一次性运行到全 cohort；不得回到 v26 或
本批三场追加试验。readiness 继续为 **0/8**。

### 2026-07-23：v25 realistic cohort 封存与 v26 O4 未见场景确认

本轮继续执行“先构建新版 realistic Kino-Fail，再在其上重做 A0–A7”的主线。v25 在任何 O4
结果打开前，冻结了 24 个 EmbodiedGen simple-room 请求和严格前缀停止规则：至少 5 个完整
admitted stack、至少 4 个 room family；每个候选只生成一次，禁止修复、替换 seed 或跳过失败。
最终在第 16 个候选首次满足门槛并立即停流，共得到 **6 个 admitted scene / 4 个 family**：
`indoor_office_72`、`indoor_livingroom_73`、`indoor_kitchen_74`、`indoor_office_78`、
`indoor_office_84`、`indoor_house_87`，覆盖 Office、LivingRoom、Kitchen、House。第 17–24 个
后缀保持未触碰。16 个处理候选的首个终态为 6 admitted、3 base-compile rejection、2 static-RTX
rejection、2 motion-proxy rejection、1 corridor rejection、1 source-preflight rejection 和
1 generation exception；所有失败后的下游 artifact 均不存在。DiningRoom 第 12 候选的
generation exception 是启动封装器时漏传仓库 `PYTHONPATH` 的执行错误，不代表 DiningRoom
场景本身质量；由于一次尝试合同已写出正式异常审计，该候选未被重跑或替换。

v25 postrun 位于
`outputs/kinofail_realistic/operator_confirmation/o4_action_v25_scene_stream/postrun_audit.json`
（SHA-256 `e99a08ec07e8548e75c241f8b5b2b1ae1fa07b64b946498de347c4e5e31b5c8a`），
`passed=true`、`sealed=true`。它只冻结可供实验的 realistic cohort，**不计 A0–A7 证据**。

随后 v26 将 train-only 已通过的 v21 Continue/Backstep 参数原样绑定到上述全部 6 场：相同 scene、
seed、operator 和分叉前 telemetry，首次足端 attachment 后固定 dwell 25 步，再令 Continue 保持
0.32 m/s 前进，Backstep 以 0.24 m/s 后退 75 步并保持；恢复距离门仍为 0.15 m。场景、材质和
运行 seed 在 O4 outcome 前冻结，批处理器无论先前结果正负都执行全部场景。独立 postrun 证明
6/6 case 的执行与 artifact 完整性通过，但主结果仅 **3/6**：

- Office72、LivingRoom73、Kitchen74 全门通过；Backstep 分别后退 0.2533/0.2583/0.3014 m，
  均有 4 次 post-decision peel、未跌倒，Continue 均在分叉后跌倒。
- Office78、Office84 的 Backstep 分别在 step 121/124 跌倒，0 次 peel、0 m 后退；分叉前
  telemetry 仍逐行相同，且 Continue 均按预期失败。这是真实的跨场景反例，不是对照不匹配。
- House87 的 Backstep 保持站立、完成 300 steps 并产生 3 次 peel，但仅后退 0.1294 m，未达到
  未改动的 0.15 m 门，因此保持失败。

v26 postrun 位于
`outputs/kinofail_realistic/operator_confirmation/o4_action_v26_heldout_confirmation/postrun_audit.json`
（SHA-256 `ed66bdbdc375a04abf3a3f15c2d08b7e661f6344b595a2035d450379d28fc3dd`），
`integrity_passed=true`、`sealed=true`、`passed=false`。当前最可能的架构缺口是固定
“首次 attachment + 25 steps”没有控制分叉时的支撑相位/可剥离状态：两个失败 Office 在分叉时
只形成 2 次 attachment，并在产生 peel 前翻倒；通过场景形成 4 次 attachment。v26 场景不得再
用于调参。下一版只能在独立 development scenes 上设计由实测支撑足、接触卸载裕度和稳定窗口
共同触发的 phase-conditioned recovery branch，然后重新生成全新 held-out cohort；同时推进
O5–O11 的 realistic confirmation，不能让 O4 迭代替代完整 corpus 与 A0–A7 重跑。当前
realistic A0–A7 readiness 仍为 **0/8**，A8 仍永久排除。

### 2026-07-22：v16 realistic 未见场景栈与 O1–O4 重新校准

本轮工作的验收对象已经明确改为：**在新版 realistic Kino-Fail 上重新采集、训练并复现
A0–A7，而不是把旧 A0–A7 整理完就视为完成**。旧 controlled-core 结果只作为效应方向、协议
设计和判据开发的参考；禁止与新版 realistic 指标混算。A8 不在范围内。当前 realistic
A0–A7 readiness 仍为 **0/8**。

为此冻结并执行了
`kinofail_embodiedgen_realistic_route_policy_v16_unseen_batch_v1`。六个预注册且不替换的全新
EmbodiedGen 请求中，LivingRoom、Kitchen、Bathroom 三个 scene stack 通过从 source geometry、
collision/route、PBR route surface、RTX/Go2-front 到 nominal locomotion 的完整准入；Bedroom
因源床体 bounds 下探到约 −3.6 km 被拒，Office 无无碰撞 route endpoint，DiningRoom 的三个
近地/机身俯仰视角过暗。最终准入率为 **3/6**，失败全部保留。三条 admitted nominal 在
650-step 冻结 runtime 中均前进约 1.56 m、最大横向偏差 1.83–2.13 cm、未跌倒。封存审计为
`outputs/kinofail_realistic/operator_confirmation/embodiedgen_realistic_route_policy_v16_unseen_batch_v1/postrun_audit.json`。

在这三个 admitted stack 上，O1/O2/O3 采用相同场景、路线、Go2 机身固定相机和随机 PBR
材质完成一次性重新校准：

- O1 nominal/anomaly 共 6 组 pair 的 12 条 lane 全部通过。异常只改变中心隐藏碰撞分区摩擦至
  μs/μd=0.04/0.03；三场 anomaly 的 mean slip 为 0.645–0.683，均跌倒，nominal 均完成且
  不跌倒。
- O2 anomaly 在三场均通过：最大实际足端沉陷 8.88–8.89 cm、剪切功 6.63–9.72 J、至少两个
  承重足端，且连续 PBR 表面只在接触后形成物理足迹。对应 nominal 均为零沉陷、零剪切且不跌倒。
- O3 nominal/anomaly 共 6 组 pair 的 12 条 lane 全部通过。nominal 使用不可达阈值且不坍塌；
  anomaly 在约 25.8–27.0 N·s 的实测冲量后同步关闭 13 个 collider 并变形连续视觉表面，三场
  均跌倒，视觉与物理 trigger step 一致。

O1/O3 成对后审计为
`outputs/kinofail_realistic/operator_recalibration/o1_o3_v16_admitted_v1/postrun_audit.json`
（6/6 pair，`sealed=true`）；O2 后审计为
`outputs/kinofail_realistic/operator_recalibration/o2_continuous_support_v16_admitted_v1/postrun_audit.json`
（3/3 case，`sealed=true`）。这些结果证明 O1–O3 物理干预可以进入真实室内 stack，仍然只是
operator recalibration，均写死 `counts_as_a0_a7_evidence=false`。

O4 随后使用相同三个 admitted stack、35 N 足端 world-anchor adhesion 和冻结的单一
recoverable-peel 合同进行一次性检验。主目标结果为诚实的 **0/3**，审计完整性通过并已封存：
三场都在约 1.0 s 后由命名足端附着、读回 35 N、产生显著姿态/高度后果并最终跌倒，但都在
冻结的 5 s forward phase 结束前终止，因而没有进入 peel/release phase；两场跌倒后的横向位置
还超过 0.3 m route budget。结果位于
`outputs/kinofail_realistic/operator_recalibration/o4_v16_admitted_v1/postrun_audit.json`，其中
`accepted_pair_count=0/3`、`mechanism_diagnostic_pass_count=3/3`、`primary_target_passed=false`。
不得把 3/3 机制生效改写成 O4 pair 通过。

该失败暴露的是 endpoint 设计错误，而不是继续挑阈值即可解决的问题。下一版 O4 必须先在
train-only 场景校准，再于任何 held-out 运行前冻结两个不同 endpoint：moderate/recoverable
用于 A4 的实际恢复后果，severe/terminal 用于 A0/A3 的异常归因与失效检测。两类可以共享足端
adhesion 物理，但不能共享“必须 peel 且不提前跌倒”的成功定义。完成 O4 修复后，按同一
realistic paired-runtime 合同覆盖 O5–O11；之后才是 registry-bound corpus、A0 数据证书、
新版模型训练/推理和 A1–A7 正式重跑。

随后已按上述规则在明确降级为 train-only 的 LivingRoom 上冻结并完整运行 8/12/16 N 三档
moderate force-cap 搜索；未在 35 N 的 0/3 结果上补跑。三档全部未选中：首次附着均为 step 50，
跌倒分别发生于 step 150/144/135，均未到达 peel phase；倾角增量为 0.769/0.831/0.784 rad，
最低机身高度下降 17.14/17.43/17.53 cm。8 N 虽保持在 0.3 m route budget 内，仍提前跌倒；
12/16 N 同时越出预算。更关键的是，终止前 `progress_gain_lag_m` 分别为
−1.84/−5.05/−1.41 cm，证明旧 locomotion-lag 指标会把翻倒瞬态误读成“异常更快”，不适合
承担 terminal O4 的效应门。封存审计为
`outputs/kinofail_realistic/operator_recalibration/o4_v17_train_only/postrun_audit.json`，其中
`selected_moderate_force_cap_n=null`、`selection_passed=false`、`sealed=true`。

逐步遥测定位到物理状态模型的具体错误：当前实现把单个足端固定在 world anchor 上，即使该足
已经卸载并进入摆动相也不会按接触状态/法向剥离释放，只能等全局反向命令或 105 N raw break；
8 N lane 的 raw spring force 已累积到 43.89 N，持续单足偏置力矩最终将机器人横向拉倒。因此
force-cap 搜索不能修复构念。v18 必须把 O4 拆为两个物理子型：`adhesive_contact` 使用承重状态、
法向/切向各向异性 peel 与可重复 stance attachment cycle；`tether_entanglement` 才允许持续
world anchor，并必须在 RGB 中存在可见 cable/tether 且采用 terminal endpoint。该修改先进入
新文件/新后端适配器，不能改写已封存 v16/v17 代码哈希。

v18 已按上述设计新增 `adhesion_v2.py` 与独立 Isaac 后端子类，没有修改旧 backend/model；新旧
模型合计 16 个单元测试全部通过。一次性 train-only Isaac 结果修复了永久锚定：12 N 切向、
6 N 法向 cap 下 anomaly 完成 430 步、不跌倒、最大横向偏差 3.59 cm，共出现 23 次 attachment
和 23 次 contact-aware peel，切向耗散 5.16 J。该结果仍保持失败，因为冻结效应门没有达到：
进度滞后 3.14 cm、倾角增加 0.037 rad、最低高度下降 1.03 cm，低于 8 cm 且
0.10 rad/3 cm 的双通道门。它只证明安全的状态架构成立，不能写成 O4 已重新校准完成。

随后预声明的 v19 train-only 强度校准只测试 18/24 N 两档，但同时固定采用更高 normal cap 与
更长 peel height/dwell。两档均保留 attach/peel 事件且横向偏差只有 4.49/5.30 cm，却都在
step 167 跌倒；峰值总 adhesion force 为 24.55/33.65 N，切向耗散为 4.21/5.31 J。由于 v18→v19
不只改变 force cap，禁止把相同跌倒时刻解释成单因素剂量关系；两档均未选中。封存审计为
`outputs/kinofail_realistic/operator_recalibration/o4_surface_v19_train_only/postrun_audit.json`，
其中 `selected_tangential_force_cap_n=null`、`selection_passed=false`、`sealed=true`。

下一步不再做无目标的 cap 搜索。v20 应以 v19 的约 2.34 s 终止窗口构建 action-conditioned
后果实验：两条 lane 都启用同一 `adhesive_contact`，在首次 attachment 后冻结的决策时刻之前
要求完整 telemetry/hash 一致，之后才分叉 `Continue` 与 `Backstep`。合格结果必须是 Backstep
触发及时 peel、保持站立并退出区域，而 Continue 出现冻结定义的 terminal/immobilized 失败；
这才是 A4 所需的“正确归因改变真实恢复后果”。v20 仍先在 train-only 场景开发，通过后再生成
全新 held-out scene，不能回收 v16–v19 的任何失败。

本轮已经把 terrain 外观随机化从“计划”推进为可复现的资产、PBR 绑定、真实 PhysX
垂直切片、O2/O3 物理状态—视觉状态同步，以及 O2/O3/O4/O5/O6/O9/O10 四档配对工程
剂量门，并修正了这些算子的主要物理替身/测量问题。terrain texture-swap 已进入
design-v3/runtime-v5：每条物理轨迹在同一状态和时间戳下渲染 3 个 PBR 外观，物理、
proprio 和 telemetry 不变。O4 采集链已从非正式 smoke 推进到首个冻结 formal subset：
`indoor_workstudio_01` 中 moderate/severe 两个 nominal/anomaly 反事实组共 4 条 physical
episode、12 条同步 appearance-view sequence，全部通过 runtime-v5 并获得
`artifact_state=validated`、`evaluation_eligible=true`。当前正式状态为
`planned=392`、`validated=4`、`evaluation_eligible=4/396`；历史 smoke 只保留为工程记录，
不与这 4 条正式重采样重复计数。**Kino-Fail realistic v1 仍未完成，publication freeze 仍为
false**，不能把首个 1.01% 子集外推为整个 benchmark 已完成。

2026-07-22 新增的 wild/O2 photographic vertical slice 把“terrain 换贴图”推进为三套冻结的
完整场景外观族。Monk's Forest、Whipple Creek 与 Hochsal Forest 使用三个不同 4K CC0 HDRI、
至少两种路线 PBR 和两种周边 PBR，其中 Hochsal 的周边材质在模型评估前冻结为 `val`。三套
rendered pre-entry 图像哈希不同，且 contact-local PBR 压痕只在实测足端承重后生成。但是汇总
审计 `forest_o2_three_appearance_families_v1/audit.json` 现已被标为 **superseded，不再承担
nominal/anomaly 因果证据**：复核 USD 后发现原场景 nominal floor 为 μs/μd=0.8/0.6，而当时
anomaly 的隐藏分区周边被写成 0.8/0.8。两条 lane 同时改变了 O2 与区域外接触材质，因此不能
把旧的 10.318 cm 机身高度差、8.921 cm 足端沉陷、15.096 J 剪切耗散或跌倒结局归因于 O2
单一因素。冻结外观资产、不同图像哈希和接触局部压痕实现仍保留为开发记录。

随后完成的 Monk's Forest 修正 pair 在两条 lane 中都使用相同的 4 周边 + 3 中心分段；nominal
中心保持平坦且全部使用场景 0.8/0.6，不施加土壤力，anomaly 才启用下沉坡面、软床、剪切
耗散和接触局部压痕。前 125 步完整遥测逐行相同，第 126 步首次区域承重接触且首次分歧。
nominal 完成路线，anomaly 最大足端沉陷 8.94 cm、剪切耗散 26.01 J、3 个足端承重并跌倒；
中位机身高度差 10.57 cm、前进亏损 1.1140 m。审计位于
`forest_hybrid_monks_dev01/isaac_hybrid_composition_dev_v14/o2_matched_topology_pair_dev_v14_attempt2/pair_audit.json`。

同日已把 O1 从彩色/抬高薄板迁移到连续森林地表：原 PBR mesh 平坦可见，下面是不可见的
4 周边 + 1 中心共面碰撞分区。旧三外观汇总审计 `forest_o1_three_appearance_families_v1/audit.json`
同样已 **superseded**，因为当时 nominal 使用 0.8/0.6 单 floor，而 anomaly 周边错误使用
0.8/0.8；旧的 +0.7660 slip、−1.3511 m 进度和 +0.2017 m 横向偏移不再作为因果效应量，只
保留三种 photographic/PBR 外观确实不同的资产事实。

修正后，O1 nominal 与 anomaly 都安装完全相同的五分区隐藏碰撞拓扑；nominal 中心和周边均
直接读取场景 μs/μd=0.8/0.6，anomaly 仅将中心改为 0.04/0.03。Monk's Forest fresh-process
pair 的前 125 步完整遥测逐行相同；第 126 步首次测到区域承重足端接触，也正是首次动力学
分歧。nominal 完成路线且不跌倒，anomaly 跌倒；区域平均 slip 增量 0.8318、前进亏损
1.3525 m、横向偏移增量 0.2006 m。严格审计位于
`forest_hybrid_monks_dev01/isaac_hybrid_composition_dev_v14/o1_matched_topology_pair_dev_v14_attempt3/pair_audit.json`。

O3 也已采用严格匹配对照：两条 lane 都使用同一个 4×4、零间隙、不可见、nominal-friction
支撑网格；nominal 只把损伤阈值设为不可达，anomaly 使用 25 N·s 实测足端法向冲量。最终
attempt6 的前 141 行遥测逐字段完全相同，第 142 步首次分歧且正好是触发捕获；同一步关闭
13 个支撑 collider 并使连续 PBR 网格的 293 个顶点发生不规则下陷，最深 −0.1141 m。
nominal 完成路线，anomaly 跌倒，前进亏损 1.4571 m、中位区域机身高度差 0.1623 m。审计位于
`forest_hybrid_monks_dev01/isaac_hybrid_composition_dev_v14/o3_triggered_pair_dev_v14_attempt6/pair_audit.json`。
机身固定前置相机在触发瞬间无法直接看到足下凹坑，因此这里证明的是物理—网格状态同步，
不是异常外观必然在 RGB 中可观测。

修正协议随后已在 Monk's Forest、Whipple Creek、Hochsal Forest 三套冻结 photographic/PBR
外观上完整重跑 O1/O2/O3。三项汇总审计均通过：三个 panorama 哈希和 pre-entry RTX 哈希互异，
包含 2 个 train 外观与 1 个 held-out validation 材质；每个外观内的严格 nominal/anomaly pair
均通过，且 O1/O2/O3 的接触前或触发前反事实边界保持成立。对应审计为
`forest_o1_matched_three_appearance_families_v2/audit.json`、
`forest_o2_matched_three_appearance_families_v2/audit.json` 与
`forest_o3_matched_three_appearance_families_v1/audit.json`。

这关闭的是**三外观开发门，不是三物理场景复现门**。逐文件复核显示三套编译场景的
`collision.usda`、`prop_collision.usda`、`route.usda` 哈希完全相同；它们共享同一 metric route、
碰撞布局、operator 坐标、控制命令、seed 和动力学轨迹。换 HDRI/PBR 后得到相同结果能证明外观
干预没有改变物理后果，但不能增加独立 physical episode/scene realization 样本量。由于当前
domain randomization 关闭，简单换 seed 也只会确定性复制同一轨迹，因此禁止以重复运行冒充
multi-seed 统计。O1/O2/O3 仍须补不同 route/collision 哈希、不同进入相位/轨迹以及不同
substrate realization，再进入 registry/formal schedule。三项均保持
`scene_registry_eligible=false`、`counts_as_a0_a7_evidence=false`；必须在新版 realistic corpus
上重新训练、预测和统计 A0–A7，当前 readiness 仍为 **0/8**。

随后已补上第一轮真正不同的物理几何，而不是继续复制 appearance：G02/v19 与 G03/v20 分别
改变 route 中心线、长度、宽度、endpoint margin、刚性道具位姿和碰撞代理，并冻结不同的
operator 位置、入射横向偏置、初始航向、速度与 seed。G01/G02/G03 的 `collision.usda`、
`prop_collision.usda`、`route.usda` 哈希逐项两两不同，最小 route clearance 分别为
0.769/1.121/0.926 m。注意顶层 `episode.usda` 只是相同的相对 sublayer 清单，可能字节哈希相同；
几何身份必须由 compiled-audit 与三个解析后物理子层哈希判定。

G02 明确作为 development calibration，暴露并修正了两个过窄审计定义：O1 的横向偏移不应是
低摩擦构念的必要主终点；O2 的 lowered-support 因果边界应取任一 lane 首次与处理分区承重交互
的最早时刻，而不能只等 anomaly 足端在下陷后重新接触。两项定义均在 G03 运行前版本化并冻结，
G03 只运行一次 held-out geometry confirmation，且 O1/O2/O3 三项均通过：

- O1：前 199 行完全相同，step 200 首次承重接触且首次分歧；mean slip +0.7663、route progress
  −1.0876 m，nominal 完成、anomaly 跌倒。
- O2：前 165 行完全相同，step 166 首次表面交互，step 179 首次分歧；最大沉陷 9.44 cm、剪切
  36.64 J、中位机身高度差 13.19 cm、route progress −1.1679 m，nominal 完成、anomaly 跌倒。
- O3：触发前 239 行完全相同，step 240 首次分歧与事件截图一致；26.22 N·s 关闭 13 个支撑
  collider，同步变形 309 个 PBR 顶点，最低偏移 −11.25 cm；高度差 15.01 cm、route progress
  −1.5006 m，nominal 完成、anomaly 跌倒。

统一汇总 `forest_metric_geometry_o1_o3_summary_v1/audit.json` 为 PASS，并确认三项 held-out seed
和 trajectory contract 互异。但它只关闭 **O1–O3 首个独立 metric-geometry 开发门**：G02/G03
仍复用同一个 forest visual-shell 来源，且每个算子只有一个 held-out 物理几何，没有跨 train/
val/test 的多个独立 scene family、跨几何剂量响应、实物标定、registry/formal episode，更没有
realistic A0–A7 的重训、prediction 与统计。因此该汇总继续写死
`counts_as_a0_a7_evidence=false`，readiness 仍为 **0/8**。

在此基础上又执行了两层不替换失败样本的冻结复现。第一层 G04/G05 将 Whipple/Hochsal 两个既有
外观族与两套新 route/prop-collision 几何交叉；strict-v2 为 **4/6**：G04-O2 的物理后果强，
但视觉最低偏移 −6.75 cm 略低于冻结 −7 cm；G05-O1 的滑移增量 0.408 低于冻结 0.50，虽然
前进亏损 1.494 m、横向偏移变化 0.145 m 且异常跌倒。两项失败均保留，没有重跑。由此在
G04/G05 校准集上版本化 O1 phase-robust 与 O2 sinkage-normalized 的候选 v3；校准 6/6 不作
确认结论。

第二层在任何异常运行前，冻结两个从未用于 G01–G05 算子实验的 photographic family 与两套新
物理几何：G06/Forest Slope + `test_ground073`，G07/River Walk 1 + `test_gravel022`。预审计证明
G01–G07 的 `collision.usda`、`prop_collision.usda`、`route.usda` 哈希逐类全部互异，且
G06/G07 的 appearance/background 哈希未出现在 G01–G05；两条 nominal route 均一次通过。随后
按一场景一算子一尝试、禁止改阈值/时长/换场的合同完成 O1–O3 六组 pair。总审计位于
`forest_unseen_scene_operator_v3_confirmation_dev_v1/postrun_audit.json`：

- O1：G06/G07 均通过候选 v3，且均通过更严 strict-v2；slip 增量 0.729/0.796，route-progress
  deficit 1.409/1.349 m，异常均跌倒。
- O2：G06/G07 均通过 strict-v2 与 v3；物理沉陷 8.931/8.930 cm，视觉深度
  8.776/8.628 cm，visual/sinkage 比 0.983/0.966，异常均跌倒。
- O3：G07 全项通过；G06 在 25.62 N·s 后关闭 13 个支撑单元、变形 314 顶点至 −11.34 cm，
  高度差 27.55 cm、进度亏损 1.631 m，但冻结的 580 步内未被二元 fall detector 判跌倒，故
  该 pair 保持失败。strict-v2 和全六格结果均为 **5/6**，而 v3 实际修改所针对的 O1/O2 为
  **4/4 unseen-scene confirmation PASS**；未修改的 O3 为 **1/2**。

这批结果关闭的是 O1/O2 候选判据的未见场景确认，并暴露了 O3 “强陷落/失能但未触发 fall”
的终点定义问题；它仍是 development-only operator-architecture evidence，不能进入 A0 分母。
尤其不能把 G06/G07 的 5/6 写成新版 Kino-Fail 已复现 A0–A7。下一步必须把 route-centric
O1–O3 接到 EmbodiedGen 室内场景，并扩展 production/wild 的实体 registry；只有
registry-bound formal schedule 完成后，才开始新版数据上的训练、prediction 与 A0–A7 统计。

跨域适配已经开始实际落地，而不是只留作下一步文字。首个对象是已通过 source/RTX/Go2/
route-surface-v7 admission 的 `indoor_kitchen_31`。集成过程保留了四次开发失败：原始 EmbodiedGen
floor 缺少显式 PhysX material；只补 0.8/0.6 后，四顶点 route surface 仍不能承担局部形变；补成
2635 顶点/2520 quad、最大间距 4 cm 的无碰撞连续 PBR mesh 后，错误从 waypoint 0 起跑又未进入
operator 区。最终适配器强制读取既有 admission 冻结的 `collector_start_waypoint_index=2`，而不是
根据失败轨迹选起点。第五次 nominal calibration 才全项通过：完成 1.561 m、未跌倒、进入区域
76 个采样步，五分区 matched topology、单一 floor authority、μs/μd=0.8/0.6 运行时读回、隐藏
collision、连续 PBR 和 body-fixed RTX 三事件帧均通过。完整审计为
`terrain_adapter_v3_dev/calibration_audit.json`。

该里程碑证明 route-centric O1–O3 架构能接入真实 EmbodiedGen 室内场景，但仍**不进入 A0**：
成功 nominal 是四次失败后的 calibration；当前采集器还是对冻结 forest collector 的可追溯
development bridge，manifest 明确 `formal_corpus_eligible=false`；尚未运行 indoor anomaly pair，
更没有两个 operator-naive scene family confirmation。正式晋级顺序固定为：native v3 collector
→ generalized pair auditors → 新 calibration/confirmation 场景在 anomaly 前冻结 → operator
capability registry → registry-bound formal schedule → 396 条 pilot → 新数据上的 A0–A7。

同时，新的 EmbodiedGen 场景物理验证链已经形成与上述 runtime-v5 corpus 分离的 O4
confirmatory evidence。Bedroom 与 LivingRoom 仅用于冻结 route-relative O4 合同；随后在从未
运行过 O4 的 Kitchen（独立 RoomGen seed/family）上预注册并只运行一次 nominal/O4 pair，正式
结果通过。两组均完成 362 步、各 38 帧且无跌倒；O4 从第 41 步附着、第 107 步反向剥离，配对
前进窗口中进度滞后 10.18 cm、相对速度抑制 15.68%、最大倾角增加 0.330 rad、最低机身高度
下降 6.06 cm。独立 admission audit 重新验证 7 个冻结文件、协议/seed/scene 绑定、标定场景排除
和 RGB/telemetry 轨迹去重，全部通过。该 pair 尚未进入 396 条 runtime-v5 schedule，也没有
同状态 3 贴图，因此只支撑“跨场景 O4 物理后果”，不增加 `evaluation_eligible=4/396` 的计数。

Kitchen 之后的首个三-family 确认批次已经封存，不能再描述为“待运行”：在生成前冻结的
DiningRoom 与 House 两个新增请求均在 O4 注入前被拒绝，连同 Kitchen parent 计为
`formal_passed=1/3`。DiningRoom 的 nominal Go2 虽完成 0.502 m 且未跌倒，但最大横向漂移
0.252 m、最大倾角 0.709 rad，未通过稳定性门；House 的四视角均成功渲染，但旧的全局照明
使路线中段/Go2 高度视角只有 42/21 个 5-bit 颜色，未通过 appearance-complexity 门。批次
审计逐项重算生成请求、源包、compile、RTX/Go2、拒绝记录和冻结代码哈希，结果
`sealed=true`、`audit_integrity_passed=true`；这两个负结果保留在分母中，未换 seed、未改阈值、
未执行正式 O4。

上述失败催生了版本化的 robust-corridor v2，而没有改动已封存的 v1 文件：规划器显式预留
0.34 m 机器人半径 + 0.10 m 物理裕量 + 0.30 m 闭环跟踪误差，即要求中心线净空至少
0.74 m，并沿 2.45 m 直线实验走廊布置 5 个顶灯。开发场景 House 的最终 v2 走廊实测净空
1.073 m；RTX Go2-height 颜色数由 21 提升到 541、黑像素由 68.9% 降至 11.0%，articulated
Go2 前进 0.504 m、最大横向漂移 1.36 cm，19 项独立准入检查全通过。DiningRoom 在 0.74 m
合同下被编译器提前结构化拒绝，继续作为开发排除证据。

随后冻结的 Bathroom/Office/Bedroom corridor-v2 扩展批次已经完整封存，而不是仍在采集。
Bathroom 在基础路线门拒收；Office 与 Bedroom 均通过 source/compile/RTX/Go2/19 项场景准入，
但两个唯一正式 O4 pair 的前进进度滞后分别只有 9.54/9.60 cm，略低于预注册 10 cm 门，尽管
相对速度抑制均约 15%、姿态通道均通过且均按顺序在剥离后跌倒。因此该批次诚实记为
`formal_passed=0/3`，审计为 `sealed=true`、`audit_integrity_passed=true`，没有通过降阈值或
加力改写结果。

针对这一单一边界问题，v5 仅把同一 35 N O4 的前进观察窗从 2.14 s 延长到 2.40 s；修改在
执行前冻结，并且只允许 Office/Bedroom 各一个开发校准。两条开发 pair 均通过：进度滞后
11.64/11.61 cm、相对速度抑制 15.23%/15.18%，开发审计据此只作出“允许新 held-out 批次”的
结论。随后在任何目标生成前冻结 Kitchen/LivingRoom/DiningRoom 三场景 v5 确认批次，且每个
场景只允许一次生成、一次正式运行、不得替换。最终审计仍为完整但主目标失败：Kitchen 的 O4
运动/姿态后果全部通过（滞后 11.50 cm、相对抑制 15.03%、倾角增加 0.415 rad、高度下降
9.19 cm），却因深色地面在机身前倾后造成 O4 序列有效帧仅 68.75% 而整对失败；LivingRoom
正式 pair 与独立性审计全项通过；DiningRoom 在 O4 前的 Go2 门因最大倾角 0.678 rad 和末帧
退化而拒收。故批次为 `formal_passed=1/3`、`primary_target_met=false`、`sealed=true`、
`audit_integrity_passed=true`。这说明 2.40 s 窗口解决了已准入场景中的 O4 力学可观测性，
但当前场景/照明准入仍不足以承担高通过率采集；不能据此声称 O4 已完成跨场景确认。

此后没有把 Kitchen 的暗场物理正例事后改写为确认结果，而是先在 4 个正例和 2 个物理负例上
冻结 phase-aware v6：首次附着前的 context 帧必须信息充分，全部采集帧仍须达到传感有效性门，
注册的 terminal fall 帧必须与首个跌倒步一致；所有旧的非视觉因果检查保持不变。开发审计为
4/4 正例与 2/2 负例判别正确。首个 v6 确认批次随后因沿用只接受旧 pair 总判定的 independence
auditor 而发生协议不兼容，完整 5 场被行政作废为 0/5 admitted，不能回收其中 LivingRoom 的
物理+v6 正例。修正只发生在新版本的 v6-native independence auditor 中，并在下一批场景生成前
冻结；它要求 formal v6 审计的路径/哈希、协议、seed、scene、代码以及独立 physics/RGB 全部绑定。

最新的 6 场 v6-native 确认批次现已封存：`sealed=true`、`audit_integrity_passed=true`，但只有
`formal_v6_passes=2/6`，低于预注册的 `>=3/6`，所以 `primary_target_met=false`。LivingRoom
seed 20260827 与 Kitchen seed 20260837 是两个跨房型、跨 seed、与 Office/Bedroom 校准数据在
physics/RGB 上均去重的正式正例；其进度滞后为 11.91/12.12 cm、相对速度抑制为
15.33%/15.88%、倾角增加为 0.439/0.332 rad、机身高度下降为 10.38/9.40 cm，39+16 帧/场
全部通过 phase-aware v6。其余四场均在 O4 前停止：两个 LivingRoom 分别因入口视角近黑和反向
视角全黑被 RTX 门拒绝；House 含约 `1e23 m` 的 WindowFactory 异常平移；DiningRoom 只有
5 个 mesh/520 个面，触发 `insufficient_visual_faces`。因此现有证据支持“35 N O4 在两个真正
准入的新室内房型中可重复产生多通道后果”，但也直接否定当前生成/照明管线已具备 benchmark
规模高产率；它仍不能支撑 Kino-Fail 或 O4 已达到 ICRA benchmark publication-ready。

针对这四个前置拒绝，后续工作保持为**开发证据**，没有回收或重标上述 6 场确认结果。旧
lighting-v4 矩阵在首格后被行政作废：诊断证明 Isaac Camera 默认近裁剪面约为 1 m，而 Go2
pitch45/70/80 视角到地面的距离只有约 0.59/0.45/0.43 m，因此所谓 pitch 黑帧主要是近裁剪
失配，不是提高 Dome/RectLight 强度能够解决的问题。新的 v5 场景视觉合同把 near clip 固定为
0.01 m，并与 O4 phase-aware 判据对齐：4 个 canonical 视角使用严格 context 门，3 个 body-pitch
视角使用仍要求非平坦、非全黑/全白的 sensor-valid 门。保持 1.2 m、无 collision、`purpose=render`
的路线 PBR surface 与 7 个路线包络灯不变后，冻结的 7 场×3 材质 v5 开发矩阵完整运行并封存为
`13/21`、`matrix_passed=false`、`audit_integrity_passed=true`。其中 `train_concrete034` 因
pitch70/80 局部对比不足为 0/7；`train_tiles141` 为 6/7，唯一失败是 LivingRoom17 reverse
视角只有 50 个 5-bit 颜色（门槛 64）；`train_road007` 为 7/7。失败格全部保留，没有降阈值。

源资产侧新增了在正式 compile/RTX/Go2 之前运行的 source-geometry preflight v1。它绑定 source
manifest 与主 USD 哈希，并复用现有编译器的 100 m 坐标、5 meshes、1000 faces、Z-up/米制与
metric floor 门。三点校准审计 `validation_passed=true`：House33 的 41/41 meshes 均含约
`1e23 m` 异常 world transform，primary gate 为 `implausible_world_transform`；Dining35 为
5 meshes/520 faces，primary gate 为 `insufficient_visual_faces`；Kitchen37 控制为
6 meshes/13244 faces 并通过。该 preflight 只提前拒绝必败源，不能替代 route、RTX 或 Go2 门。

材质侧先在 train split 做宽松静态 basecolor 预筛，明确不读取 val/test；`concrete034` 的动态
范围仅 0.106、5-bit 颜色仅 16，四项静态内容门全部失败。静态筛选仍不能替代 RTX：
`concrete046` 虽通过预筛，却在排除于矩阵分母的 Kitchen11 pilot 中再次因 pitch70/80 std
约 0.00380/0.00404 失败。`ground054` 与 `ground037` pilot 通过后，才在任何目标格运行前冻结
六个未参与候选选择的场景×两材质 v6 开发矩阵。最终 12/12 全部通过，且
`sealed=true`、`audit_integrity_passed=true`、`matrix_passed=true`。跨六场最坏值为：
`ground054` pitch std 0.01069、pitch colors 46、canonical colors 138；`ground037` 分别为
0.02538、261、427，均高于 0.005/8/64 门。结合相同代码下 `road007` 的 v5 7/7，当前只允许把
`road007 + ground054 + ground037` 称为**训练域的稳健默认候选池**；v6 矩阵仍是开发证据，
必须在全新 EmbodiedGen scene family 和未见 val/test 材质上做冻结确认，才能支撑高产率主张。

该确认现已完成，而不是仍停留在计划阶段。`route-surface-v7` 在任何新场景生成或 held-out
RTX 检查前冻结 12 个 request：6 个稳定 RoomGen 房型各 2 个实例，simple/minimalist 与
val/test 各半，12 个 val/test 材质各使用一次；每个 request 只允许一次源生成、不得换 seed、
换材质、修源或改门槛。最终官方审计为 `sealed=true`、`audit_integrity_passed=true`，但正式
独立通过只有 **4/12**（val 2、test 2），预注册 `>=8/12` 主目标以及 split、房型、材质覆盖
子目标全部失败。逐门产率为：source 12/12、source preflight 11/12、base route 9/12、
corridor 9/12、RTX 7/12、Go2 6/12、v7 admission 6/12、phase-aware v6 6/12、family-level
independence 4/12。四个最终正例为 Kitchen25/`val_ground071`、DiningRoom26/`val_gravel023`、
LivingRoom30/`test_asphalt031`、Kitchen31/`test_ground073`，覆盖 3 个房型和 4 种未见材质。

八个失败均保留在分母并有首个失败门：Bathroom29 只有 4 meshes/288 faces；Bathroom23 无
collision-free route endpoint；DiningRoom32 的最长 Go2-width 路径只有 1.813 m（门槛 4 m）；
Office21 的 pitch 帧最少仅 5 个 5-bit 颜色；LivingRoom24 的 pitch 帧最低 std 0.00068、颜色
数 1；Bedroom22 在 Go2 QA 中只前进 0.434 m 并倾覆（max tilt 0.816 rad）；Office27 与
Bedroom28 虽完成 pair 且 phase-aware v6 通过，但与 Office/Bedroom calibration family 重合，
被冻结的 family-level independence 合同拒绝。六个实际采集 pair 的 phase-aware v6 均通过，
说明在通过前置门的场景上 O4 因果后果可稳定复现；但 benchmark 的端到端承重指标是 4/12，
不能报告成 6/6，更不能用它宣称 ICRA-ready。两份 amendment 只补齐“编译器在写原生 JSON 前
抛异常”的留痕，以及明确 phase-v6 对旧 raw visual aggregate 的注册优先级；没有改变任何
轨迹、阈值、材质分配、独立性要求或分母。

场景层的新 registry 又暴露了一个必须在扩量前修复的设计问题：Bedroom、LivingRoom、Kitchen
三个真实 EmbodiedGen scene family 已逐文件重哈希并通过 source/compile/RTX/Go2 全部门，恰好
覆盖 life 域 train/val/test 各一个；但旧 396 条 schedule 使用的 9 个场景仍是另一组概念 ID，
当前实体绑定为 **0/9**。因此 `planned=392` 只能解释为设计行数，不能解释为可立即采集的真实
场景 episode。后续正式 corpus 必须换用 registry-bound schedule，并为 production/wild 各补
3 个真实 scene family；禁止通过字符串改名把 life 房间冒充另外两个域。

## 已冻结的设计与资产

- 设计规模：pilot 396 条、full 5,940 条 physical episode，对应 1,188/17,820 条同步
  appearance-view sequence；统计单位仍是 physical episode，不按外观视图扩样本。每个异常都有
  同 scene/geometry/appearance/camera/seed 的 nominal counterfactual。design-v3 的 pilot
  不再用“单场景族×3 seed”冒充场景复现，
  而是每域 3 个独立 scene family × 每场景 1 seed；每个算子覆盖 9 个 scene family。
- split：按 scene family 与 material family 整组隔离，不按 frame 随机切分。
- 泄漏门按全部展开视图重算：pilot/full operator-material normalized MI 分别为
  `2.61e-17`/`1.37e-17`，condition-material MI 均为 `0.0`；每种材质覆盖全部 11 个算子和
  nominal/anomaly 两种条件；pilot 最小每域 3 个、每算子 9 个 scene family。
- 同物理多贴图门：每条 episode 冻结 `primary/swap_01/swap_02`；材质、surface state、真实尺度
  UV/旋转/偏移、albedo brightness、normal strength 和 roughness 独立随机化，但字段均标记
  `visual_intervention_only=true`。pilot 每组 3 个材质 family，full 每组至少 2 个。
- terrain 资产：18 套 ambientCG CC0 1K PBR，含 archive/map SHA-256、真实物理尺寸和 license lock。
- 运行时状态机：`planned -> collected -> validated_smoke|validated`。文件哈希、相机同步、
  proprio/telemetry、场景/外观/几何和算子读回均须独立复算；即使全部 artifact QA 通过，
  非 `formal_pilot` 采集协议也只能进入 `validated_smoke`，不能设置
  `evaluation_eligible=true`。runtime-v5 还逐视图验证时间同步、尺寸、图像/provenance 哈希、
  PBR 参数读回和语义可见性，并要求三条 RTX 序列不同、每个 swap 相对 primary 的平均归一化
  RGB L1 至少为 `0.015`，以拒绝 renderer cache 或“只改 metadata”的假随机化。
- `formal_pilot` 不再是可自由填写的字符串。正式 episode 必须复制冻结 protocol 文件并通过
  哈希、schema、schedule、collector bundle、runtime validator 以及允许的
  scene/operator/physical-realization/counterfactual-group scope 检查；缺失、篡改或越权均不能
  获得 evaluation eligibility。首个 protocol 仅授权 workstudio O4 的 2 个组，不能替其他
  392 条记录背书。

## Realistic operator gate 状态

| 模块 | 状态 | 已证实内容 | 尚缺内容 |
|---|---:|---|---|
| O1 低摩擦 + PBR | PASS（v3 未见场景确认 2/2；strict-v2 同为 2/2） | 在 G06/G07 两个新 photographic family、新 route/collision/trajectory 上，slip 增量 0.729/0.796、进度亏损 1.409/1.349 m；两条 anomaly 均跌倒，且接触前 telemetry 严格匹配 | 仍是 forest development-only；须接入 EmbodiedGen/production、多严重度与 formal registry，完成 fixture 标定 |
| O2 柔软地形 | PASS（v3 未见场景确认 2/2；strict-v2 同为 2/2） | G06/G07 沉陷 8.931/8.930 cm，视觉深度/沉陷比 0.983/0.966；nominal 完成而 anomaly 跌倒，首次表面交互前 telemetry 匹配 | 仍缺跨域 substrate family、连续辙迹、多严重度、土槽 fixture 与 formal registry |
| O3 坍塌 | PARTIAL（未见场景 1/2） | G07 全项通过；G06 仍触发 13 单元、314 顶点与强高度/进度后果，但冻结时长内未触发二元 fall，故失败保留 | 必须在新版本中预先区分 fall 与 immobilized/subsided failure，并在新的 calibration→heldout 批次验证；不得回收 G06 失败。仍需跨域板材/scene、前置 RGB 可观测性、fixture 与 formal registry |
| O4 足端粘附 | PARTIAL（v34 全新 held-out 5/6；artifact/causal integrity 6/6；v32 开发 4/5） | v34 在 candidate 137–157 的首个冻结 6 scene/4 family cohort 上验证了 phase-aware containment 与 state-aware visual gate；五例 Continue 跌倒而 Backstep 安全后撤 0.236–0.275 m。LivingRoom 157 的旧通用视觉误拒被前瞻性合同正确识别为倒地后视觉后果 | v34 唯一真实负例 Office 156：Backstep 分支后仍前侵 0.219 m，FL 未 peel，FR 再粘附并跌倒；不得重跑或事后改判。v35 仅作为新开发候选：保持接触物理不变，以物理紧急态/2.5 cm 前侵门锁存更强后撤命令；必须在全新 development→heldout cohort 验证。仍缺第二物理 realization、production/wild、fixture 与 formal registry |
| O5 刚性负载 | PASS（单点 + 4 档×5 seeds） | 偏置单点验证完整 mass/CoM/inertia 与可见体合同；中心安装 2/4/6 kg 剂量下，CoM 位移 2.41/3.94/4.99 cm、惯量迹增量 0.0877/0.1623/0.2298 kg·m²、四足承载 167.13/186.05/205.84 N；关节力矩利用率 P90 为 0.302/0.322/0.454/0.526，5/5 严格排序；severe 限幅占比 4.09%，2/5 跌倒 | 可更换真实负载外观、独立安装位姿与恢复后果；旧 `effort_ratio` 因“平均后再过 70% 阈值”仍为 0，只保留兼容，不再承担证据 |
| O6 外力冲击 | PASS（4 档×5 配对需求条件） | 以 0.12 s 有限 PhysX wrench 取代根节点速度写入；作用点 `[0,-0.085,0.075]` m 位于实时读取的 Go2 base 包围盒内；2/6/12 N·s 线冲量均完整施加并产生预期滚转角冲量；横向速度增量均值 0.240/0.573/0.966 m/s，5/5 严格排序；severe 最终横向偏差均值 0.711 m | 当前只有单一地面和单一推力方向/作用点；需独立场景、冲击方向/时相/持续时间 realization、动态碰撞体与真实摆锤标定 |
| O7 视觉—物理错配 | PASS（5 PBR×4 档配对工程门） | Go2-front body-fixed RTX RGB-D；真实 mask 后采集 depth bias；RGB pipeline 前后同哈希；raw depth/mask 反事实匹配；0/8/18/32 cm bias 对应地图位移 0/6.49/14.58/25.94 cm，5/5 严格排序；μd 0.60/0.30/0.13/0.06 回读 | 真实 Go2 相机外参/深度噪声标定、动态轨迹与恢复后果、正式 counterfactual episode |
| O8 透明障碍 | PASS（3 档光学难度×5 配对需求条件） | 0.38 m 真实 OmniGlass/PhysX 面板；每个 anomaly 都有同一 USD 几何、材质和 shader、仅关闭 collision 的 nominal；实际霜化粗糙度 0.12/0.06/0.02 在 5/5 条件严格排序，玻璃 IOR=1.49；三档物理后果精确不变，平均进度亏损 1.368 m、停滞增加 3.86 s，nominal 全穿越、anomaly 全阻挡且 0/15 跌倒 | 仍缺 RTX 像素级可见性/成对图像等价门、室内多背景、边框/污渍/反射 realization、低杆独立类别与真实亚克力 fixture |
| O9 托底/高中心 | PASS（单点 + 4 档×5 seeds） | 真实 0.3762×0.0935×0.114 m base collider；固定 0.35 m 宽托盘中梁，围绕净空扫描 350/355/360 mm；连续腹部接触 5/290/300 steps，占空比 10.28/96.69/99.34%，进度亏损 0.004/1.243/1.260 m；呈清晰临界跃迁，非线性响应 | 自然驶入式圆木/路沿/石脊 realization 与真实 fixture 标定 |
| O10 执行器降额 | PASS（4 档×5 配对需求条件） | 同步缩放 continuous effort cap 与 DCMotor 堵转力矩/torque-speed envelope，读回 23.5/17.625/11.75/5.875 N·m；关节利用率 P90 均值 0.308/0.411/0.616/0.965，5/5 条件严格排序；severe 限幅占比 24.71%，进度亏损 1.884 m（95% bootstrap CI 1.772–2.016） | 当前是 endpoint-cap 工程门，不是温度动力学；仍需 voltage/temperature/thermal ramp、单腿/单关节模式和真实电机标定 |
| O11 观测偏置 | PASS（4 档×5 配对需求条件） | Isaac base IMU 与带时间戳 odometry 在统一估计器前注入 fault；原始 tilt error 均值 0/0.080/0.179/0.348 rad，估计后 0/0.044/0.139/0.306 rad，5/5 严格排序；实际 age 精确为 0/40/100/200 ms，真值轨迹逐 seed 同哈希 | dropout 仅有单元门；需真实 Go2 bias/noise/温漂/丢帧标定，并让恢复控制消费故障观测 |

O9 此前误判的根因已经定位：训练环境把任何 `base_contact` 当作终止，并在同一个
`env.step()` 内自动 reset，telemetry 因而读到重置后的 0 N。benchmark 现在关闭这条训练期
自动终止，保留 Kino-Fail 自己的高度/倾角判跌倒，并同时读取三帧 physics-rate contact history。
新 gate 不接受单次初始化冲量：要求真实 base collider、足端卸载、连续腹部承重与运动后果同时成立。

## 11 算子 readiness

| 算子 | realistic readiness | 进入 pilot？ |
|---|---|---:|
| O1 | v3 在 G06/G07 未见外观+几何上 2/2，strict-v2 亦 2/2 | 否；尚未跨域接入 EmbodiedGen/production，也无 multi-dose、fixture 与 formal collector |
| O2 | v3 在 G06/G07 未见外观+几何上 2/2，strict-v2 亦 2/2 | 否；仍须跨域 substrate、multi-dose、土槽标定、连续辙迹和 formal collector |
| O3 | G06/G07 冻结确认 1/2；G06 强陷落但未触发 fall 的失败保留 | 否；先版本化 failure endpoint 并做全新确认，再补跨域板材/scene、自然驶入、RGB 可观测性与 fixture |
| O4 | v34 operator-blind held-out integrity 6/6、严格 5/6；v32 development integrity 5/5、严格 4/5 | 部分；v34 已前瞻性解决失稳后 route consequence 与倒地视觉误拒，但发现一个真实的步态相位相关 Backstep 失败。v34 六场均永久封存；v35 自适应后撤只能在新场景开发并再次 held-out 确认。canonical corpus、跨域 realization、production/wild 和真实 fixture 仍缺 |
| O5 | 可见负载 vertical slice 与中心安装 4 档×5-seed 质量剂量门通过；joint torque utilization/power/energy 已接通 | 否；仍需外观/安装 realization 与恢复后果 |
| O6 | 0.12 s 偏置作用点 PhysX wrench 与 4 档×5 配对剂量门通过；输入冲量、角冲量、实时机身包围盒和运动响应均有读回 | 否；仍缺多场景/方向/时相/持续时间 realization、动态碰撞体和真实摆锤标定 |
| O7 | 5 套 PBR 的 Go2-front RTX/depth/摩擦 4 档配对合同门通过 | 否；仍需实机相机标定、动态恢复后果与 corpus runtime manifest |
| O8 | 固定物理的真实 OmniGlass 透明面板与 collision-off/on 视觉同一 counterfactual 门通过；光学 shader 三档×5 条件通过 | 否；仍缺 RTX 像素级门、多背景/反射 realization、低杆独立类别和真实 fixture |
| O9 | 真实接触门与净空临界剂量门通过 | 否；仍缺自然驶入、多几何与真实 fixture |
| O10 | 全机 endpoint cap 4 档×5 配对需求条件通过，continuous/stall torque cap 与曲线拐点均读回 | 否；仍缺 time-varying thermal state、局部降额与真实电机标定 |
| O11 | Isaac raw IMU + timestamped odometry 在统一估计器前注入 bias/random walk/latency；4 档×5 配对工程门通过 | 否；仍需 dropout Isaac 门、真实 Go2 日志标定及 fault-aware recovery consequence |

## 对 A0–A7 的影响

- 旧 controlled causal core 保留，不覆盖、不改写历史结论。
- 新增机器可执行的 realistic replication 合同与 readiness audit：
  `configs/eval/kinofail_realistic_a0_a7_v1.json`、
  `outputs/eval/realistic_a0_a7/readiness_audit.json`。当前严格结果为 **0/8**；A0–A7 全部仍缺，
  因为旧 controlled 产物被明确禁止作为 realistic completion evidence。A0 当前观测为
  4/396 eligible episodes、2/198 完整反事实组、1/9 scene family、1/3 domain、1/11 operator、
  1/3 camera profile；A1–A7 的 realistic confirmatory JSON 和统一 prediction JSONL 均不存在。
- realistic replication 已产生首批可进入后续分析的正式数据，但 **不能启动完整主结果统计**：
  15/15 工程门和 4 条 O4 正式记录不等于 11 算子/9 scene family readiness；canonical formal
  audit 仍为 392 planned + 4 validated，只有 2/198 完整反事实组。
- 事件对齐 snapshot development 管线已对现有 O4 子集生成 12 条多视角记录（4 条 physical
  episode、2 个独立反事实 pair、每 episode 3 个同步 appearance view）。异常 episode 的
  `attached` privileged event 决定唯一 decision time，nominal 严格复用同一时间；五帧 RGB 和
  500 ms proprio window 的源文件哈希均已复核。该里程碑只证明抽取器可用，三个 appearance
  view 不作独立样本，`counts_as_experiment_ready=false`，readiness 仍为 0/8。
- texture-swap 的数据/运行时/损失/评测代码已就绪，但还没有 realistic A0–A7 模型 prediction
  JSONL，因此 0.95 consistency、0.95 agreement 和 0.05 JSD 是冻结验收门，不是已取得结果。
- 设计编译器新增 registry-bound formal gate。`python scripts/build_kinofail_realistic_corpus.py
  --mode pilot --require-registry-binding` 在当前状态按预期失败：设计本身仍为 396/396、泄漏门
  通过，但 9 个概念 schedule scene 中 0 个绑定到通过审计的同 ID 实体。正式采集不得绕过该门。
- production/wild 场景补齐已经启动 development-only EmbodiedGen visual-shell 路线；prompt、seed、
  模型版本、像素预检以及“生成 mesh 不得承担主要可行走碰撞”规则预先冻结在
  `configs/data/kinofail_embodiedgen_visual_shell_development_v1.json`。截至当前，冻结 seed 的
  warehouse 与 forest panorama 均通过像素预检；人工 development review 判定 forest 可作为
  3D 候选，warehouse 只适合作为稀疏远景/开发候选，二者都尚未通过独立人工 QA。3D mesh
  阶段另以锁定 EmbodiedGen/Pano2Room/PyTorch3D/gsplat/CUDA 的合同运行，并明确
  `counts_as_a0_a7_evidence=false`。通过 mesh 门后仍须完成 novel-view、Kino-owned
  collision/heightfield、body-fixed Go2 RTX QA、operator capability audit 和 registry binding；
  当前 scene registry 新增计数仍为 **0**，不能进入 A0 分母。
- 下一批工作顺序：保留 v4 0/3、v5 1/3、作废的旧 v6 0/5、v6-native 2/6、route-surface-v7
  4/12、lighting-v5 13/21、material-v6 12/12、v31 1/6、v32 4/5 和 v34 5/6 全部封存结果。
  v34 的 scene acquisition 在冻结候选前缀 137–157 中处理 21 条、入选 6 条、覆盖 4 family；
  suffix 158–160 保持未触碰。其唯一负例是 Backstep 真失败，不是审计误拒；因此先在全新
  development cohort 验证 v35 闭环升级后撤，再冻结新的 held-out cohort。只有 O4 再确认通过，
  才扩展 O5–O11 realistic operator coverage。随后把 396 条设计改为真实
  scene-registry 绑定并补 production/wild 场景；并行完成 O5 独立
  scene/appearance/mount realization，O2/O3 独立
  soil/substrate/fracture realization，O11 dropout/实机标定及 O10 thermal ramp → 统一 collector
  采 396 pilot → 运行时完整性/泄漏/同步审计
  → 才进入 A0–A7。

## 证据入口

- 设计审计：`outputs/kinofail_realistic/design_v1/`
- 资产锁与 contact sheet：`outputs/assets/terrain_pbr_v1/`
- O1/PBR：`outputs/kinofail_realistic/terrain_pbr_gate/`
- O1 旧三外观审计（已 superseded，仅保留资产/外观记录）：
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_o1_three_appearance_families_v1/audit.json`
- O1 修正后的 matched-topology pair：
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_hybrid_monks_dev01/isaac_hybrid_composition_dev_v14/o1_matched_topology_pair_dev_v14_attempt4/pair_audit.json`
- O1 修正后三外观汇总：
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_o1_matched_three_appearance_families_v2/audit.json`
- O3 修正后的 matched-topology trigger pair：
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_hybrid_monks_dev01/isaac_hybrid_composition_dev_v14/o3_matched_topology_pair_dev_v14_attempt7/pair_audit.json`
- O3 修正后三外观汇总：
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_o3_matched_three_appearance_families_v1/audit.json`
- O1–O3 独立 metric-geometry 汇总（G02 calibration + G03 frozen held-out）：
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_metric_geometry_o1_o3_summary_v1/audit.json`
- O1 独立几何 protocol/preflight/postrun：
  `configs/data/kinofail_forest_metric_geometry_o1_confirmation_dev_v1.json`、
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_metric_geometry_o1_confirmation_dev_v1/`
- O2 独立几何 protocol/preflight/postrun：
  `configs/data/kinofail_forest_metric_geometry_o2_confirmation_dev_v1.json`、
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_metric_geometry_o2_confirmation_dev_v1/`
- O3 独立几何 protocol/preflight/postrun：
  `configs/data/kinofail_forest_metric_geometry_o3_confirmation_dev_v1.json`、
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_metric_geometry_o3_confirmation_dev_v1/`
- O2：`outputs/kinofail_realistic/o2_terramechanics_gate/`
- O2 剂量：`outputs/kinofail_realistic/o2_dose_sweep/`
- O2 修正后的 matched-partition pair：
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_hybrid_monks_dev01/isaac_hybrid_composition_dev_v14/o2_matched_topology_pair_dev_v14_attempt2/pair_audit.json`
- O2 修正后三外观汇总：
  `outputs/kinofail_realistic/scene_sources/polyhaven_hdri_v1/forest_o2_matched_three_appearance_families_v2/audit.json`
- O3：`outputs/kinofail_realistic/o3_topology_gate/`
- O3 剂量：`outputs/kinofail_realistic/o3_dose_sweep/`
- O2/O3 live RTX 多视角审阅包：`outputs/kinofail_realistic/o2_o3_visual_evidence/`
- O4 足端粘附剂量：`outputs/kinofail_realistic/o4_adhesion_dose_sweep/`
- EmbodiedGen 场景 registry 与逐文件审计：
  `configs/data/kinofail_embodiedgen_scene_registry_v1.json`、
  `outputs/kinofail_realistic/scene_registry/registry_audit.json`
- Kitchen O4 正式 pair 与独立 admission audit：
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/Kitchen_seed20260727/kinofail/o4_pairs/formal_heldout_seed20260728_v4/`
- 新增 held-out scene family 的生成前确认批次：
  `configs/data/kinofail_embodiedgen_o4_confirmation_batch_v1.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_confirmation_batch_v1_audit.json`
- robust-corridor v2 合同、开发准入与新三场景冻结批次：
  `kino_vla/sim/embodiedgen_scene_v2.py`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/House_seed20260731/kinofail_corridor_v2_tracking030/corridor_v2_admission_audit.json`、
  `configs/data/kinofail_embodiedgen_o4_corridor_v2_extension_batch_v1.json`
- corridor-v2 扩展批次封存审计、v5 开发冻结/审计与新 held-out 批次封存审计：
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_corridor_v2_extension_batch_v1_audit.json`、
  `configs/data/kinofail_o4_forward_window_v5_development_plan.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_forward_window_v5_development_audit.json`、
  `configs/data/kinofail_embodiedgen_o4_forward_window_v5_confirmation_batch_v1.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_forward_window_v5_confirmation_batch_v1_audit.json`
- phase-aware v6 开发、作废批次、v6-native 独立性与最新 6 场封存审计：
  `configs/data/kinofail_o4_phase_aware_v6_development_plan.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_phase_aware_v6_development_audit.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_phase_aware_v6_confirmation_batch_v1_invalidation_audit.json`、
  `configs/data/kinofail_o4_independence_v6_development_plan.json`、
  `configs/data/kinofail_embodiedgen_o4_v6_native_confirmation_batch_v1.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_v6_native_confirmation_batch_v1_audit.json`
- 路线表面/照明诊断、v5 失败矩阵、source preflight 与 v6 材质开发矩阵：
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/route_surface_lighting_v4_development_matrix_v1_invalidation_audit.json`、
  `configs/data/kinofail_embodiedgen_route_surface_lighting_v5_development_matrix_v1.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/route_surface_lighting_v5_development_matrix_v1_audit.json`、
  `configs/data/kinofail_embodiedgen_source_geometry_preflight_v1.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/source_geometry_preflight_v1_validation_audit.json`、
  `outputs/kinofail_realistic/terrain_material_visual_preflight_v1/train_audit.json`、
  `configs/data/kinofail_embodiedgen_route_surface_material_v6_development_matrix_v1.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/route_surface_material_v6_development_matrix_v1_audit.json`
- 12 场 val/test route-surface-v7 冻结确认、两份行政 amendment 与官方封存审计：
  `configs/data/kinofail_embodiedgen_o4_route_surface_v7_confirmation_batch_v1.json`、
  `configs/data/kinofail_embodiedgen_o4_route_surface_v7_confirmation_batch_v1_amendment1.json`、
  `configs/data/kinofail_embodiedgen_o4_route_surface_v7_confirmation_batch_v1_amendment2.json`、
  `outputs/kinofail_realistic/scene_sources/embodiedgen_v2/o4_route_surface_v7_confirmation_batch_v1_audit.json`
- O4 v34 operator-blind acquisition、一次性 held-out 结果与 v35 开发候选：
  `configs/data/kinofail_o4_scene_stream_v34_heldout_acquisition_freeze_v1.json`、
  `outputs/kinofail_realistic/operator_confirmation/o4_action_v34_scene_stream/postrun_audit.json`、
  `configs/data/kinofail_o4_action_consequence_v34_heldout_confirmation_freeze_v1.json`、
  `outputs/kinofail_realistic/operator_confirmation/o4_action_v34_heldout_confirmation/postrun_audit.json`、
  `scripts/isaac_collect_o4_action_consequence_v35.py`
- O5：`outputs/kinofail_realistic/o5_payload_gate/`
- O5 剂量：`outputs/kinofail_realistic/o5_dose_sweep/`
- O6 有限时长外力剂量：`outputs/kinofail_realistic/o6_force_pulse_dose_sweep/`
- O7 Go2-front RTX/depth 合同：`outputs/kinofail_realistic/o7_rtx_depth_gate/`
- O8 透明面板视觉—物理合同：`outputs/kinofail_realistic/o8_transparent_gate/`
- O9（通过证据与旧误判诊断）：`outputs/kinofail_realistic/o9_high_centering_gate/`
- O9 净空阈值剂量：`outputs/kinofail_realistic/o9_dose_sweep/`
- O10 执行器剂量：`outputs/kinofail_realistic/o10_dose_sweep/`
- O11 原始传感故障剂量：`outputs/kinofail_realistic/o11_sensor_fault_gate/`
- 汇总门：`outputs/kinofail_realistic/terrain_operator_gates/`
- 全算子工程汇总：`outputs/kinofail_realistic/operator_gates/`
- runtime corpus 审计：`outputs/kinofail_realistic/runtime_audit/pilot/`
- O4 runtime-v5 smoke pair：`outputs/kinofail_realistic/corpus_v1/smoke_pair_summary.json`
- O4 冻结正式子集：`outputs/kinofail_realistic/corpus_v1_formal/o4_formal_pilot_summary.json`
- 正式 partial runtime 审计：`outputs/kinofail_realistic/runtime_audit/formal_pilot_partial/`
- 正式覆盖/随机化报告：
  `outputs/kinofail_realistic/runtime_audit/formal_pilot_partial/coverage.json`
- 同状态三贴图 contact sheet：
  `outputs/kinofail_realistic/corpus_v1/texture_swap_same_state_contact_sheet.png`

当前汇总不仅检查 manifest 的 `passed`，还逐项重算 policy/backend/operator/model/gate/config
的 SHA-256；任一输入变动都会把对应 gate 自动标记为 stale。当前已用包含
O2/O3 视觉同步、O7/O11 与 Go2-front camera 的最终后端统一复跑全部 15 个工程门；汇总逐项重算
policy/backend/operator/model/gate/config SHA-256，结果为 **15/15 fresh PASS**。
O2/O3 的 6 张 live RTX 图和白底 contact sheet 仅标记为
`review_bundle_not_corpus_evidence`，不计入 15 个统计/物理工程门，也不替代独立 realization。
`publication_ready=false` 仍保持不变。正式 partial audit 当前为 `validated=4/396`、
完整反事实组 `2/198`；实际覆盖 1/11 operator、1/9 scene family、1/3 domain、1/3 camera
profile、4/18 material family。4 条数据共含 774 帧 RTX RGB 和 1,278 条 raw-proprio；8 个
primary→swap 比较的平均归一化 RGB L1 为 `0.0183–0.1278`，均通过 `>=0.015` 可见干预门。
这证明冻结正式采集与贴图干预链已开始产生合格数据，不是模型不变性成绩，也不替代剩余
392 条 pilot、A0–A7 replication 或真实 Go2 anchor。
