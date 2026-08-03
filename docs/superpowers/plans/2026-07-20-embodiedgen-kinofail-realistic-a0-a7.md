# EmbodiedGen × Kino-Fail 真实异常归因数据集与 A0–A7 增强计划

> 2026-07-23 v27 更新：在不改 O4 物理、Go2 policy、route controller、命令速度和验收阈值的
> 前提下，v27 将固定决策时长改为 5–25 步的有界姿态紧急触发，并将固定 75-step Backstep
> 改为最多 120 步、目标 0.20 m 且 adhesion 连续清零 5 步的闭环 clearance。三条已暴露 v16
> development scene 一次性运行后 3/3 通过，matched-state、物理 peel、站立恢复和 0.15 m
> 距离门均通过；postrun `sealed=true`。但三场都走 maximum-dwell，未覆盖 urgency 分支，且
> development 数据不计 A0–A7。下一门是先以 nominal-only QA 冻结全新、从未执行 O4 的
> realistic cohort，再原样执行 v27；readiness 仍为 **0/8**。
>
> 2026-07-23 v25/v26 更新：v25 严格前缀处理 16 个 EmbodiedGen simple-room 候选，在第 16 个
> 候选首次达到 6 个 admitted scene、4 个 family 并立即停流；第 17–24 个未触碰。v25
> postrun `passed=true, sealed=true`，只冻结 realistic cohort，不计 A0–A7。随后 v26 把 v21
> Continue/Backstep 合同原样绑定全部 6 场并一次性执行，完整性 6/6，但主结论仅 3/6：前三场
> 正向；Office78/84 的 Backstep 在 peel 前跌倒；House87 完成站立与 peel 但只后退 0.1294 m，
> 未过 0.15 m 门。v26 `sealed=true, passed=false`，不得在这 6 场调参或删失败。下一步是在独立
> development scenes 上实现支撑相位/卸载裕度触发的 phase-conditioned recovery，再用全新
> held-out cohort 确认；并行主线仍是 O5–O11 realistic 覆盖、registry-bound corpus、重训和
> A0–A7 正式重跑。readiness 保持 **0/8**，A8 永久排除。

> 2026-07-22 v16 执行更新：目标保持为在新版 realistic Kino-Fail 上**重新完成 A0–A7**，
> 不是整理旧实验。六个未见 EmbodiedGen 请求按不替换合同得到 3/6 admitted stack
>（LivingRoom/Kitchen/Bathroom）；O1/O2/O3 已在三场上完成物理重新校准并封存，但仍不计入
> A0–A7。O4 使用冻结 35 N recoverable-peel 单合同得到主目标 0/3：三场机制均生效但均在 peel
> 前终止跌倒，说明 severe terminal 与 moderate recoverable 必须拆成不同预注册 endpoint。
> 下一步固定为 O4 train-only 双 endpoint 校准与全新 held-out 确认，再扩展 O5–O11，随后绑定
> scene registry 和 396 条 pilot，最后在新语料上训练、推理并正式重跑 A0–A7。严格 readiness
> 仍为 **0/8**；A8 继续排除。
>
> O4 v17 train-only 追加诊断：预声明 8/12/16 N 搜索得到 0/3，没有选择 moderate dose。
> 三档均因单足 world anchor 跨摆动相持续存在而在 peel 前跌倒；这证明问题在状态模型，不在
> force cap。v18 将把 surface `adhesive_contact`（接触/承重感知、各向异性剥离、可重复 stance
> cycle）与可见 `tether_entanglement`（持续锚点、terminal endpoint）拆开，并替换 terminal
> 场景中的 progress-lag 指标。v16 0/3 与 v17 0/3 均保留，不能通过新版本回填。
>
> O4 v18/v19 更新：contact-aware v2 已实现并通过 16 个新旧回归测试。v18 产生 23 次
> attachment/peel cycle、5.16 J 耗散且不跌倒，但效应量不足；v19 18/24 N 均在 step 167
> 跌倒且没有可选强度。下一版不再优化 cap，而是冻结同异常、同决策状态的 Continue/Backstep
> action branch，以实际恢复后果作为 A4 架构门。所有结果仍为 train-only，readiness 0/8。

> 实施状态（2026-07-22）：设计编译、18 套 PBR 资产、运行时 manifest gate、O1/O2/O3/O9
> 11 个算子已形成 15 个 realistic 工程门，并在最终 O7/O11/Go2-front camera 后端上统一
> 复跑为 15/15 fresh PASS；其中包含 O2/O3/O5/O9 的 4 档×5-seed 配对剂量门、
> O4/O6/O10/O11 的 4 档×5 配对条件门、O7 的 5 PBR×4 档 RTX/depth 门和 O8 的
> 3 光学档×5 条件 collision 反事实门；O2 接触足迹与 O3 触发后破裂面已完成同状态
> RTX 审阅包，但不作为 corpus 证据。terrain 外观随机化现已进入 design-v3/runtime-v5：
> 每条物理 episode 有 3 个同状态、同时间戳的 PBR 外观视图，pilot/full 分别为
> 396/5,940 条物理记录和 1,188/17,820 条外观序列。采集层现已增加不可由单个 status 字段
> 冒充的冻结 formal protocol gate，并完成首个 workstudio O4 子集：moderate/severe 两个
> nominal/anomaly 反事实组共 4 条 physical episode、12 条 appearance sequence、774 帧 RTX
> RGB 和 1,278 条 raw-proprio，全部通过 runtime-v5 并获得 evaluation eligibility；实际
> texture-swap RGB L1 为 0.0183–0.1278。
> 当前 canonical formal audit 为 392 planned + 4 validated（4/396 eligible、2/198 完整组），
> `publication_freeze_ready=false`，尚不能运行完整 A0–A7 realistic 主结果。逐项证据见
> `docs/datasets/kinofail_realistic_v1_progress.md`。
>
> 2026-07-22 追加审计把“旧 controlled A0–A7 完整”和“新版 realistic A0–A7 已复现”彻底
> 分开：新合同 `configs/eval/kinofail_realistic_a0_a7_v1.json` 要求 396/396 physical episodes、
> 198/198 counterfactual groups、11 算子、9 scene families、3 domains、3 camera profiles 先通过
> A0，再分别产生 A1–A7 的新 confirmatory 产物与 prediction JSONL。当前机器判定为 **0/8**；
> 已封存 O4 route-surface-v7 的 4/12 仅是 operator/pipeline replication，不计作完整 A0。
> 正式 schedule 生成器现已增加 `--require-registry-binding`：当前真实运行会因计划 9 场景中
> `validated_scheduled_scene_count=0` 而非零退出，从机制上禁止把 396 条设计行误报为可采数据。
> 现有 O4 已另生成 12 条 development snapshot 记录（4 physical episode / 2 pair / 3 同步
> appearance view），但抽取审计硬编码 `counts_as_experiment_ready=false`。production/wild 的
> warehouse/forest 冻结 panorama 均通过像素预检，forest 被选入 3D development；两者仍为
> `scene_registry_eligible=false`。这些只是为了最终在新版真实场景中重采 A0–A7，而不是用资产
> 生成里程碑代替 A0–A7 复现。
>
> 2026-07-22 wild/O2 更正：Monk's Forest、Whipple Creek、Hochsal Forest 三套冻结
> photographic/PBR appearance family、不同 RTX 图像和 contact-local 压痕实现仍保留，但旧三
> 外观 nominal/anomaly 因果审计已 **superseded**。复核发现场景 nominal floor 为 μs/μd=
> 0.8/0.6，而当时 anomaly 隐藏分区周边被写成 0.8/0.8；旧的 10.318 cm 高度差、8.921 cm
> 沉陷、15.096 J 剪切耗散和跌倒不能归因于 O2 单一因素。修正后的 Monk's pair 已用相同
> 4+3 分段重采：前 125 行完全相同，第 126 步首次区域承重接触且首次分歧；nominal 完成，
> anomaly 沉陷 8.94 cm、剪切 26.01 J、高度差 10.57 cm、进度亏损 1.1140 m 并跌倒。
> 修正协议现已在 Monk's/Whipple/Hochsal 三套冻结 photographic/PBR 外观上全部通过；但三套
> `collision.usda`、`prop_collision.usda`、`route.usda` 哈希完全相同，所以它只关闭三外观门，
> 不构成三物理场景复现。multi-dose、独立 soil/scene/metric geometry 和 A0–A7 重跑未完成，
> realistic readiness 继续是 **0/8**。
>
> 2026-07-22 wild/O1 更正与修复：旧三外观审计也因上述 0.8/0.6 vs 0.8/0.8 周边摩擦
> 混杂而 superseded。修正后 nominal/anomaly 都安装完全相同的 4+1 共面隐藏分区；nominal
> 中心/周边直接读取场景 0.8/0.6，anomaly 只把中心改为 0.04/0.03。Monk's Forest 新 pair
> 前 125 步完整遥测逐行相同，第 126 步首次区域承重接触且首次分歧；nominal 完成，anomaly
> 跌倒，slip +0.8318、进度 −1.3525 m、横向偏移 +0.2006 m。该协议随后在三套冻结外观上
> 全部通过；由于三者共享同一 metric geometry、控制命令和动力学轨迹，只关闭 appearance
> robustness 开发门。multi-dose、独立 scene/geometry/trajectory、fixture、正式 schedule 与
> A0–A7 重训/统计仍缺，readiness 保持 **0/8**。
>
> 2026-07-22 wild/O3 追加进展：nominal/anomaly 都使用相同 4×4 零缝隙隐藏支撑拓扑和
> 0.8/0.6 摩擦；nominal 仅使用不可达损伤阈值，anomaly 由 25 N·s 实测足端冲量触发。
> attempt6 前 141 行遥测完全相同，第 142 步首次分歧与触发一致；同一步关闭 13 个 collider，
> 使连续 PBR 的 293 个顶点不规则下陷至 −0.1141 m。nominal 完成，anomaly 跌倒，进度亏损
> 1.4571 m。前置 RGB 在触发瞬间看不到足下凹坑，因此这里只证明物理—网格同步。修正协议也
> 已在三套冻结外观上全部通过，但共享同一 metric geometry/trajectory，仍只是 appearance
> robustness 开发证据，不进入 registry/A0–A7，readiness 仍为 **0/8**。
>
> 2026-07-22 independent metric-geometry 追加进展：新增 G02/v19 与 G03/v20，两者相对 G01
> 改变路线中心线、长度、宽度、endpoint margin、刚性道具碰撞布局和入射轨迹；三套
> `collision.usda`、`prop_collision.usda`、`route.usda` 哈希逐项两两不同。G02 只作
> development calibration；在任何 G03 operator outcome 运行前分别冻结 O1/O2/O3 一次性协议。
> 三项 G03 held-out geometry confirmation 均通过：O1 slip +0.7663、进度 −1.0876 m；O2 沉陷
> 9.44 cm、剪切 36.64 J、高度差 13.19 cm、进度 −1.1679 m；O3 由 26.22 N·s 触发 13 个
> collider 失效、309 顶点下陷至 −11.25 cm、进度 −1.5006 m。三项均为 nominal 完成、anomaly
> 跌倒，并通过严格前干预相等审计。统一汇总为
> `forest_metric_geometry_o1_o3_summary_v1/audit.json`。这只关闭首个独立物理几何开发门；每个
> 算子仍只有一个 held-out geometry，且 G02/G03 共用同一 forest visual-shell 来源，不进入
> registry/A0–A7，readiness 仍为 **0/8**。

> 日期：2026-07-21  
> 状态：已批准启动；当前垂直切片为“室内场景 + O4 adhesion/external-attachment”  
> 实验边界：所有新增证据继续归入 A0–A7，不建立或恢复外部 A8  
> 目标投稿：ICRA；优先保证因果有效性、物理可解释性、可复现性和诚实的 sim2real 边界

## 0. 一句话目标

以 EmbodiedGen V2 作为场景与资产生成底座，以 Kino-Fail 自研的场景编译器、Go2 前置相机链、足端/约束物理和因果标签作为可信执行层，构建同时具有受控因果证据、真实场景复杂度和真实机器人锚点的异常归因数据集，并在同一 A0–A7 实验体系内重新验证现有主张。

## 1. 背景与问题定义

现有 Kino-Fail 已经建立了 11 个异常算子、冻结 snapshot、跨模态归因、恢复后果矩阵和 A0–A7 证据链。但目前数据生成过程仍存在四类承重风险：

1. **视觉环境过于简化。** 正式 A0/A3/A5.6 采集器没有接入 live camera 时，会落入程序化地面渲染；场景主要由纯色/简单贴图 patch 和默认地面组成。
2. **相机不等价于未来真实 Go2 传感器。** 现有 RTX 相机是 world-frame 跟拍视角，不是固定在机身上的前置相机；现有 live snapshot 路径还会复制单帧并令 depth 为零。
3. **原基线中部分算子只有等效症状，没有正确局部物理。** O2/O4 在 Isaac 中使用躯干外力；O3 以摩擦切换代替坍塌；O5 未落实 CoM offset；O6 直接改根节点速度。这些问题必须逐项通过 realistic gate 关闭。
4. **sim2real 主张缺少真实锚点。** 纯合成数据最多能称为 sim2real-oriented；若没有真实 Go2 fixture 的标定和盲测，不能称为已验证的 sim2real 数据集。

EmbodiedGen V2 可以提供高质量室内场景、实例化物体、PBR 外观、碰撞代理和 USD 导出，但不能成为异常物理真值。其质量、摩擦和尺度中的一部分来自 VLM 推断；公开评测主要针对物体可用性和机械臂抓取，不等价于足式接触真实性。因此本计划采用“生成场景与物理执行解耦”的架构。

## 2. 科学主张与诚实边界

### 2.1 计划支持的主张

- 在真实感室内、生产和野外环境中，不同异常原因可以产生相似早期运动症状，但需要不同恢复。
- 视觉与本体感觉的可靠性随异常家族变化，固定信任任一模态都不足以覆盖双向冲突。
- 经过物理校准和无泄漏反事实设计的多模态归因，可以在场景/材质留出条件下改善归因和恢复决策。
- 受控因果核心与真实感复现实验共同支持结论：前者回答“为什么”，后者回答“是否依赖简化场景或贴图捷径”。

### 2.2 暂不允许的主张

- 不把 EmbodiedGen 的 `sim-ready` 直接写成“地形物理真实”。
- 不把生成资产中的 VLM 质量、惯量或摩擦当作 ground truth。
- 不把程序化贴图、world-follow 相机或重复单帧称为 Go2 前置 RTX 序列。
- 在完成真实 Go2 fixture 盲测前，不使用“validated sim2real dataset”；只能使用“sim2real-oriented synthetic causal benchmark”。
- 不将原有 structured-v2 的满分直接继承到新数据生成过程。相机、物理和场景改变后，所有相关结论必须重新测量。

## 3. 总体架构

```text
EmbodiedGen V2
  文本/图像 -> 场景、物体、纹理、初始碰撞代理
        |
        v
Kino Scene Compiler
  尺度审计 -> 可行走面提取 -> visual/collision 分离
  -> 物理材质重绑定 -> semantic/instance 元数据 -> USD 组合
        |
        v
Kino Physics Override
  足端接触 / 真实约束 / 损伤状态 / 负载惯量 / 执行器 / 传感器
        |
        v
Go2 body-fixed RTX Sensor + Causal Logger
  多帧 RGB(/D) + proprio + contact + operator state + recovery outcome
        |
        v
A0–A7 Controlled Core + Realistic Replication
```

### 3.1 USD 分层合同

每个场景必须由以下可独立替换的层组成：

- `visual.usd`：场景视觉网格、PBR 材质、灯光和非碰撞装饰。
- `collision.usd`：经审计的可行走面、障碍碰撞体和简化几何。
- `physics.usd`：PhysX 材质、质量、质心、惯量、关节/约束和接触参数。
- `operators.usd`：当前 episode 的异常算子实例及其状态实体。
- `episode.usd`：组合上述 sublayer，并冻结场景 seed、算子 seed、相机 profile 和严重度。
- `manifest.yaml`：记录来源、许可证、EmbodiedGen commit、生成 prompt、模型、seed、资产哈希、物理参数来源和审计结论。

外观、碰撞、物理和标签必须可独立交换，以支持“同外观不同物理”和“同物理不同外观”的反事实实验。

## 4. 场景域设计

### 4.1 生活域

- 室内住宅、办公室、车库、走廊、公共室内空间。
- 优先使用 EmbodiedGen V2 RoomGen 的多房间/单房间场景。
- 保留家具实例，但清除妨碍 Go2 通行或碰撞代理明显错误的装饰。
- 当前首个 demo：**写实室内工作室/生活空间 + O4 adhesion/external attachment**。

### 4.2 生产域

- 仓库、装配区、维修间、施工通道。
- EmbodiedGen 负责托盘、箱体、工具、线缆、负载、障碍等资产；背景使用经过审计的仓库/工厂 USD 或扩展后的程序生成场景。
- 不把 RoomGen 的住宅房型硬称为工厂场景。

### 4.3 野外域

- 林地、碎石路、草地、泥地、坡面、倒木环境。
- EmbodiedGen 的单全景 3DGS/mesh 只能作为远景或视觉 shell，不能直接作为主要可行走碰撞地形。
- 近场使用程序地形、真实扫描 mesh 或高度场；EmbodiedGen 用于生成倒木、石块、植被和场景外观变化。

### 4.4 场景数量

最终数据至少包含每域 3 个独立 scene family；同一场景的随机灯光或摆件变化不计为新的独立场景。

## 5. Go2 相机与传感器合同

### 5.1 相机安装

- 相机 prim 必须位于 Go2 base/front camera frame 下，而不是 `/World`。
- 使用目标真实相机的实测外参、内参、分辨率、畸变、曝光、帧率和延迟。
- 相机随机身完整 roll/pitch/yaw 运动，不执行自动瞄准。
- 原始数据保留原生分辨率；模型输入可以在预处理时缩放。

### 5.2 时间序列

- 每个 snapshot 的 5 帧来自 5 个不同仿真时刻。
- RGB、depth、proprio 和接触状态具有统一时间戳。
- 禁止将同一帧复制为时间序列。
- 如果真实部署只使用前置 RGB，则主模型只使用 RGB；depth 只能作为独立的可选传感器实验，不能用全零 depth 代替。

### 5.3 传感器随机化

- 光照、曝光、白平衡、运动模糊、镜头畸变、延迟、丢帧和局部遮挡。
- 随机化范围来自真实采集统计或明确的工程上界。
- 相机 profile 必须参与数据分组，避免同一 calibration profile 泄漏到 train/test 两侧。

## 6. Terrain 外观—几何—物理合同

每个地面材质资产至少包含：

```text
TerrainMaterial
  albedo
  normal
  roughness
  displacement/height
  collision geometry
  physical parameters + uncertainty
  deformable/contact model
  semantic material ID
  calibration provenance
```

### 6.1 纹理来源

- headline 地形优先采用有真实尺寸标尺的扫描 PBR 材质。
- EmbodiedGen texture generation 用于外观扩充、道具和远景，不单独承担地形物理真实性。
- 所有纹理采用世界尺度 UV 或 triplanar mapping，记录 texel density，避免材质尺度错误。

### 6.2 几何与碰撞

- normal map 仅改变光照，不能作为物理粗糙度或碰撞高度。
- 数毫米以上、可能影响足端接触的结构必须进入 height field 或 collision mesh。
- 地面异常 patch 应切分原始地面或使用 per-face material；不得在原地面上叠加共面双碰撞体。
- 柔软地形需记录并可视化足迹/沉陷；坍塌地形需改变真实碰撞拓扑。

### 6.3 防止视觉标签泄漏

- 不使用“黄色必为 adhesion、棕色必为 mud”的一一映射。
- 每种物理机制至少有多个外观家族；每种关键外观至少有正常和异常物理反事实。
- patch 使用不规则空间 mask 和自然边缘过渡，不保留固定矩形轮廓。
- 数据 split 按 scene family、material family 和 physical realization 分组，而不是按 frame 随机切分。

### 6.4 同物理—多贴图干预协议（已实现）

terrain 算子的物理层继续由 Isaac/PhysX 中既有 O1/O2/O3/O7/O9 实现承担；本协议只改变
渲染外观，不重新定义摩擦、沉陷、坍塌、碰撞几何或高中心机制。具体合同如下：

- 每个 physical episode 冻结 3 个 `appearance_view`：`primary`、`swap_01`、`swap_02`。
  物理仿真在每个采样时刻只推进一次；随后暂停 timeline，在完全相同的 physics state 上
  依次换材质并只刷新 Fabric/Hydra/RTX，不推进 PhysX。
- 三个视图共享完全相同的 RGB 时间戳、proprio、contact、operator state、privileged
  telemetry、collision geometry 和 physics parameters。nominal/anomaly counterfactual 还共享
  byte-identical 的三视图外观表，从而不让外观替物理条件泄漏标签。
- 每个视图独立冻结 PBR family、surface state、真实尺度 UV、旋转/偏移、albedo 亮度、
  normal strength 和 roughness multiplier。所有字段标记 `visual_intervention_only=true`；算子 ID
  不参与 appearance hash 或采样。
- design-v3 对全部展开后的视图做平衡审计：pilot/full 的 operator–material normalized MI
  分别为 `2.61e-17`/`1.37e-17`，condition–material MI 均为 `0`；每种材质覆盖 11 个算子以及
  nominal/anomaly 两种条件。pilot 每个同物理组有 3 种材质，full 至少 2 种材质。
- runtime-v5 不接受“manifest 变了、图没变”：逐视图复算 PNG 哈希、尺寸、亮度、可见区域、
  provenance 和 PBR 参数读回；三条序列必须哈希不同，且每个 swap 相对 primary 的逐帧平均
  `mean(|RGB_primary-RGB_swap|)/255 >= 0.015`。视图数量缺失、时间戳不同步或 RTX 缓存复用
  都会硬失败。
- 训练时可按 `texture_swap_group_id` 使用 generalized Jensen–Shannon consistency loss；正式
  评测把归因正确率和贴图不变性分开报告，冻结门为 group hard consistency ≥0.95、pairwise
  agreement ≥0.95、mean probability JSD ≤0.05。稳定地预测错误仍算错误，不能用一致性替代
  A0–A7 任务成绩。

当前 live smoke 已执行 3 套结构化 PBR。设计表中的 70% structured PBR / 20% PBR+decal /
10% geometry-conditioned sequence generation 是正式 corpus 的后续分层目标；在对应 renderer
和 provenance 真正执行前，不把后两层写成已有数据。

## 7. 11 个异常算子修正方案

### O1 低摩擦

- 保留 PhysX 接触材质路线，但改为地面 material region，消除薄板台阶。
- 用真实 Go2 足垫—地面组合的静/动摩擦范围标定。
- realistic counterfactual 必须在两条 lane 中都安装完全相同的共面分区，nominal 中心块使用
  场景原始材质，anomaly 才改变中心块摩擦；区域承重接触前的逐步遥测必须完全相同。
- 旧三外观结果因 nominal μd=0.6 与 anomaly 周边 μd=0.8 的混杂已撤回。修正后的 Monk's
  Forest pair 在首次承重接触前 125 行完全相同，首次分歧与首次接触同为 step 126；修正协议
  已在 Monk's/Whipple/Hochsal 三外观上通过。另一个预冻结 G03 独立 metric geometry/trajectory
  held-out pair 也通过；当前必须继续补跨 split 多 scene family 与跨几何 multi-dose。

### O2 柔软/可沉陷地形

- 从躯干 drag 改为逐足压力—沉陷—剪切模型。
- 快速数据生成使用经标定的低维 terramechanics/contact law；少量高保真颗粒/PBD 或真实沙箱用于校准与验证。
- `d_sink` 必须真实影响足端和机身高度，并在 RGB/几何中形成足迹。
- 当前逐足模型已完成 nominal/mild/moderate/severe × 5 seeds 工程门：沉陷、机身高度损失、
  任务进度亏损均满足种子内排序，且 mild/moderate 可通行。该重复只证明确定性物理复现，
  不构成 corpus 统计证据。
- 当前单点门由实测承重接触生成 10 个持久、无碰撞足迹，覆盖 2 个命名足端，足迹中心与
  接触 XY 读回误差 <0.1 mm；三视角 RTX 仅供人工审阅。土槽标定、连续辙迹几何和独立
  scene/soil realization 未完成，因此仍不作为 corpus 统计证据。
- wild photographic vertical slice 进一步替换了“整块 operator 区域预变形”方案：初始地表
  不显示 operator boundary，只有 load-bearing 足端按实测 sinkage 在单一连续 PBR mesh 上
  形成局部压痕。Monk's/Whipple/Hochsal 三外观族共享 byte-identical 物理层，且在 held-out
  `val` 周边材质上得到一致的 anomaly 轨迹。但旧 nominal/anomaly pair 的周边摩擦不匹配，
  因而旧三族审计只证明外观资产不同与 anomaly 在换外观后可重复，不能证明 O2 因果效应。
  修正后的控制 lane 与 anomaly 使用相同 4+3 XY 分段和相同 0.8/0.6 周边；nominal 中心平坦
  且不施加土壤力，anomaly 中心才变为坡面/软床并启用剪切。单场景审计已证明接触前 125 行
  完全相同，首次分歧不早于首次区域承重接触。修正协议现已在三套冻结外观上全部通过；但
  三者共享同一 metric geometry 和轨迹；现已另有一个预冻结 G03 独立几何 held-out 通过，但
  仍须补跨几何 multi-dose 与多个独立 soil/substrate/scene family；该结果也不能替代土槽参数
  标定或 A0–A7 任务结果。

### O3 触发坍塌

- 触发量由累计足端法向冲量、压力或损伤状态决定，而非仅由 CoM dwell time 决定。
- 触发后移除/下沉碰撞片或切换破损拓扑；摩擦变化只能是伴随效应。
- 当前固定 25 N·s 实测足端冲量触发，按 0/1/4/13 个失效支撑 cell 构成四档，
  5-seed 中失效面积、跌倒结局和任务进度亏损均保持有序。severe 单点在同一 trigger step
  关闭并隐藏 13 个失效 cell，生成 26 个可见、无碰撞、带 source-cell provenance 的破裂面。
  最新 forest matched-topology pair 进一步要求两条 lane 具有相同的 4×4 零缝隙支撑网格和
  nominal friction；前 141 行完全相同，第 142 步首次分歧与实测冲量触发相同，并同步变形
  连续 PBR 网格。修正协议已在三套冻结外观上全部通过，并另在一个预冻结 G03 独立几何上
  held-out 通过；下一步扩展跨几何多阈值、多个独立板材/scene、非规则断裂几何和自然驶入
  realization；前置 RGB 对足下坍塌的可观测性也必须单独评估。

### O4 adhesion / external attachment

- 用足端或腿部的明确 attachment point 建立约束，不再仅对躯干施加反向力。
- canonical 上位类使用 `external_attachment`，子类型至少区分 `adhesive_contact` 和 `tether_entanglement`。
- 记录 grip force、penetration/extension、peel 状态、attachment point、break event 和释放能量。
- 受控核心可保留严格匹配构造；真实感层只要求在归因窗口的可观测分布上通过 C2ST/等价检验。
- 当前室内真实感门以每足 applied-force cap 0/8/16/32 N 为单因素剂量，五个配对命令条件中
  均严格读回；作用冲量均值为 0/17.0/25.1/31.2 N·s，后退恢复距离为
  0.513/0.544/0.168/0 m，跌倒数为 0/0/2/5。所有异常均在命名足端 attach，并以
  peel 或 break 释放；旧 trunk-resistance 容器始终为空，deep reset 后无残留。
- 该门仍只有一个室内 scene 与一个 adhesion 外观，属于工程剂量证据；正式 corpus 前必须增加
  独立 scene/appearance/material realization，并接入真实可剥离 fixture 标定。
- runtime-v5 垂直切片已闭环一组冻结 schedule 的 nominal/anomaly pair：采用 base-fixed
  Go2-front RTX，每条 physical trajectory 在同状态下渲染 3 个计划内 PBR appearance view；
  nominal/anomaly 分别为 74×3=222/55×3=165 帧 RGB、366/273 条 19 维 raw proprio 和逐步
  privileged telemetry；异常侧发生 2 次足端 attach，峰值单足力
  26.05 N。两条的哈希、时钟、场景/外观/几何、语义像素和算子读回均通过独立 artifact QA。
  相机工程 profile c 由过度俯视的 0.22 rad 修正为 0.10 rad，并在远端加入工具板、门和
  设备上下文；复采后场景上下文像素门通过。该切片仍是规则几何为主的工程 smoke，且相机
  外参未做真实 fixture 标定，因此 runtime-v5 强制标为 `validated_smoke`、
  `evaluation_eligible=false`，不得进入 A0–A7。

### O5 负载

- 生成并显示真实负载物体，采用固定关节安装到机身。
- 使用实测质量、质心、惯量和安装位姿；不采用 VLM 估计值作为真值。
- 当前 6 kg 偏置 cuboid vertical slice 已通过：PhysX base mass、CoM 与完整惯量张量和解析
  刚体合成一致，可见负载随 base 六自由度同步且不重复碰撞质量，足端总承载随附加重量上升，
  deep reset 可精确恢复。该实现是“等效刚性合成质量属性 + 无碰撞可见体”，不是独立 fixed-joint
  刚体。
- 当前中心安装 2/4/6 kg 质量剂量已完成 5-seed 配对门：质量、CoM 位移、惯量迹与足端
  承载均严格有序，mild/moderate 不跌倒，severe 为 2/5 跌倒。偏置安装的首轮饱和失败
  manifest 被保留，未用于正式结论。
- 关节级 telemetry 已接通：nominal/2/4/6 kg 的 torque-utilization P90 为
  0.302/0.322/0.454/0.526，质量剂量在 5/5 seeds 严格排序；6 kg 平均 actuator-binding
  fraction 为 4.09%，机械功率约 57.7 W。旧 `effort_ratio` 因先跨关节求均值再过 70%
  阈值而保持 0，只作为兼容字段，不再承担 O5↔O10 证据。
- 后续仍需负载外观/安装位姿 realization、恢复后果与真实称重标定。

### O6 外力冲击

- 使用动态物体碰撞或有限时长、带作用点的外力。
- 理想速度冲量保留为受控单元测试，不作为 realistic headline。
- 当前有限外力实现已完成：不再写 root velocity，而是在 0.12 s（6 个 50 Hz 控制步、
  每步 4 个 200 Hz physics substeps）内向 base 施加 world-frame PhysX wrench。线冲量剂量为
  nominal/2/6/12 N·s，作用点为 body-frame `[0,-0.085,0.075]` m；实时 USD 读出的 Go2
  base 包围盒为 x `[-0.128,0.332]`、y `[-0.097,0.097]`、z `[-0.097,0.089]` m，故作用点
  位于真实机身表面范围内，而不是空间中的虚构杠杆点。
- 五个配对运动条件下，输入线冲量全部完整回读，偏置点产生预期负滚转角冲量；横向速度增量
  均值为 0.084/0.240/0.573/0.966 m/s（nominal/mild/moderate/severe），5/5 条件严格排序；
  滚转响应在容差内 5/5 有序，severe 最终横向偏差均值 0.711 m，mild 0/5 跌倒。
- 该结果是 mechanics engineering gate，不是 corpus 统计。正式 benchmark 仍需多场景、±方向、
  不同 gait phase、持续时间与作用点 realization，并加入带质量/顺应性的动态碰撞体及真实摆锤标定。

### O7 视觉—物理错配

- 同一外观构造正常/异常物理；同一物理构造多外观。
- 若使用 depth corruption，必须作用于真实相机输出及其时间戳，而不是只修改程序化语义区域。
- 当前 Go2-front RTX vertical slice 已完成：perception camera 不再使用机器人后上方的
  自动 look-at 视角，而是按 base 六自由度刚性同步 `[0.335,0,0.065]` m、向下 0.52 rad
  的工程外参；显式启用 Isaac Lab `update_latest_camera_pose`，避免 RGB/depth 已刷新但
  `pos_w/quat_w_world` 仍停留在上一场景的错误。最终平移读回误差为 0，最大姿态误差
  0.00098 rad；正式论文前仍须用真实 Go2 fixture 标定替换工程外参。
- O7 的 PBR、语义 mask、碰撞摩擦统一落在同一个 USD prim；depth fault 作用于真实 RTX
  `distance_to_image_plane` 输出的指定 mask，保留 frame/source/read timestamp、输入/输出
  哈希与实际偏置。5 套独立 AmbientCG PBR × 4 档配对门中，每帧 28,708–28,743 个像素
  受控，RGB 在 pipeline 前后逐帧 bitwise identical，反事实 raw depth MAE 最大
  `8.98e-18 m`、mask IoU=1.0；0/0.08/0.18/0.32 m 偏置造成平均地图质心位移
  0/0.0649/0.1458/0.2594 m，5/5 严格排序，同时 PhysX μd 0.60/0.30/0.13/0.06 全量回读。
- 跨独立重渲染的 RGB MAE 最大 7.20/255，低于预注册的 10/255 工程阈值，但不把该
  随机渲染差异当作“bitwise 配对”；严格无泄漏合同由同一已采集 RTX 帧进入 depth pipeline
  前后的哈希相等承担。该 gate 仍非正式 corpus episode，也未证明恢复策略后果。

### O8 视觉弱障碍

- 使用透明亚克力、玻璃门、细线、被遮挡低杆等真实存在的碰撞几何。
- 完全不可见的“魔法墙”只保留为受控边界测试。
- 当前透明面板 vertical slice 已完成：固定 0.036×2.0×0.38 m 碰撞几何和 IOR=1.49 的
  OmniGlass 材质，只以 shader 霜化粗糙度 0.12/0.06/0.02 定义 mild/moderate/severe
  光学歧义；不再错误地用面板高度作为线性风险等级。每个异常都有同一 prim path、尺寸、
  材质路径和 shader 参数，仅 `collisionEnabled` 从 false 切到 true 的 nominal counterfactual。
- 五个配对运动条件下，15 个 collision-off nominal 全部穿越，15 个 collision-on anomaly
  全部阻挡且无跌倒；三档物理响应按设计精确不变：平均进度亏损 1.368 m、停滞增加 3.86 s、
  tracking-error 增量 0.324 m/s。由此可把视觉难度与物理后果分开归因。
- 当前仍是 USD/shader/physics 合同门；需以 Go2 前置 RTX 相机补像素可见性、collision-off/on
  接触前成对图像等价、多室内背景/反射/污渍/边框 realization。低杆因呈非单调绊倒风险，
  保留为独立 geometry subtype，不进入透明度剂量排序。

### O9 托底/高中心

- 使用圆木、路沿、托盘边、石脊等多种实际碰撞几何。
- ground truth 来自真实足接触比例和腹部接触，不由手工 `residual_support` 直接指定。
- severe 托盘中梁 vertical slice 已通过：300 个连续控制步有真实 base 承重接触。
- 以真实 base collider 约 0.351 m 净空为中心的 350/355/360 mm 单因素高度扫描已通过
  5-seed 工程门：连续托腹 5/290/300 steps，显示临界非线性，而非简单线性剂量关系。
  后续仍需自然驶入、多种圆木/路沿/石脊 realization 和真实 fixture 标定。

### O10 执行器降额

- 使用关节相关的 torque-speed envelope，并建模电压、温度和热状态。
- 支持全机、单腿和单关节降额。
- 当前全机 endpoint-cap vertical slice 已完成：continuous effort cap 与 DCMotor 私有
  stall/saturation torque 同步按 1.00/0.75/0.50/0.25 缩放，并重新计算 torque-speed 曲线拐点；
  PhysX 读回为 23.5/17.625/11.75/5.875 N·m。
- 五个 seed 分别冻结不同的轻微 `vx/vy/yaw` 需求，同一需求在四档间配对复用，消除了固定命令
  造成的伪重复。利用率 P90 为 0.308/0.411/0.616/0.965，5/5 严格排序；severe 的
  binding fraction 为 24.71%，进度亏损 1.884 m（95% bootstrap CI 1.772–2.016）。
- 该结果只支持“endpoint actuator-cap 可控且可观测”，不能声称已经建模热衰减；正式 corpus 前
  仍需 time-varying voltage/temperature/thermal ramp、局部关节/单腿降额和真实电机标定。

### O11 观测偏置

- 在原始 IMU/里程计层注入 bias、random walk、温漂、延迟和丢帧，再通过统一状态估计器。
- 参数来自真实静置/运动日志或明确的工程上限。
- 当前 raw-sensor vertical slice 已完成：Isaac Lab 的 base-mounted IMU 提供原始
  `projected_gravity_b/angular_velocity/linear_acceleration`，articulation odometry 以带时间戳
  的 packet 进入同一个状态估计器；故障在估计前注入，monitor/planner 看见退化量，而
  低层策略在本工程门中刻意保持真值输入，以隔离传感链路因果效应。
- 五个配对运动条件下，tilt bias/random walk/odometry latency 剂量为
  nominal=`0/0/0 ms`、mild=`0.08 rad/0.004 rad·s^-1/2/40 ms`、
  moderate=`0.18/0.010/100 ms`、severe=`0.35/0.020/200 ms`。原始倾斜误差均值
  0/0.080/0.179/0.348 rad、估计后误差 0/0.044/0.139/0.306 rad，5/5 seed 严格排序；
  实测 odometry age 精确为 0/40/100/200 ms，且四档真值轨迹哈希逐 seed 完全相同。
- 速度延迟误差在 100–200 ms 区间因轨迹局部平稳而饱和，不作严格单调主张。dropout 的
  hold-last 与 seeded replay 已由纯单元测试验证，但尚未作为 Isaac 多档工程门；正式 corpus
  仍需真实 Go2 静置/运动日志标定 bias/noise/温漂/丢帧分布，并让恢复控制也消费故障观测。

## 8. A0–A7 增强方式

### A0：基础设施与数据合同

- 新增 scene/asset/physics/camera provenance、内容哈希和 license manifest。
- 重做 body-fixed camera、多帧同步和 USD determinism gate。
- 增加碰撞稳定性、可通行性、重复纹理和数据泄漏审计。

### A1：双向必要性与可辨识性

- 保留旧 controlled causal core。
- 在修正后的 O2/O4 足端模型上，通过参数优化匹配归因窗口统计，并用 fresh-app/deep-reset C2ST 重新认证。
- 视觉分离必须跨外观家族成立，不能依赖固定颜色。

### A2：冲突学习主结果

- 在跨 scene family、material family 的数据上重新训练相同 baseline roster。
- 报告 scene-heldout、material-heldout 和 combined heldout。

### A3：双向 conflict battery

- 在生活、生产、野外三个域上重复 T1–T5。
- 增加 O7 reverse、O8 transparent/thin-obstacle 和相机扰动测试。

### A4：归因到恢复后果

- 在修正物理上重跑关键 label-swap crux cells。
- 全矩阵可继续使用受控核心；headline 至少包含 realistic O2/O4/O5/O8/O9 的实际恢复后果。

### A5：泛化与弃权

- 增加 scene、material、lighting、camera-profile 和 physical-realization OOD。
- 加入真实 fixture heldout 作为最终 anchor；不把该层编号为 A8。

### A6：物理严重度边界

- 严重度使用可测参数：摩擦、沉陷/剪切参数、断裂力、负载惯量、执行器电压/温度。
- 报告参数后验和区间，不只给插值点估计。

### A7：方法与真实性消融

- 默认网格 vs EmbodiedGen/真实感场景。
- 程序贴图 vs 扫描 PBR。
- world-follow vs body-fixed camera。
- 旧躯干等效力 vs 足端/真实约束物理。
- 纹理变化、几何变化、物理变化分别消融。

## 9. 数据规模与 split

### 9.1 Pilot

- 3 域 × 每域 3 个独立场景族 × 11 算子 × 2 严重度 × 每场景 1 seed = 198 个异常
  episode。pilot 的预算优先用于 scene-cluster 覆盖；同一场景内的 5-seed 方差由 full corpus
  承担。
- 每个异常配置配一个正常/反事实控制，总计 396 个独立 physical episode；每条有 3 个同步
  外观视图，因此产生 1,188 条 RGB view sequence，但统计单位仍是 396 个 episode/198 个
  counterfactual group，不能把外观复渲染当独立样本扩大样本量。

### 9.2 Full corpus

- 11 算子 × 3 域 × 3 scene family × 2 physical realization × 3 severity × 5 seeds = 2,970 个异常 episode。
- 加一对一反事实控制后为 5,940 个 physical episode、17,820 条同步 RGB view sequence。

### 9.3 分组规则

- train/val/test 按 scene family、material family、physical realization、camera profile、real fixture 分组。
- 同一 episode、同一生成资产的不同帧不得跨 split。
- 主要统计单位为独立 episode/scene cluster，不把 frame 当独立样本。

## 10. 真实 Go2 锚点

### 10.1 最小规模

- 11 算子 × 2 severity × 10 repeats，约 220 个真实 episode。
- 高风险算子使用安全吊架、浅坑、泡沫破断片和低能量碰撞装置。

### 10.2 建议 fixture

- O1：低摩擦板/可控湿面。
- O2：泡沫、沙箱、松散颗粒箱。
- O3：浅支撑上的可替换泡沫/薄板模块。
- O4：弹性 tether、可剥离垫、可控缆线。
- O5：已知质量和安装位姿的负载。
- O6：摆锤软包或低能量推力装置。
- O7：外观—物理交换的地面样片。
- O8：透明板、细线、遮挡低杆。
- O9：可调高度圆杆/路沿模块。
- O10：软件力矩限制/电压模拟。
- O11：软件传感器偏置和延迟注入。

### 10.3 sim2real 判据

- 比较触发时间、速度亏损、足滑、机身姿态、接触状态、约束力和恢复结果的 sim/real 分布。
- 校准使用 held-in fixture，最终结果使用 heldout fixture/材质。
- 报告 C2ST、Wasserstein/MMD、时序误差和方法排序是否在 sim/real 间保持。

## 11. 统计、质量门和可复现性

### 11.1 场景质量门

- 无初始穿插、无无意悬浮、无双重地面碰撞。
- Go2 起点、目标点和计划路径满足净空。
- 静态场景经过重力 settle 后保持稳定。
- 视觉 mesh 与 collision mesh 的关键行走边界偏差受控并记录。

### 11.2 算子质量门

- 作用位置、时刻、严重度和状态转换可从 privileged telemetry 回读。
- 非目标区域与 nominal 对照统计一致。
- 同一算子同一 seed 在 fresh-app/deep-reset 下可复现。
- 每个 physical realization 均有单元测试、可视化和至少一种反事实控制。
- 每个 physical episode 的 appearance views 必须在同状态/同时间戳下采集；逐视图图像差异、
  材质读回和 provenance 通过 runtime-v5，且所有视图只允许改变渲染属性。

### 11.3 论文统计

- 分类：macro-F1、balanced accuracy、per-cause/per-domain CI。
- 选择性恢复：risk–coverage、expected recovery cost、released precision。
- 后果：lane/scene-grouped bootstrap 和 paired tests。
- sim2real：分布距离、真实盲测成功率和方法排序一致性。

## 12. 首个垂直切片：室内 + adhesion

### 12.1 场景叙事

场景为真实尺度的室内工作室/生活空间：木质或复合地面、墙体、踢脚线、货架/桌柜、收纳箱、软质地毯、玻璃/金属/木质材料和自然照明。Go2 从房间入口向前导航，在路径中部接触一块外观自然、非纯色矩形的可剥离地面保护膜/粘附垫；粘附通过明确的足端 attachment point 产生，并允许后退剥离或达到阈值后断裂。

### 12.2 物理定义

- 上位类别：`external_attachment`。
- 子类型：`adhesive_contact`。
- activation：任一足端进入 irregular adhesion mask 且接触成立。
- attachment：记录触发足、世界锚点、局部锚点和初始自由长度。
- loading：沿足端—锚点方向的弹簧/阻尼或受限力曲线；不得直接以 CoM 位置作为唯一 penetration。
- peel：反向退出时使用较低 peel force。
- break：达到可配置断裂力/能量后释放，并记录 break event。
- control：相同外观、无 attachment；相同 attachment、至少两种外观。

### 12.3 相机与画面

- 主画面来自 Go2 body-fixed front RTX camera。
- 审查材料同时提供：第一视角 RGB、第三人称场景图、顶视图/算子 mask、接触/attachment 可视化。
- 第一视角不得显示调试色块、网格地面或标签文字。

### 12.4 Demo 验收标准

- 场景视觉达到可用于 ICRA teaser/demo 的室内真实感，无默认网格和纯色矩形异常区。
- Go2 在 nominal 对照中能通过；在 adhesion 中出现可重复速度亏损/约束力。
- attachment 发生在足端或腿部明确位置，不是匿名躯干反向力。
- 后退/剥离与继续前进具有可解释的不同后果。
- 输出完整 manifest、配置、seed、视频/关键帧、遥测曲线和运行命令。
- 所有数字只作为 demo/工程门，未经 A0 determinism 和正式统计前不进入论文 headline。

## 13. 实施阶段与交付物

### Phase P0：基础合同

- 新增本计划。
- 冻结 EmbodiedGen commit、依赖和资产许可证。
- 实现 scene manifest schema、USD layer contract 和静态审计器。
- 修复 body-fixed multi-frame RTX 路径。

### Phase P1：室内 adhesion 垂直切片

- 生成/导入一个室内场景。
- 编译 visual/collision/physics/operator USD 层。
- 实现足端 attachment 状态机与 privileged telemetry。
- 生成 nominal/adhesion/reverse-peel 三条审查轨迹。

### Phase P2：Pilot 扩展

- 把 scene compiler 扩展到三域。
- 修正全部 11 个算子。
- 生成约 396 episode pilot，完成泄漏、物理和 camera QA。
- 当前进度：冻结 protocol 下已完成 O4/workstudio 的 4/396 条、2/198 个完整反事实组；
  覆盖 1/11 operator、1/9 scene family、1/3 domain 和 1/3 camera profile，不能以此替代
  剩余 392 条。
- O1/O2 的判据已在 G06/G07 两个未见 photographic family + 新物理几何上按冻结合同 4/4
  确认；O3 未改判据为 1/2，G06 的强陷落未触发二元 fall。该 5/6 只属于 operator development，
  所有产物保持 `counts_as_a0_a7_evidence=false`。下一工程路径不是继续堆 forest demo，而是实现
  EmbodiedGen route-centric operator adapter，使 O1–O3 与其余算子共享同一 registry-bound
  scene/schedule/runtime-v5 数据合同，然后按 9 scene family 采集 396 条 pilot。
- 当前 adapter calibration 已在 `indoor_kitchen_31` nominal-only 上通过：新增显式 floor
  `PhysicsMaterialAPI`、2635 顶点的 render-only PBR operator surface，并从原 admission 的
  waypoint 2 完成 1.561 m。四个失败版本均保留。该结果只证明接口可行；动态 bridge 必须由
  native v3 collector 替换，且至少再用两个 operator-naive scene family 做冻结确认，才能写入
  operator capability registry。
- O4 的新 realistic acquisition 已从“场景生成计划”推进到 v25 的 6 个 admitted scene/4 family
  冻结 cohort；v26 已对全部场景执行一次性 matched-state Continue/Backstep。实验过程与因果
  对照完整，但主结果只有 3/6。v27 已在三条独立、已暴露 development scene 上完成
  phase-conditioned/closed-loop architecture 验证并封存 3/3；三场都走 maximum-dwell，因此仍
  不能称为紧急触发的跨场景确认。其后 v30 以 nominal-only gate 冻结新的 6 scene/4 family，
  v31 原样一次性确认得到 artifact/causal integrity 6/6、严格结果 1/6，否定了 v27 的跨场景
  泛化。失败由两个 attachment 后分支过晚、两个 120-step 上限后未完成 active-feet clearance，
  以及两个只发生在 Continue 倒地 outcome 的错误通用纹理门组成（类别有重叠）。v31 不得重跑。
  v33 随后冻结 candidate 113–122 首个合格前缀中的 5 scene/5 family；v32 在完整五场一次执行，
  artifact/causal integrity 5/5、预注册严格结果 4/5，且五例均为 Continue 分支后跌倒、Backstep
  安全完成 clearance。唯一负例来自 Continue 已进入物理失稳后侧滑越过旧的全时域 route budget，
  不是 Backstep 失败或决策前路线混杂；该结果仍封存为 4/5，不得事后改判。下一门保持 v32
  动作/物理不变，在 candidate 137+ 新 operator-blind cohort 前瞻性冻结 phase-aware containment。
  v34 acquisition 严格处理 candidate 137–157 的首个合格前缀，得到 6 scene/4 family，158–160
  永久未触碰；一次性 held-out 执行 integrity 6/6、严格 5/6。五例 Continue 跌倒而 Backstep
  安全后撤 0.236–0.275 m；唯一负例 Office 156 是真实动作失败：Backstep 后仍前侵 0.219 m，
  首个 FL 粘连未 peel，随后 FR 再粘附并跌倒。该六场不得重跑。v35 开发候选保持物理算子、
  分支时刻和基础命令不变，只在物理紧急态或分支后前侵 2.5 cm 时锁存更强的可执行后撤命令；
  它必须经过全新 development→heldout 场景链，当前不构成正向结论。完整分析位于
  `docs/datasets/kinofail_o4_v31_failure_analysis.md` 与
  `docs/datasets/kinofail_o4_v32_development_analysis.md`；通过 operator gate 后仍必须进入
  registry-bound corpus，而不能把该 gate 当作 A0–A7 完成。

### Phase P3：A0–A7 realistic replication

- 按 A0–A7 顺序重新运行、冻结并报告。
- 旧 controlled core 与新 realistic replication 并列，不覆盖旧证据。
- 完成状态以 `outputs/eval/realistic_a0_a7/readiness_audit.json` 为准；旧 A0–A7 的任何满分、
  哈希或统计结果都不能让新实验自动通过。

| 实验 | realistic 承重对象 | 冻结完成门 | 当前状态 |
|---|---|---|---|
| A0 | 真实 registry-bound corpus、同步、determinism、泄漏与 texture swap | 396/396 episode、198/198 pair、11 算子、9 场景、3 域、3 camera profile，`publication_freeze_ready=true` | **未通过：4/396、2/198、1/11、1/9、1/3、1/3** |
| A1 | 修正后 O2/O4 的 fresh/deep-reset matched construct | 至少 9 scene clusters；proprio C2ST cluster CI 全落入 chance±0.10；vision CI 下界≥0.80 | 未采齐 |
| A2 | scene/material/combined-heldout conflict learning | 5 train seeds；combined balanced accuracy≥0.80；ours−unshaped cluster CI 下界>0；如未胜所有 baseline 则不得写 champion | 无 realistic predictions |
| A3 | 3 域 T1–T5 bidirectional battery | 5/5 seeds 逐个过门；worst≥0.90、macro≥0.95、至少 33 independent scene/physics clusters | 无 realistic report |
| A4 | 修正物理上的 actual-action consequence | O2/O4/O5/O8/O9 headline；每 action cell≥10 seeds；scene-cluster paired bootstrap；保留弱格 | 工程门不等于 consequence matrix |
| A5 | scene/material/light/camera/realization heldout 与 selective recovery | selective−safe cluster CI 上界<0；released precision≥0.95；texture-swap 0.95/0.95/JSD≤0.05 | 无 realistic predictions |
| A6 | 可测物理参数的 severity/boundary | ≥5 算子参数族、每族≥4 档×5 seeds、区间估计；未识别时只报 bracket | 只有分散工程剂量门 |
| A7 | realism/camera/local-physics/texture-swap 消融 | 6 组预登记消融、3 train seeds、paired scene-cluster statistics | 未运行 |

以上是 P3 的完成定义而不是目标缩水。若真实感下降导致旧 structured 方法失去满分，应先排除
同步/相机/标签错误，然后如实训练和比较改进方法；不得复用旧输出或删掉困难 test cells。

### Phase P4：真实锚点

- 标定参数、采集真实 fixture、完成 heldout 对比。
- 达标后再将术语从 sim2real-oriented 提升为 validated sim2real。

## 14. 风险与回退

- **EmbodiedGen 场景生成失败或依赖过重：** 使用其已生成 USD/URDF 资产；若仍不可得，先用经过来源审计的室内 USD 完成 scene compiler 和物理垂直切片，但不得称为 EmbodiedGen-generated。
- **生成碰撞代理不可靠：** 保留视觉资产，重新制作简化碰撞 mesh。
- **足端动态约束在 Isaac 中不稳定：** 先使用明确足端受力、局部锚点和状态机的稳定实现；同时保留 D6 joint 参考实现做慢速验证。任何回退都必须在 manifest 中标记 fidelity level。
- **真实感导致旧模型性能下降：** 视为需要测量的科学结果，不调标签或偷偷缩小 test；先排除相机/数据管线错误，再报告差距。
- **生成纹理形成标签捷径：** 启动同外观异物理、同物理异外观反事实和 scene/material heldout；不通过则不进入正式数据。

## 15. 完成定义

### 不可降级的 completion invariant

本项目的交付对象不是一组“看起来更真实”的场景截图，也不是 11 个算子各自通过一次工程门，
而是**在新版 realistic Kino-Fail 上重新建立完整的 A0–A7 发表证据链**。旧 controlled-core
A0–A7 只用于冻结研究问题、协议和效应量参照，不能满足新版实验的任何完成门。正式审计必须
逐级校验：新版 registry-bound corpus → 新版 A0 证书 → 基于该 corpus 的新训练与推理 →
绑定同一 A0 证据包和 predictions 的新版 A1–A7 输出。任一环缺失时，ready 数保持未完成；
场景、operator demo、calibration 或旧满分结果均不得代替这一链条。

本计划整体完成需要同时满足：

1. 三域、11 算子、约 5,940 episode 的分组语料与 manifest 完整。
2. body-fixed Go2 相机和同步多帧数据通过 A0 determinism/时间一致性门。
3. 11 个算子的 realistic 实现均通过局部物理 QA 和反事实控制。
4. A0–A7 realistic replication 全部完成并保留可追溯配置、seed、commit 和统计。
5. 至少约 220 个真实 Go2 fixture episode 完成 heldout 验证。
6. 论文明确区分 controlled causal、realistic simulation 和 real anchor 三层证据。
