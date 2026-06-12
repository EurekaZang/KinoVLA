# Kino-VLA v2.0: 闭环具身反思流基准 (Closed-Loop Embodied Reflection)

这是针对系统论文/架构讨论稿的 **v2.0 重构版本**，目标投稿 RSS / CoRL / ICRA / IROS。相比 v1.2，本版本完成了五项结构性升级：(1) 与自适应控制（RMA 一系）划清学术边界，将卖点收敛至"必须语义介入"的失效子空间；(2) 扩充恢复原语库，建立"归因→恢复策略"的多对多映射以论证语义推理的不可替代性；(3) 以仿真特权物理真值（Privileged Ground Truth）作为 Kino-Tokens 与 Hindsight CoT 的双重监督锚点；(4) 以控制屏障函数（CBF）将 Safety Shield 形式化为"LLM 提议、CBF 裁决"的带保证架构；(5) 将拓扑标记升级为持久化的**语义可通行性地图（Semantic Traversability Map）**，并以组合泛化实验取代"正交基"声明。

## 1. 全景技术架构拓扑 (System Framework)

系统涵盖"离线仿真训练"与"在线实机闭环"的完整架构：

- **离线仿真与训练平台 (Offline)：**
    - **环境与数据：** 基于 Isaac Lab 搭建动力学仿真环境，采集高频状态序列与**特权物理真值**（摩擦系数、接触刚度、附着弹性参数、负载质量与偏移、执行器衰减率）；构建按 4 条仿真机制轴组织、含 11 个连续参数化失效算子与 5 个基准套件（Cal / Sem / Comp / Bound / OOD）的 **Kino-Fail v2 算子库**（见第 8 节）。
    - **Sim-to-Real 桥接：** 引入 Actuator Network 建模电机/齿轮箱非线性，并对接触信号（力尖峰、穿透伪影）进行域随机化，确保高频通道的迁移性。
    - **预训练：** 以特权物理参数回归/蒸馏为主监督、物理-文本对比学习为辅监督的特征对齐预训练框架。
- **在线部署闭环 (Online Runtime Loop，分为五层)：**
    - **感知与异常监控层 (01)：** 以 Unitree Go2 为平台。感知流分为 **1000Hz 本体传感器**（电机电流估计力矩、关节角度、IMU 姿态；注：Go2 标准版无真实足底力传感器，足端力由电流反演并经 Actuator Network 校准）与 **30Hz 外感传感器**（RGB-D）。内置 **Kino-Monitor** 高频计算速度跟踪误差与滑移分数，作为 Reflex Loop 触发器；其检测阈值在真机上以 **ROC 曲线（误报率-漏报率权衡）** 标定并在论文中报告。
    - **Kino-Tokens 提取层 (02)：** 500ms 滑动窗口 + 1D-CNN + Perceiver Resampler，将高频物理序列压缩为定长 `Kino-Tokens`，投影至 VLM Embedding 空间。Latent 以**特权物理量预测误差**为可解释性度量（见第 5 节）。
    - **语义反思与拓扑记忆层 (03)：** ~1Hz 的 **VLA Recovery Planner** 接收文本、视觉、Kino-Tokens 三模态输入，输出结构化恢复原语；同时维护并读写一张持久化的**语义可通行性地图**（open-vocab 分割 + 深度反投影 + costmap，见第 8 节），物理失效作为可通行性标签的在线修正源。
    - **安全裁决与原语编译层 (04)：** **Primitive Compiler + CBF Safety Shield**。VLA 输出的任意原语参数先被投影至由控制屏障函数定义的安全不变集内（姿态角、速度、扭矩约束），再降级编译为 Sport Client 指令并挂载终止/中断条件，提供"无论 LLM 输出什么，系统不进入失稳集"的形式保证（见第 7 节）。
    - **控制与执行反射层 (05)：** 双闭环执行网络。**高频身体反射 (Reflex Loop)** 绕过 LLM，由 Kino-Monitor 触发毫秒级自稳；**低频语义恢复 (Reflection Loop)** 顺序执行原语并回传 Execution Report。两环之间以**显式延迟预算表**（异常发生 → Reflex 介入 ms 级 → Kino 窗口填满 500ms → VLA 输出 ~1s → 原语生效）论证 Reflex 可维持的"安全停留时间"覆盖 VLA 推理延迟——这是双闭环架构成立的形式前提。

## 2. 核心动机 (Core Motivation)

保留"视觉错觉 vs 动力学真相"的根本矛盾，但将论证收紧一层：

- **视觉欺骗性：** 纯视觉 VLA 受采样率、遮挡与浅层特征欺骗（反光冰面、伪装重物纸箱），导致盲目自信的灾难性决策。
- **动力学真相：** 物理真相隐藏在 1000Hz 底层交互中（打滑、扭矩饱和、异常接触力），是触发空间重规划的核心信号。
- **【新】不可替代性论证 (Irreplaceability Argument)：** 自适应控制（RMA / Lee et al. / DreamWaQ 一系）已能从本体感受历史中隐式编码环境动力学并在连续参数空间内自适应（冰面降速、负载补偿）。因此本工作**不主张**"感知动力学异常"本身的新颖性，而主张：存在一类失效，其正确恢复需要**离散的、拓扑级的、策略级的决策**——后退脱离缠绕而非继续适应、将整片区域标记为不可通行而非局部调参、依据视觉归因在"相反的恢复策略"之间二选一。这类决策超出任何连续自适应控制器的假设空间，构成语义层的不可替代域（见第 3 节的失效二分法）。
- **具身类比：** 脊髓反射处理毫秒级失稳（对应自适应控制可解域），大脑反思处理"此路不通、换条路走"的认知级决策（对应语义介入必需域）。Kino-VLA 的贡献在于打通两者之间缺失的上行通道。

## 2.5. 学术前沿定位与核心科学问题 (Positioning & Core Scientific Question)

将现有路线梳理为**六条**（v1.2 的五条 + 必须正面交锋的失效归因 LLM 路线），并以"失效可恢复性二分法"重新定义核心科学问题。

### 1) 六条路线的物理瓶颈

1. **通用端到端动作生成 (RT-2, OpenVLA)：** 开环/低频映射，缺乏物理受力反馈通道，突发扰动下盲目执行预定动作导致失稳。
2. **视觉闭环反思 VLA (COME-robot, SC-VLA, CLOVER)：** 受相机采样率（10-30Hz）与遮挡限制，对摩擦系数突降、隐形超载、电机过热等**显性失败尚未发生**的隐性物理状态变化无法提前介入。
3. **几何运动学 VLA (KineVLA)：** 运动学不考虑力与质量，无法验证轨迹的动力学可行性（Kinodynamic Feasibility），冰面/重载下强行执行几何合理的轨迹必然力学崩溃。
4. **机械臂力/触觉 VLA (ForceVLA, AT-VLA)：** 末端准静态局部力反馈，无法推广至浮动基座、欠驱动、多足交替接触的整机动力学平衡问题。
5. **足式 VLA 导航 (NaVILA, QuadrupedGPT, Helpful DoggyBot)：** Locomotion 被视作单向黑盒 API，高层收不到底层异常反馈，无法归因故障并生成针对性脱困策略。
6. **【新增】多模态失效归因与 LLM 重规划 (REFLECT, AHA)：** 该路线与本工作最近，必须逐条对齐差异。REFLECT 以分层多模态摘要（含听觉）做事后失效归因；AHA 训练 VLM 做操作失效检测。_瓶颈：_ (a) 失效信号源为低频视觉/听觉事件，缺乏 1000Hz 本体动力学通道，对足式平台的隐性力学失效（滑移率、扭矩饱和）不可见；(b) 归因发生在失败**之后**（摔倒已成事实），缺乏毫秒级反射层在归因期间维持系统存活；(c) 面向机械臂操作任务，未触及欠驱动足式平台的全身动力学与可通行性地图修正问题；(d) 注入形式为离散文本摘要，存在量化误差，无连续 latent 通道。

### 2) 【新】核心科学问题：失效可恢复性二分法 (The Recoverability Dichotomy)

本工作明确提出并以实验回答以下问题：**"哪些物理失效是低层自适应控制原则上可解的，哪些必须语义介入？"** 据此将 Kino-Fail 全部场景做二分标注：

- **A 类 — 低层可恢复 (Low-level Recoverable)：** 连续参数自适应即可存活。例：均匀低摩擦（降速+降重心）、静态负载偏移（力矩补偿）、平缓粘性地面（步态频率调整）。**实验承诺：** 在 A 类场景中报告 RMA 式自适应 baseline 的成功率，预期其表现与 Kino-VLA 持平——本工作不在 A 类上声明贡献，反而以此自证 baseline 实现的公允性。
- **B 类 — 必须语义介入 (Semantic-intervention Required)：** 正确恢复需要离散决策、拓扑修正或基于视觉归因的策略选择，且**错误归因导致相反的恢复动作**。例：
    - **弹性羁绊（线缆缠绕）：** 自适应控制会持续加大力矩对抗胡克定律弹力，导致能量积累后灾难性弹射；唯一正确策略是反直觉的"后退脱困"。
    - **粘鼠板 vs 泥地（归因模糊对）：** 本体信号同为"抬腿阻力增大"，但粘鼠板需后退+绕行+拓扑标记，泥地可高抬腿步态强行通过——只有融合视觉（黄色板面 vs 棕色泥泞）的因果归因才能选对策略，纯本体的自适应控制器在此**原则上**无法区分。
    - **坍塌地表（薄冰破碎）：** 局部自适应无意义，必须将整片视觉同质区域标记为不可通行并全局重规划。
    - **隐形超载死锁：** 扭矩已饱和，连续空间内无解，必须做出"放弃当前任务分支/呼叫求助/卸载"的策略级决策。
- **科学声明：** Kino-VLA 的全部新颖性声明锚定在 B 类子空间。系统贡献 = 在 A 类上不劣于自适应控制（由 Reflex Loop 与底层 RL 策略保证）+ 在 B 类上显著超越所有无语义层的方案。

### 3) 为什么 Kino-VLA 能够解决 B 类问题？(五步解耦流)

- **第一步 Sense & Monitor (1000Hz)：** Kino-Monitor 高频监听本体状态，实时计算滑移分数与跟踪误差，触发异常警告。
- **第二步 Reflex (1000Hz)：** 脊髓反射式毫秒级自稳（刹车、调刚度、降重心），阻断瞬间失控，为高层推理**争取并保证**安全停留时间（延迟预算表见第 7 节）。
- **第三步 Reflect (500ms 窗口)：** 高频物理数据压缩为 `Kino-Tokens` 投影至 VLM 嵌入空间，latent 受特权物理量蒸馏约束而具备可解释性。
- **第四步 Attribute & Recover (~1Hz)：** VLA 融合视觉先验与 Kino-Tokens 完成失效**因果归因**，在扩充原语库中选择针对性恢复策略（归因→策略为多对多映射），同时写入语义可通行性地图。
- **第五步 Shield & Compile：** CBF Safety Shield 将原语参数投影至安全集，Primitive Compiler 编译为 Sport Client 指令，执行后回传 Execution Report 打通反思重规划环路。

## 3. 模态选择与协同架构 (Architecture & Modality Selection)

核心挑战不变：将 1000Hz 物理信号对齐给 ~1Hz 的 LLM。两条路线与协同方案保留，但监督信号升级：

- **路线 A：离散文本表征。** 解释性强、系统解耦，但量化误差丢失高频非线性特征。作为阶段一 Baseline 与 REFLECT 式注入的对照组。
- **路线 B：隐式连续表征 (`Kino-Tokens`)。** 高保真、低延迟，但黑盒对齐成本高。**v2 改进：** 以特权物理参数蒸馏（第 5 节）取代纯对比学习作为主监督，使 latent 不再是无锚点的黑盒。
- **协同折中方案：** 阶段一部署文本表征验证可行性；阶段二引入连续特征处理文本难以描述的复杂现象（液体晃动、弹性缠绕的相位特征），并以消融实验论证连续 Token 对精细化恢复参数（如 Backstep 距离、限速值）生成质量的增益。

## 4. 端到端提取器：从四个构思到一个原则性选择 (Extractor: Principled Design)

v1.2 平行列举了四个构思；v2 以"监督信号的信息来源"为轴做出原则性取舍：

1. **【主干】特权物理蒸馏 (Privileged Physics Distillation)：** Isaac Lab 中拥有上帝视角物理真值（真实摩擦系数 μ、接触力向量、材料刚度 k、负载质量 m、下陷深度）。提取器以这些特权量为回归目标训练（teacher-student 范式，类比 RMA，但蒸馏目标是对齐到 VLM embedding 空间而非控制策略）。**收益：** (a) latent 天然物理可解释；(b) OOD 的"流形偏离"从玄学变为可量化的物理量预测误差；(c) 为 Hindsight CoT 的真值校验提供锚点。
2. **【辅助】物理-文本对比学习 (Kino-Text Contrastive)：** InfoNCE 拉近物理感受与文本描述，仅作为语义可读性的辅助 loss，不再承担主监督。
3. **【骨架】时序 Tokenizer：** 1D-CNN 捕获局部突变 + Perceiver Resampler 交叉注意力压缩为定长 Token，作为网络骨架保留。
4. **【效率】异常驱动门控 (Anomaly-Gated Injection)：** 稳态时物理 Token 权重置零；Kino-Monitor 报警时激活门控唤醒反思，节省稳态算力。
5. **【消融对照】RL 价值函数引导：** Critic 的 TD-Error 梯度特征降级为消融实验中的对照表征，不再作为并列方案。

## 5. 【新】扩充原语库与"归因→恢复"多对多映射 (Expanded Primitive Library)

v1.2 仅有 `Backstep` 与 `Replan_Waypoint` 两个原语，使整个反思链在功能上退化为二分类器——一个 200 行的规则状态机（异常→后退→写 costmap→重规划）即可在行为上复现。v2 将原语库扩充至 Sport Client 支持的完整策略空间，使归因结果决定**互不相同甚至相反**的恢复路径：

- **原语库 (Primitive Library)：**
    - `Backstep {distance_m}` — 脱离物理陷阱（缠绕、粘附、塌陷边缘）。
    - `Replan_Waypoint {RGB_2D_Point}` — 输出安全目标的 2D 像素坐标，经深度反投影为 3D 航点。
    - `Switch_Gait {mode: high_step | trot | crawl}` — 泥地/碎石→高抬腿；常态→trot；低净空→爬行。
    - `Adjust_Posture {body_height_m, pitch_deg}` — 冰面降重心；托底脱困抬升底盘。
    - `Set_Constraint {max_speed, stiffness}` — 低摩擦限速；柔顺接触降低刚度。
    - `Update_Topology {Region, Status}` — 写入语义可通行性地图（第 8 节）。
    - `Hold_and_Request {reason}` — 物理无解场景（扭矩饱和死锁）的策略级放弃与求助。
- **多对多映射示例（归因错误 ⇒ 恢复失败）：**
    - 泥地 → `Switch_Gait(high_step)` 强行通过；粘鼠板 → `Backstep` + `Update_Topology` + 绕行。（同一本体信号，相反策略）
    - 冰面 → `Set_Constraint(low_speed)` + `Adjust_Posture(low)` 继续直行；薄冰破碎 → 整区 `Update_Topology(Untraversable)` + 全局重规划。（同为低摩擦，决策粒度相反）
    - 线缆缠绕 → `Backstep`（释放弹性势能）；高阻泥浆 → 提高力矩预算继续前进。（同为前进受阻，力学方向相反）
- **论证逻辑：** 当且仅当归因正确性决定恢复成败时，融合视觉的语义推理才不可被规则状态机替代。实验中将显式构造上述"归因模糊对"场景并对比 rule-based FSM baseline（第 12 节）。

## 6. 【新】CBF Safety Shield：LLM 提议、屏障函数裁决 (Formalized Safety Shield)

将 v1.2 中的名词性 "Safety Shield" 形式化为带数学保证的安全层：

- **总览：** VLA 输出的任意原语参数在编译前经 CBF-QP 被最小修改地投影至安全集内；不可行原语直接拒绝并回传拒绝原因码供下一轮反思。形式声明：无论 LLM 输出任何幻觉指令，闭环系统状态不离开安全不变集 $\mathcal{C}$——这将系统从"工程集成"提升为"带形式保证的 VLA 架构"，填补 LLM-in-the-loop 足式控制安全性的空白。完整推导如下。

### 6.1 简化模型与状态定义 (Reduced-Order Model)

采用线性倒立摆模型（LIP）作为裁决层的降阶模型：质心（CoM）保持恒定高度 $z_c$（由当前姿态模式 $\sigma$ 决定），腿部质量与角动量忽略不计。定义自然频率 $\omega = \sqrt{g/z_c}$。状态为质心水平位置与速度 $x = [p^\top, v^\top]^\top \in \mathbb{R}^4$，其中 $p, v \in \mathbb{R}^2$。LIP 动力学：

$$\dot{p} = v, \qquad \dot{v} = \omega^2 (p - u)$$

其中 $u \in \mathbb{R}^2$ 为零力矩点（ZMP），是整机控制器通过足端力分配实际实现的等效控制输入——裁决层在 ZMP 层面过滤，与底层 Sport Client 的实现解耦。

引入发散运动分量（DCM / Capture Point）：

$$\xi = p + \frac{v}{\omega}, \qquad \dot{\xi} = \omega (\xi - u)$$

关键性质：CoM 内动力学 $\dot{p} = \omega(\xi - p)$ 对 $\xi$ 是稳定的（指数收敛），因此只需约束发散分量 $\xi$ 即可约束整个系统的失稳模态。

### 6.2 支撑多边形与安全集 (Support Polygon & Safe Set)

设当前支撑足端水平位置为 ${p_{f,i}}_{i=1}^{N_c}$，支撑多边形为其凸包的半平面交表示：

$$\mathcal{S} = \mathrm{conv}{p_{f,i}} = {y \in \mathbb{R}^2 : a_j^\top y \le b_j,\ j = 1,\dots,M}, \qquad |a_j| = 1$$

其中 $a_j$ 为第 $j$ 条边的单位外法向。以收缩裕度 $\delta > 0$（吸收模型误差与状态估计噪声）定义每条边的屏障函数与安全集：

$$h_j(x) = (b_j - \delta) - a_j^\top \xi(x), \qquad \mathcal{C} = {x : h_j(x) \ge 0,\ \forall j}$$

物理含义即 **0 步可捕获性（0-step capturability）**：只要 Capture Point $\xi$ 位于收缩支撑多边形内，机器人总存在一个不迈步的 ZMP 策略使自身静止，即"原地可刹停、必不跌倒"。

### 6.3 CBF 约束推导 (Barrier Condition)

对 $h_j$ 沿 DCM 动力学求导：

$$\dot{h}_j = -a_j^\top \dot{\xi} = -\omega, a_j^\top \xi + \omega, a_j^\top u$$

取线性 class-$\mathcal{K}$ 函数 $\alpha(h) = \alpha h$（$\alpha > 0$），CBF 条件 $\dot{h}_j \ge -\alpha h_j$ 化为对 $u$ 的**线性不等式**：

$$\omega, a_j^\top u ;\ge; \omega, a_j^\top \xi - \alpha, h_j(x), \qquad j = 1,\dots,M$$

### 6.4 原语到名义 ZMP 的映射 (Primitive-to-Nominal Mapping)

Primitive Compiler 将 VLA 原语编译为期望速度 $v_{cmd}$ 与模式 $\sigma$（步态/姿态）。定义期望 DCM 为 $\xi_{des} = p + v_{cmd}/\omega$，采用 DCM 跟踪律生成名义 ZMP：

$$u_{nom} = \xi + K_\xi (\xi - \xi_{des}), \qquad K_\xi > 0$$

代入 DCM 动力学得 $\dot{\xi} = -\omega K_\xi (\xi - \xi_{des})$，即名义控制下 DCM 指数收敛至期望值，机器人以 $v_{cmd}$ 巡航。

### 6.5 CBF-QP 与约束集 (The QP)

每个控制周期（500Hz–1kHz）求解凸 QP：

$$u^* = \arg\min_{u \in \mathbb{R}^2}\ \tfrac{1}{2}|u - u_{nom}|^2$$

$$\text{s.t.}\quad \omega, a_j^\top u \ge \omega, a_j^\top \xi - \alpha h_j(x), \quad \forall j \qquad \text{(CBF 安全约束)}$$

$$\phantom{\text{s.t.}}\quad A_{\mathcal{S}}, u \le b_{\mathcal{S}} - \delta_u \mathbf{1} \qquad \text{(ZMP 物理可实现性：单边接触下 ZMP 必须位于支撑多边形内)}$$

$$\phantom{\text{s.t.}}\quad |p - u| \le \mu z_c \qquad \text{(摩擦锥：LIP 下水平力 } F_t = m\omega^2(p-u) \text{，由 } |F_t| \le \mu m g \text{ 直接导出，实现时取多边形内逼近)}$$

QP 维度为 2 变量、$O(M)$ 个线性约束，单次求解 $\ll 1$ms，满足实时性。**注意摩擦锥约束中 $\mu$ 由 Kino-Tokens 的特权蒸馏头在线估计（第 4 节）——这是感知层与安全层的显式耦合点：检测到冰面 ⇒ $\hat{\mu}$ 下调 ⇒ 可行 ZMP 调制范围收缩 ⇒ Shield 自动变保守。**

滤波后指令的反解（回传给 Sport Client 的实际速度指令）：由 6.4 节映射关系反解得

$$v_{cmd}^* = v + \frac{\omega}{K_\xi}(\xi - u^*)$$

### 6.6 前向不变性保证 (Forward Invariance Theorem)

**命题：** 设 $x(0) \in \mathcal{C}$，且每个控制周期施加的 $u^*$ 均满足 6.5 节约束，则在 LIP 模型假设下 $x(t) \in \mathcal{C},\ \forall t \ge 0$，即 Capture Point 永不离开收缩支撑多边形，0 步可捕获性全程保持——构成"无论 VLA 输出什么，系统不跌倒"的形式保证。证明直接援引 CBF 前向不变性定理（Ames et al., 2017/2019）：诸 $h_j$ 沿轨迹满足 $\dot{h}_j \ge -\alpha h_j$ 蕴含 $h_j(t) \ge h_j(0) e^{-\alpha t} \ge 0$。

**诚实的假设边界（论文中明示）：** 保证条件于 LIP 假设（恒定质心高度、忽略腿部惯量与角动量）、支撑多边形已知、模型误差有界且被裕度 $\delta$ 吸收。对于模型失配可扩展为输入-状态安全（ISSf）形式，将 $\delta$ 与扰动界定量关联——这部分作为论文附录。

### 6.7 步态切换与离散原语的准入条件 (Mode Switching Admission)

离散原语改变安全集本身：姿态原语改变 $z_c$ 从而改变 $\omega_\sigma$；步态原语改变支撑形态——特别地，trot 的双足对角支撑是退化的线段（零面积多边形），对此采用**虚拟支撑多边形**：取一个步态周期内全部规划落足点的凸包 $\mathcal{S}_\sigma^{virt}$，并配以步态相关的增大裕度 $\delta_\sigma$。每个模式 $\sigma$ 对应安全集 $\mathcal{C}_\sigma$。

**准入规则：** 模式切换原语 $\sigma \to \sigma'$ 被批准，当且仅当当前状态满足 $h_j^{\sigma'}(x) \ge \varepsilon_{switch} > 0,\ \forall j$（即当前状态已落入新模式的收缩安全集深处）；否则原语被拒绝，并以结构化原因码（如 `REJECT: x ∉ C_crawl, margin = -0.04m`）回传 VLA 触发下一轮反思。这保证了分段切换系统的每段都满足 6.6 节不变性，整体安全性由归纳法成立。

### 6.8 不可行回退与裁决指标 (Infeasibility Fallback & Metrics)

若 QP 不可行（如剧烈滑移导致支撑多边形瞬间塌缩），不引入软化松弛变量（安全约束保持硬性），而是触发模式回退链：Reflex 阻尼站立 → 降低质心、加宽站位以扩大 $\mathcal{S}$ → 回传 Execution Report。CBF 干预幅度 $|u^* - u_{nom}|$ 与激活频次作为系统指标记录（接入第 12 节指标体系）：干预幅度长期偏高意味着 VLA 输出系统性越界，本身即是反思质量的可量化诊断信号。

### 6.9 端到端延迟预算表 (Latency Budget)

异常发生 → Kino-Monitor 检出 (<5ms) → Reflex 自稳介入 (<20ms) → Kino-Token 窗口填满 (500ms) → VLA 推理输出 (~1s) → CBF 裁决+编译 (<10ms) → 原语生效。论文将实测各段延迟，并验证 Reflex 维持的准静态安全停留时间 $T_{safe}$ 严格覆盖 $T_{VLA}$——结合 6.6 节，这给出双闭环架构成立的量化前提：**Reflex 负责把状态钉在 $\mathcal{C}$ 内争取时间，CBF 负责 VLA 醒来后说的话不会把状态推出 $\mathcal{C}$。**

## 7. 【新】语义可通行性地图：拓扑记忆的持久化 (Semantic Traversability Map)

v1.2 的 `Update_Topology` 仅是 prompt 内的一句历史文本，机器人转身后记忆即失效。v2 将其升级为独立子系统与独立贡献点：

- **区域 Grounding：** open-vocabulary 分割（如 SAM + CLIP 特征）将 RGB 划分为语义区域；结合 RGB-D 深度反投影，将 2D 像素区域持久化为机器人里程计坐标系下的 3D 区域。
- **可通行性标签的在线修正：** 视觉先验给出初始 traversability 估计；当物理失效发生时，Kino-Tokens 归因结果作为高置信度证据**覆写**对应 3D 区域的 costmap 标签，并通过 CLIP 特征相似度将标签传播至视觉同质的邻接区域（踩破一块薄冰 ⇒ 整片同质冰面降权）。
- **跨时间步与跨视角一致性：** costmap 随里程计维护，VLA 每轮接收当前地图的局部裁剪作为额外上下文，杜绝"转个身就忘"的死循环。
- **学术定位：** "以本体动力学反馈在线修正视觉可通行性地图"与 VLN 社区的 VLMaps / value map 一脉相承但反向而行（彼为视觉写地图，此为物理改地图），可作为独立可评估的贡献章节。

## 8. 评价基准：Kino-Fail v2 —— 参数化失效算子库 (Parameterized Failure Operator Library)

废弃"5 大维度 12 基元"的静态场景清单——该清单既未经 Isaac Sim 工程可行性核验（流体、FEM 软体、FEM 绳索在 RL 并行规模下不可行或不稳定），也未与算法贡献逐一对位。v2 将基准重构为**按仿真器实现机制组织的、带连续特权参数的失效算子库**。

### 8.1 设计原则 (Design Principles)

- **P1 — 工程可实现性优先：** 每个算子必须落到 Isaac Sim / PhysX 的具体机制（物理材质 API、D6 弹簧关节、脚本化碰撞体交换、区域力场、effort limit 调度、渲染-物理解耦绑定），标注计算成本与已知陷阱，保证可在数千环境并行下稳定运行。
- **P2 — 连续参数族而非离散标签：** 每个失效是带连续参数向量 $\theta$ 的算子，$\theta$ 即特权真值，直接服务于第 4 节的蒸馏监督与第 9 节的流形度量。"12 分类器"质疑从根上消解：基准是 $\theta$ 空间上的连续分布，离散实例只是其中的取样点。
- **P3 — 贡献针对性：** 每个算子至少服务一项算法声明——归因模糊对针对 B2/纯本体 baseline 的原则性失效；算子叠加针对组合泛化；参数扫描针对 A/B 边界定位；冲量算子针对 Reflex 层独立验证。
- **P4 — 本体特征匹配对的构造性设计：** 归因模糊对不靠经验巧合，而是通过参数整定使配对场景的本体统计特征（切向阻力-位移曲线、滑移率分布、扭矩谱）**显式匹配**，从而可证明"仅凭本体信号原则上不可区分，必须视觉归因"——这是 B 类不可替代性的构造性证据。

### 8.2 四条机制轴与算子清单 (Mechanism Axes & Operators)

**Axis I — 接触材质场 (Contact Material Field)：**

- **O1 摩擦场重写 (μ-Field)：** _实现：_ PhysX 物理材质 API 按区域设置 $(\mu_s, \mu_d, e)$，静态配置，零运行时开销。_特权参数：_ $\theta = (\mu_s, \mu_d, e, \text{区域掩码})$。_典型实例：_ 均匀冰面 [A]、单侧油污 [A]、摩擦梯度带 [A]。
- **O2 柔顺下陷场 (Compliance-Field)：** _实现：_ compliant contact 参数 $(k_c, c_c)$ + 脚本化高度场局部凹陷；**明确不用 FEM 软体**（并行规模下成本与稳定性不可行），以柔顺接触 + 几何下陷近似。_特权参数：_ $\theta = (k_c, c_c, d_{sink})$。_实例：_ 浅泥 [A] → 深陷软体 [B]——**A/B 边界在 $\theta$ 空间内连续可扫（见 Suite-Bound）**。
- **O3 触发式坍塌 (Collapse)：** _实现：_ 足端接触载荷超阈值 $F_{th}$ 后脚本交换碰撞体（刚性→柔顺/移除），带迟滞防抖；纯事件脚本，成本极低。_特权参数：_ $\theta = (F_{th}, \text{坍塌后 } k_c, \text{区域})$。_实例：_ 薄冰破碎 [B：区域级拓扑标记]。

**Axis II — 外部力旋量与附着 (External Wrench & Attachment)：**

- **O4 弹性附着 (Tether / Adhesion)：** _实现：_ 接触或区域触发时动态创建 spring-damper（D6 关节），参数 $(k, d, L_0, F_{break})$；**明确不用 FEM 绳索**，弹簧关节既满足胡克定律叙事又数值稳定。附着点可选：足端（粘鼠板 [B]）或躯干/腿部+地面锚点（线缆缠绕 [B]）。_关键设计：_ **与 O2 构成构造性归因模糊对**——整定 $(k, d)$ 使其切向阻力-位移曲线与 O2 高粘滞档的阻力曲线匹配。
- **O5 负载注入 (Payload)：** _实现：_ 躯干固连刚体质量 $(m, r_{offset})$ [隐形超载死锁 B：扭矩饱和、策略级放弃/求助]；**液体晃动以 2-DoF 摆式负载近似**（真流体在 RL 并行规模不可行，摆模型为社区标准做法）[动态重心偏移 A]；突发脱载 = 关节断裂事件 [A]。_特权参数：_ $\theta = (m, r_{offset}, l_{pend}, t_{detach})$。
- **O6 外力脉冲 (Push)：** _实现：_ 随机冲量 $(J, \text{方向}, \text{作用点}, \text{时刻})$。[A —— **Reflex 层的专属验证算子**，经典 push-recovery 协议，同时校准 6.9 节 $T_{safe}$]。

**Axis III — 几何与视-物解耦 (Geometry & Visual-Physics Decoupling)：**

- **O7 视-物重映射 (Visual-Physics Remap)：** _实现：_ 渲染材质与物理材质独立随机绑定——这正是第 10 节 PHASE 1 抽象迷宫的基础设施，在此显式化为基准算子。_实例：_ 视觉欺骗断崖 [B]。_工程要点：_ **必须同步注入深度通道腐蚀**（建模过曝/红外吸收/镜面导致的深度失效），否则 RGB-D 的 D 通道直接看穿欺骗、场景失效；该腐蚀对应真实深度相机的已知物理失效模式，叙事自洽。_特权参数：_ $\theta = (\text{视觉-物理绑定矩阵}, \text{深度腐蚀模型参数})$。
- **O8 不可见碰撞体 (Invisible Collider)：** _实现：_ 碰撞体不挂渲染几何。_实例：_ 玻璃门 [B：拓扑标记+绕行]。
- **O9 几何托底 (High-Centering)：** _实现：_ 低矮脊状凸起使足端部分悬空、腹部受力。_实例：_ 底盘托底 [B：姿态原语抬升底盘]。

**Axis IV — 本体退化 (Embodiment Degradation)：**

- **O10 执行器衰减 (Effort-Decay)：** _实现：_ 对指定关节按时间表衰减 effort limit $\tau_{max}(t)$，可叠加关节摩擦增大；仿真中即改 actuator 配置，零额外成本。_实例：_ 热衰减跛行 [A/B 边界：衰减率 $\theta$ 扫描——轻度衰减低层补偿即可 [A]，重度衰减需切换跛行步态/降级任务 [B]]。**与 O5 构成第二组归因模糊对：** 同为扭矩饱和征兆，外因（超载）vs 内因（衰减），恢复策略相反（卸载/求助 vs 跛行步态+限速）。
- **O11 观测偏置 (Obs-Bias)：** _实现：_ 向观测流注入 IMU 偏置/漂移（虚假坡度补偿）。[A]。

### 8.3 基准套件 (Benchmark Suites)

- **Suite-Cal（A 类校准套件）：** O1、O5(摆载)、O6、O11 的 A 类实例。用途：验证 Kino-VLA 与 B1 自适应 baseline 持平，自证 baseline 实现公允、本工作不在 A 类抢功。
- **Suite-Sem（B 类语义套件）：** 全部 B 类实例，核心是**三组构造性归因模糊对**：(1) O4 足端粘附 vs O2 高粘滞下陷——本体阻力曲线匹配，视觉（黄色板面 vs 棕色泥泞）决定后退绕行 vs 高抬腿通过；(2) O3 薄冰 vs O1 均匀冰面——同为低摩擦征兆，决策粒度相反（区域级拓扑封禁 vs 局部降速降重心）；(3) O5 超载 vs O10 衰减——同为扭矩饱和，外因 vs 内因归因相反，恢复策略相反。
- **Suite-Comp（组合套件，仅测试、绝不训练）：** 算子叠加：O1+O5（冰面+负载）、O4+斜坡地形、O2+O10（泥地+衰减）。算子化设计使叠加在工程上是天然的（参数向量拼接），直接支撑第 9 节组合泛化声明。
- **Suite-Bound（A/B 边界扫描套件）【新】：** 在 O2 的 $(k_c, d_{sink})$ 与 O10 的衰减率上连续扫描 $\theta$，以仿真穷举确定"低层适应"与"语义介入"两种策略成功率曲线的交叉点 $\theta^_$（真值边界），再检验系统实际决策翻转点与 $\theta^_$ 的偏差。**这把第 2.5 节的核心科学问题——"何时该叫醒大脑"——变成一条可画出的曲线**，是本基准独有的实验形态。
- **Suite-OOD：** 留一算子（leave-one-operator-out：如 O3 整族不参与训练）+ 参数外推（训练分布范围外的 $\theta$，检验特权蒸馏头的残差异常分数是否如第 9 节预期升高）。

### 8.4 算子-贡献映射矩阵 (Operator-to-Claim Matrix)

特权蒸馏监督：全部算子（$\theta$ 即回归目标）。归因模糊/FSM 失效论证：O4↔O2、O3↔O1、O5↔O10。组合泛化：Suite-Comp 的算子叠加。A/B 边界定位：O2、O10 扫描。Reflex 独立验证与 $T_{safe}$ 标定：O6。拓扑地图与区域传播：O3、O7、O8。姿态原语：O9。策略级放弃 (`Hold_and_Request`)：O5 饱和档。每条算法声明在基准中均有专属落点，反向亦然——不为任何声明服务的场景不进入基准。

## 9. 泛化评估：从"正交基"声明到组合泛化实证 (Generalization Protocol)

删除 v1.2 中不可证明的"任何失效可分解为 5 维度线性组合"声明，代之以三层实证协议：

- **流形偏离的物理量化：** 得益于特权蒸馏，OOD 检测不再是抽象的"流形偏离"，而是特权参数 $\theta$ 预测残差的显式异常分数——可报告、可校准、可画 ROC；Suite-OOD 的参数外推档直接检验该分数随 $\theta$ 越出训练分布的单调上升性。
- **留一算子法 (Leave-One-Operator-Out)：** 以算子整族为留出单位（如 O3 坍塌族完全不参与训练），零样本测试归因与恢复，验证插值泛化——比留出单一场景实例严格得多。
- **组合泛化 (Compositional Generalization)：** 训练只见单一算子，测试注入 Suite-Comp 的算子叠加（O1+O5、O4+斜坡、O2+O10），报告归因准确率与恢复成功率。组合实验证明的是构成性（compositionality）而非插值——这才是"不只是分类器"的决定性证据，而算子化参数设计使"叠加"在工程上即参数向量拼接，天然无缝。
- **【新】A/B 边界一致性 (Boundary Consistency)：** Suite-Bound 给出仿真穷举的真值策略翻转边界 $\theta^_$，报告系统决策翻转点与 $\theta^_$ 的偏差分布——将"何时该叫醒大脑"从定性声明变为定量曲线。
- **解耦泛化：** 底层传递物理直觉，顶层 LLM 结合视觉完成 open-vocabulary 推理（"抬腿困难"+"黄色板面"⇒ 粘鼠板），实景部署时 Region 自动泛化为 `Grass / Mud / Glue_Trap` 等开放词汇。

## 10. 数据集构建：特权真值校验的事后反思链蒸馏 (Privileged-Grounded Hindsight CoT)

保留 v1.2 的四阶段流水线（物理反事实沙盒 → 失败巡检截获 → 上帝视角打标 → SFT 蒸馏），核心升级在 PHASE 3：

- **PHASE 1 物理反事实沙盒：** 程序化生成抽象迷宫，随机打乱物理-视觉对应（蓝色平整区域 = 0.15m 下陷软体网格），强迫模型放弃视觉常识捷径，纯靠 Kino-Tokens 与 RGB 的冲突重建可通行性认知。
- **PHASE 2 失败巡检与多模态快照：** Kino-Monitor 截获异常瞬间，打包 `[前5帧 RGB] + [前5帧深度] + [Kino-Tokens] + [前几轮完整 VLA 输出] + [特权物理真值]`。
- **PHASE 3 上帝视角打标 + 【新】真值一致性过滤：** Oracle 大模型（如 Gemini 3.1 Pro）依据快照写反思链。**关键改进：** Oracle 只是看快照讲故事，其 CoT 可能是流畅的 confabulation。因此新增自动校验器：CoT 中的物理归因结论（"高粘滞软体地形"）必须与仿真特权真值（材料参数确为软体网格）一致，且其选择的原语必须属于该失效类别的可行恢复集，否则**整条样本丢弃**。这将数据集质量从"大模型猜的"提升为"物理 grounded 的"，构成独立的方法论贡献点。
- **Prompt 约束保留：** 严格原子动作（每轮一个原语，但从扩充库中选择）、2D 像素坐标输出格式、安全铁律（突发陷阱优先脱困、禁止当轮直接重规划绕行）。
- **逻辑闭环示例保留：** T 轮 `Backstep(0.5m)` + `Update_Topology(Blue_Flat_Area, Untraversable)` → T+1 轮地图已标记危险且警报解除 → 输出 `Replan_Waypoint([580,360])` → 深度反投影为 3D 航点 → Planner 执行，闭环完成。
- **PHASE 4 SFT 蒸馏：** LoRA 微调 Qwen2-VL，学会"依据物理反馈修正视觉图层"与"审视自己的过去"，切断死循环。

## 11. 端到端闭环对齐范式 (SFT + Embodied DPO)

保留两阶段范式，优化目标锚定物理导航成功率：

- **Stage 1 — Kino-SFT：** 经真值过滤的 Hindsight CoT 数据集 + LoRA 微调，赋予基础具身常识与归因逻辑。
- **Stage 2 — Embodied DPO：** SFT 模型置于 Isaac Lab 闭环，在同一失效状态节点以不同 Temperature 采样多条 Rollout；以最终物理结果（成功脱困到达 vs 跌倒/死循环）构建偏好对 $(State, Chosen, Rejected)$；DPO 直接拉高成功轨迹概率，免去 PPO 的繁重开销，实现动作输出与导航成功率的端到端对齐。**v2 补充：** 偏好对的构建天然覆盖"归因模糊对"场景——同一状态下选错策略（泥地却后退绕远 / 粘鼠板却强行通过）的轨迹成为高质量 Rejected 样本，使 DPO 直接优化归因正确性。

## 12. 【新】Sim-to-Real 策略与实验设计矩阵 (Sim-to-Real & Experimental Design)

### 1) Sim-to-Real 桥接（高频通道的房间里的大象）

- **接触伪影治理：** 仿真接触求解器在高频段存在穿透与力尖峰伪影，而 Kino-Tokens 恰恰消费高频特征。对接触信号施加结构化域随机化（尖峰注入、低通抖动、延迟扰动），并以 Actuator Network（Hwangbo et al. 范式）建模真机电机/齿轮箱非线性与温漂。
- **真机校准微调：** 在真机上采集小批量对齐数据（人为布置的安全失效场景：油布、弹力绳、配重背包），对提取器做 few-shot 校准。
- **Kino-Monitor ROC 报告：** 真机上报告异常检测的误报率-漏报率曲线与所选工作点——这是 Reflex Loop 实用性的直接证据，也是审稿人想看却几乎无人报告的指标。

### 2) Baseline 矩阵与消融

- **B1 — RMA 式自适应策略（无 LLM）：** 验证 A 类场景持平、B 类场景失效（不可替代性论证之一）。
- **B2 — Rule-based FSM（异常→脚本后退→costmap→重规划，无 LLM）：** 在归因模糊对场景上失效（不可替代性论证之二）。
- **B3 — 视觉反思 VLA（REFLECT 式文本摘要注入，无 Kino 通道）：** 验证隐性物理失效的提前介入价值。
- **B4 — Kino-VLA (Text)：** 路线 A 文本表征版本。
- **B5 — Kino-VLA (Latent, Full)：** 完整系统。
- **消融轴：** 去 Reflex Loop / 去 CBF Shield / 去拓扑地图 / 去真值过滤（用未过滤 CoT 训练）/ 去 DPO / 门控开关 / 特权蒸馏 vs 纯对比学习。
- **指标体系：** 任务成功率、跌倒率、归因准确率、恢复时间、Kino-Monitor ROC、组合泛化成功率、CBF 干预频次与幅度、端到端延迟分布。

## 13. 相关工作防线 (Related Work Defense)

必须正面对齐的文献清单与差异点速查：

- **自适应运动控制：** RMA (Kumar et al., RSS'21)、Lee et al. (Sci. Robotics'20)、DreamWaQ —— 差异：连续参数自适应 vs B 类离散语义决策；latent 服务于控制策略 vs 投影至 VLM 语义空间。
- **失效归因与重规划：** REFLECT (Liu et al., CoRL'23)、AHA —— 差异：低频视听事件归因 vs 1000Hz 本体动力学通道；事后归因 vs 反射层保活下的在线归因；机械臂操作 vs 欠驱动足式整机；文本摘要注入 vs 真值蒸馏的连续 latent。
- **足式 LLM 导航：** NaVILA、QuadrupedGPT、Helpful DoggyBot —— 差异：Locomotion 黑盒单向调用 vs 动力学异常上行对齐。
- **足式异常检测经典线：** legged robot anomaly / slip detection 文献 —— 差异：检测止于报警 vs 报警接入语义归因-恢复闭环。
- **VLN 地图表征：** VLMaps、value map 系 —— 差异：视觉写地图 vs 物理失效改地图。