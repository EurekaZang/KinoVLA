# Paper-A 组会汇报发言稿（中英文夹杂 · 配合 PPT）

**配套 PPT：** `paper/PaperA_claims_group_meeting.pptx`（12 页）  
**建议总时长：** 20–30 分钟（含提问可 35–40 分钟）  
**汇报目的：** 向老师同步 Paper-A 的 **claims（主张）**、**实验设计动机**、**真实栈结果**、以及 **结果如何支撑哪条 claim**。

**使用说明：**

- 正文以**中文叙述**为主；专业/自定义名词保留 **English**，并在首次出现时用括号给**中文释义**。
- 灰色小字「〔切换〕」提示翻页；「〔可略〕」可在时间紧时跳过。
- 文末附 **术语表**，方便老师对照提问。

---

## 开场（约 1 分钟 · PPT 第 1 页）

〔切换到 Slide 1 · 封面〕

老师好。今天用这份 **纯英文** 幻灯片同步 **Paper-A** 的进度。

论文标题是：**Feel It, See It, Recover**——副标题是  
**Cross-Modal Failure Attribution for Safe Quadrupedal Navigation Recovery**  
（跨模态故障归因，用于四足导航中的安全恢复）。

**Object of study（研究对象）** 我先说清楚：  
这篇 paper **不是**做一个“完整导航系统”的 system paper，而是研究  
**cross-modal failure attribution（跨模态故障归因）**——  
当 **proprioception（本体感觉 / 身体传感）** 与 **vision（视觉）** 给出冲突证据时，  
系统如何**正确命名物理原因**，从而选择**正确的 recovery（恢复动作）**。

**Evaluation currency（评价货币 / 用什么当证据）** 也不是 closed-loop reach  
（闭环“能不能走到终点”的系统分数），而是：

1. **Frozen failure snapshots（冻结故障快照）**——异常触发时刻的固定观测，多 agent 可重复打分；  
2. **Interventional consequence（干预性后果）**——只改 attributed label（归因标签），测物理代价。

**Evidence stack（证据栈）：** 在 **Isaac Sim 的 Unitree Go2 资产 + RTX** 上，A0–A7 已有实验产物；其中 A4 agent-cost、A5 非零覆盖、A6 learned boundary 仍需补强。当前没有物理机器人证据。

接下来我会按：问题 → C1–C5 → 各实验 design / why / results / claim bridge 汇报。

---

## 第 1 部分 · 问题与评价货币（约 2–3 分钟 · PPT 第 2 页）

〔切换到 Slide 2〕

### 1.1 故障模式是什么

四足机器人会出现一种很“坏”的情况：

- 相机看起来是**安全路面**；  
- 但身体（关节力矩、接触、effort）感觉到的是**另一套物理**——或者反过来。

所以有两个模态：

| 英文术语 | 中文释义 |
|----------|----------|
| **Proprioception** | 本体感觉 / 身体传感：关节角、力矩、接触、effort 等，**不是**相机 |
| **Vision** | 视觉：RGB / 场景外观 |
| **Attribution** | 归因：给故障起一个正确的 **physical cause（物理原因）名**，再选恢复动作 |

如果 attribution 错了，后面的 recovery 会把机器人带向 fall（摔倒）、catapult（弹射）、strand（困住）等坏结果。

### 1.2 为什么不用“闭环导航分”当主证据

早期 E 系列已经说明：**closed-loop reach** 容易被 reset 残差、trigger 噪声、顺序效应污染。  
所以 Paper-A 的主货币是：

| 英文术语 | 中文释义 |
|----------|----------|
| **Frozen snapshots** | 冻结快照：异常 onset 时把观测写盘，之后所有 agent 评同一批数据 |
| **Interventional consequence** | 干预后果：场景固定，只换 label，看物理 outcome |
| **Open-loop single decision** | 开环单次决策：在快照上做一次归因/动作选择，不跑完整导航 loop |

### 1.3 这篇 paper 是什么 / 不是什么

- **Is：** cross-modal **conflict resolution（冲突消解）** 作为可学习的 semantic operation（语义操作），以及 attribution 与 recovery safety 的因果链。  
- **Is not：** SLAM / 地图 / CBF shield 主贡献 / locomotion policy 训练 paper。  
- 完整 **KiNO** 栈（monitor、VLA planner、shield）是 **infrastructure（基础设施）**，不是 Paper-A 的贡献点。

**VLA** = Vision–Language–Action model（视觉–语言–动作模型；这里是 Qwen3-VL-4B + LoRA）。  
**Kino-Tokens** = 把身体窗口压成 latent（潜变量）摘要，注入 VLA 的可学习 token。

---

## 第 2 部分 · 五条 Claims 总览（约 3 分钟 · PPT 第 3 页）

〔切换到 Slide 3〕

Paper-A 有五条主 claim，记作 **C1–C5**。每条后面都有独立实验腿，避免“一条结果撑全文”。

### C1 — Necessity, both directions（双方向必要性）

存在一类故障：**纯 proprio 函数原则上无法区分**（matched sticky vs soft：身体流一致、视觉分开）；  
也存在一类：**视觉误导或看不见，必须靠 proprio**。  
**Neither modality subsumes the other（任一模态都不能吃掉另一模态）。**

### C2 — Learned conflict resolution（学会的冲突消解）

**Having a vision channel ≠ using it（有视觉通道 ≠ 会用视觉做证据权衡）。**  
冲突能力要 **train（训练）**；**bidirectional conflict（双向冲突）** 是压力测试，防“永远信相机”。

### C3 — Attribution ⇒ recovery / safety（归因决定恢复与安全）

同一物理场景下，只换 attributed label，outcome 可以完全反转。  
**Misattribution costs are asymmetric（误归因代价不对称）** → 不确定性下的 **safe default（安全默认动作）**。

### C4 — Generalization + abstention + boundary（泛化 + 弃权 + 边界）

现在有一个**严格限定的正面方法 claim**：冻结的 See–Feel–Act 门控在校准材质支持域内，仅当动作、归因、RGB 支持和本体异常四者一致时放行，否则采取 A4 的 safe default。两套 post-freeze 数据都严格击败 always-safe。全面 OOD、未见算子、单分数 abstention 和 learned O10 boundary 仍是明确的 failure boundaries。

### C5 — Fair opponent（公平对手）

纯身体 baseline **B1** 在新版 realistic benchmark 上仍然很强：A2 留出
balanced accuracy 为 **0.970–0.986**，A3 macro 为 **0.981–0.982**。因此多模态方法的
优势必须通过对 B1 的配对置信区间证明，不能依赖弱 baseline。旧 E1 的 matched
构造仍可作为 controlled characterization，但 scale-v8 的 A1 equivalence gate 未通过，
所以新版结果不得再写“B1 的 matched failure 必然 by construction”。

---

## 第 3 部分 · A0 公共基础设施（约 2–3 分钟 · PPT 第 4 页）

〔切换到 Slide 4〕

在讲 C1 数字之前，先说 **A0**：没有 A0，后面所有表都不可信。

### A0.1 Determinism（确定性）

**Why：** 同一 scenario 换 run order，结果必须一致；否则 closed-loop 数字是 noise。

**Result：** 采用 **deep_reset（深度重置）** + 固定几何后：

- cross-order **max|Δ| = 0**；  
- **same-op C2ST ≈ 0.5**（同算子、不同顺序：分布应不可分）。  

**C2ST** = Classifier Two-Sample Test（分类器双样本检验）：能区分两个分布吗？**≈0.5 = 分不开**。  
**Control（对照）** 在“残留未清”时会 fail，说明我们不是测不到差异。

**Bridge：** 锁住 A4 / A6 一切 closed-loop 数字。

### A0.2 Triggers（触发）

**Why：** 归因应在“事件已发生”的条件下评，避免和 detector 调参缠在一起。

**Result：** **Oracle-primary（特权 oracle 触发为主）**；  
realistic monitor 只做 robustness（稳健性），不是主 protocol。

### A0.3–A0.5 Corpus / 外观 / 注册表

**Why：** 所有 agent 必须评 **同一批 frozen 数据**。

**Result：**

- 每 taxonomy cell 目标 ≥100 snapshots；  
- **appearance train/test split（外观训练/测试划分）** 预注册，防“黄板 → Backstep”捷径；  
- **admissible recovery set（可接受恢复动作集合）** + success 准则注册表。

**McNemar** = 配对准确率检验（同一批样本上两 agent 对错对照）。  
**Wilson CI** = 二项比例置信区间。

---

## 第 4 部分 · C1 × A1（约 3–4 分钟 · PPT 第 5 页）

〔切换到 Slide 5〕

### Design motive（设计动机）

要证明：**纯视觉与纯 proprio 谁也 subsume 不了谁**——用 **certified indistinguishability（可认证的不可区分）**，不是感觉。

### 主构造：Matched O4 sticky ↔ O2 soft terrain

| 术语 | 释义 |
|------|------|
| **O4** | 粘性 / adhesion 类故障（如 tether / sticky board） |
| **O2** | 软/可变形地面（compliant terrain / mud 类） |
| **Matched pair** | 匹配对：在 force-law 族上构造，使 **binding proprio 表示** 一致 |
| **obs48+τ** | 绑定用的观测+力矩窗口特征 |

**Results：**

- Proprio **byte-identical**：max|Δobs48+τ| = **0.0**；  
- **C2ST ≈ 0.5**（身体特征分不开）；  
- **CLIP vision AUC = 1.0**（视觉完美可分）。

**CLIP** = 视觉-语言预训练模型，这里用作视觉可分性探测。

### 镜像：T3 · O7 / O8

| 术语 | 释义 |
|------|------|
| **T3** | taxonomy 中“视觉误导 / 需 proprio”的格子 |
| **O7** | 视觉 remapping 类欺骗 |
| **O8** | 不可见障碍等“视觉盲” |

**Results：** 1D-CNN proprio 在 decisive 侧 ≈ **1.0**；同外观上 CLIP ≈ **0.31**（近随机）。  
**O3↔O1** 认证关不牢已 **demote（降级）**，taxonomy 站在 O4↔O2 + T3 上。

### Claim bridge / Falsifier（主张桥 / 可证伪条件）

**Bridge：** 若任一模态足够，上述构造至少有一个会塌。  
**Falsifier：** matched O4↔O2 上 proprio C2ST ≫ 0.5，或 vision 单独解 T3 mirror。

---

## 第 5 部分 · C2 × A2+A3（+C5）（约 4–5 分钟 · PPT 第 6 页）

〔切换到 Slide 6〕

这是 paper 的**中心叙事之一**：**“有视觉”不等于“会冲突消解”**。

### Why this design

- **n=240 matched conflict snapshots**；  
- **appearance held-out（外观留出）**；  
- **open-loop single decision**。  

直接切断“agent 导航能力差异”的混淆。

### A2 三行结果（balanced conflict 上 attribution accuracy）

| Agent | 含义 | 准确率 |
|-------|------|--------|
| **B1** | 最强纯 proprio attributor | **0.50** |
| **B5-unshaped** | 同 VLA 结构 + vision，但无 conflict 课程 | **0.80** |
| **B5-conflict** | 同上 + matched conflict 监督 | **0.90** |

**McNemar：** B1 vs B5-conflict，b=0, c=24, **p&lt;1e-5**；**3 seeds** 都到 0.90。

### A3 Bidirectional battery（双向冲突电池）

| 术语 | 释义 |
|------|------|
| **B-V** | 同 VLA，mask 掉 proprio（vision-only） |
| **B5-conflict-bi** | 两个冲突方向都教（vision-true 与 proprio-true） |

**Results：** B-V 在 T3 上垮（O7≈0.19 / O8≈0.00）；B1 在 O8 上 100% 互补；  
**B5-conflict-bi** 解 T3 ≈ **0.92**，且 3/3 seeds **proprio-driven**——不是 “always trust camera”。

### C2 bridge / Falsifier

**Bridge：** 冲突能力是 **trained**，不是装上相机就有。  
**Falsifier：** B5-unshaped ≈ B5-conflict，或 bi 模型在 reverse probe 上变成视觉支配捷径。

### C5 bridge / Falsifier（同页）

**Bridge：** B1 在 realistic A2/A3 上保持约 **0.97–0.98** 的 balanced/macro
performance，是强对手；任何 multimodal gain 都必须在相同 pair 上给出严格正的
cluster-level CI。
**Falsifier：** B1 整体表现很弱，或所谓多模态优势只来自未配对比较。当前 falsifier
没有触发，但“matched failure 必然由构造造成”的更强解释也没有被 realistic A1 证实。

---

## 第 6 部分 · C3 × A4（约 3–4 分钟 · PPT 第 7 页）

〔切换到 Slide 7〕

### Why label-swap，而不是 agent 赛跑

Closed-loop “谁导航更好” 混进太多因素。  
**A4** 固定世界、**oracle trigger** 一次、**scripted recoveries（脚本恢复控制器）**——  
**只改 label ℓ**。共 **630 episodes**（9 scenarios × 7 labels × N=10）。

| 术语 | 释义 |
|------|------|
| **Label-swap matrix** | 标签置换矩阵 M(s,ℓ)：场景 s、归因标签 ℓ |
| **Canonical recovery** | 真因对应的“正确脚本恢复” |
| **Regret** | 相对最优标签的超额代价 |
| **ERS** | Expected Recovery cost（期望恢复代价） |

### Headline numbers

- Canonical success：**1.00** [0.72, 1.00] Wilson  
- Off-diagonal crux：**0.00** [0.00, 0.28]  
- T2 cost **antisymmetry（代价反对称）≈ 4×**  
- Actual-action composition：B5-conflict 相对自己的 unshaped 前身 ΔERS=**−0.510**，两阶段 95% CI **[−0.813,−0.175]**；但与 B1/B-T 的差异 CI 跨 0，所以不是所有 baseline 中最好；
- Safe-default crossover：**p\*(adhesion) ≈ 0.25**

### Claim bridge / Falsifier

**Bridge：** 若 attribution 只是装饰，矩阵对角线 ≈ 非对角线。  
**Falsifier：** crux cells 上换 label 不改物理 outcome。

---

## 第 7 部分 · C4 × A5+A6（约 4 分钟 · PPT 第 8 页）

〔切换到 Slide 8〕

### A5 — Generalization & abstention

**Why：** 杀死 color shortcut；展示 **calibrated Hold**。

| 术语 | 释义 |
|------|------|
| **Appearance OOD** | 外观分布外（held-out 外观） |
| **OOD-θ residual** | 特权物理参数 θ 相对训练支撑的距离；大 → 不确定 |
| **LOO** | Leave-one-out（留一算子族） |
| **AUROC** | 弃权/误差排序质量 |
| **Risk–coverage** | 覆盖率–期望代价曲线 |

**Results：**

- 外观留出总体变化：B-V **−0.175** 最差，B1 **+0.007**；但每个 taxonomy cell 的变化不同；
- LOO-O5：模型命名仍为 0/48，residual **16.09** → Hold → 48/48 safe；见过 O5 的 control 同样命名失败，因此这是 detectable-physics fallback，不是 novel semantics；
- residual 错误排序 AUROC：B-F **0.897**，B5-conflict-bi **0.869**；但 692 个样本中没有一个 θ 超出训练范围；
- 冻结外观 test 上 always-safe cost **1.52**；最低非零覆盖点 coverage **0.0533**、cost **1.84**，相对 safe 的两阶段 CI 为 **[+0.32,+0.32]**；
- T2 的 residual 完全常数，只能选择 coverage 0 或 1，不能做 selective prediction；
- Composition：**16/16**。

**A5.6 正向闭合：** 不改 VLA/Projector，只增加四信号后置门控。final-v2 的 coverage **0.2083**、cost **1.5417 vs safe 1.75**、paired CI **[−0.2083,−0.2083]**；同一冻结 gate 的六新外观簇复现为 coverage **0.20**、cost **1.52 vs 1.72**、CI **[−0.20,−0.20]**。两套放行归因精度均为 **1.00**，每个外观簇均无伤害。

### A6 — Intervention boundary

**Why：** **θ\*** 必须 **agent-independent（与被评 agent 无关）**，由 privileged base 定义。

| 术语 | 释义 |
|------|------|
| **θ** | 故障严重度等特权物理参数 |
| **θ\*** | base policy 无干预时“还能成功”的边界严重度 |
| **O10** | effort / actuator decay 类故障 |

**Results：** O10 base 在 floor 0.25 与 0.30 之间翻转；论文主估计是 bracket **[0.25,0.30]**，`0.2754` 只作描述性插值。
**O2：** 严格 O2_A nominal 上，B5-unshaped/B5-conflict-bi 实际干预率均为 1.0，zero-shot 为 0.5。
**Honest：** 三个 snapshot agent 在所有 floor 上都实际干预；residual 已落入 schema-v2 结果卡，但只有 5 个 floor，failure AUROC=1.0 的 exact permutation p=0.10，因此 learned boundary 未建立。

### C4 bridge / Falsifier

**Bridge：** A4 的 consequence asymmetry 给出 safe fallback，A5.6 用可观测 See–Feel–Act 共识决定何时释放更高效动作，并在两个独立冻结语料上验证。
**Falsifier 判决：** C4 只在 calibrated material support 内成立；A5.5 单分数 risk–coverage 和 A6 learned boundary 仍失败，禁止扩写成全面 OOD 或连续 severity policy。

---

## 第 8 部分 · A7 方法消融（约 3–4 分钟 · PPT 第 9 页）

〔切换到 Slide 9〕

A7 服务 **C2 的方法章节**：**哪些 ingredient load-bearing**。  
强调：**protocol-clean（协议干净）**——dose 主曲线只用统一配方。

### 1) Latent vs text（M7 route ablation）

**Controlled injection（受控注入）：** 同一 VLM，只改身体编码方式。

- Vision-only ambiguity：**0.458** vs text/latent：**1.0**  
- Latent 有 **θ head（预测特权 θ）**；相对 binned text **1.27× 更少 prompt tokens**  
- **Honest：** greedy 精度上 latent 不必“赢”text；卖点是 **θ + bandwidth + sampling**

### 2) Conflict dose（冲突剂量曲线）

Unique samples ∈ {0,5,10,20,40} × 3 seeds，**同一 SFT 配方**。

- 主结论：**dose 10** mean O4 attr **0.733**；attribution-implied A4 regret **0.40** 最好（缓存行缺动作参数，不能称 actual-action ERS）
- Protocol 下 **dose 5 不稳定**（mean 0.333，含 seed 崩塌）——**如实报告**  
- 不等价 upsample “修好 dose5” 只作 **sensitivity**，**不进主曲线**

### 3) Truth filter + grounding

| 术语 | 释义 |
|------|------|
| **ApiOracle** | 真实 API 写的 Hindsight CoT（思维链） |
| **Truth filter** | 按特权 θ / 证据关键词过滤 confabulation（胡编） |
| **Grounded-correct** | 归因对 **且** 理由命中真因线索 |

- 训练流：**0.604 → 0.765**  
- 测试时 emitted rationale GC 可达 **0.8**

### 4) Encoder grid + A7.1

- 18/18 cells；privileged θ MAE **0.0279**  
- **A7.1** 全数据 oracle curve 最佳点为 cost 1.48/coverage 0.247；真正的外观留出 test 为 cost **1.42**、coverage **0.133**，相对 safe CI **[−0.20,0]**；
- ECE=**0.522**，四项 conformal screen 均失败；**不**宣称“校准很好”或严格优于 safe

---

## 第 9 部分 · Claim × 实验矩阵（约 1–2 分钟 · PPT 第 10 页）

〔切换到 Slide 10〕

用矩阵一眼看清 **哪条 claim 由哪些实验扛**：

| Claim | 核心腿 |
|-------|--------|
| C1 | **A1** 核心，**A3** T3 侧 |
| C2 | **A2** 三行 + **A3** 双向 + **A7** 方法 |
| C3 | **A4** 核心 |
| C4 | **A5.6** 两套 frozen final（positive）+ **A5.1–5.5/A6/A7** falsifier boundaries |
| C5 | **A1/A2** 上 B1 的 fair-opponent 角色 |

**●** = core，**○** = support。  
A0 是所有实验的底座；当前证据范围止于 A7。

---

## 第 10 部分 · 状态、诚实边界、下一步（约 2–3 分钟 · PPT 第 11 页）

〔切换到 Slide 11〕

### Done（Real stack）

- A0–A7 在 **Isaac 物理 Go2 + RTX** 完成；  
- Headline 来自 frozen snapshots / label-swap / privileged sweeps；  
- **No surrogate** 撑主表（CPU surrogate / ScriptedOracle 不作 primary）；  
- 文稿：problem-first ICRA draft 已有（`paper/main.tex`）；  
- Fast gate：`pytest -m "not slow"` green。

### Honest limits（不藏）

1. A7 dose knee 是 **protocol dose 10**；dose5 的“配方外修复”只 sensitivity。  
2. Latent vs text：**greedy 持平**，卖 θ + tokens，不硬吹精度。  
3. A6：只闭合了 privileged base 的翻转区间；learned decision flip 未成立。
4. Sim 物理（spring-damper adhesion 等）是近似。  
5. 当前没有物理机器人验证；论文全程按 simulation scope 表述。
6. 待 human sign-off：deep_reset 作标准采集；#49/#52 论文措辞。

### 请老师关注的讨论点〔可略〕

- 主文图是否以 **C2 三行 + A3 双向 + A4 矩阵** 为中心？  
- A6 放主文还是附录？  
- A4/A5/A6 的 P0 缺口如何排入补实验优先级？

---

## 第 11 部分 · 术语表页（约 1 分钟 · PPT 第 12 页）

〔切换到 Slide 12〕

最后一页是 **Glossary**。组会提问时如果碰到缩写，可以直接指这一页。  
我不再逐条念；下面发言稿文末有**完整中英对照表**。

### 收尾（30 秒）

总结一句：

> Paper-A 证明的是：**跨模态冲突归因是必要的、可学习的、会改变物理安全结果的，并且能泛化、能弃权、能谈干预边界**——  
> 用的是 **real-stack + frozen snapshots + interventional matrices**，不是 surrogate 闭环刷分。

谢谢老师。欢迎提问。

---

# 附录 A · 完整术语表（学术词 + 本文自定义词）

## A.1 模态与问题设定

| 英文 | 中文解释 |
|------|----------|
| **Cross-modal** | 跨模态：同时使用身体传感与视觉等不同传感器通道 |
| **Failure attribution** | 故障归因：给当前异常指定正确的物理原因类别 |
| **Proprioception / proprio** | 本体感觉：关节、力矩、接触、effort 等身体信号 |
| **Vision / RGB** | 视觉外观信息 |
| **Conflict / conflict resolution** | 冲突 / 冲突消解：两模态证据矛盾时如何取舍 |
| **Evidence weighing** | 证据权衡：根据两侧证据决定信谁，而非固定偏置 |
| **Recovery** | 恢复动作：Backstep、改步态、减速、Hold 等 |
| **Quadruped / Go2** | 四足机器人 / Unitree Go2 |
| **Safe default** | 安全默认：不确定时选“错了代价更小”的动作 |

## A.2 系统与模型（基础设施，非本文主贡献）

| 英文 | 中文解释 |
|------|----------|
| **KiNO** | 本文系统栈总称（监控、token、VLA、shield 等） |
| **VLA** | Vision–Language–Action：吃图+语言式上下文、出动作的多模态策略 |
| **VLM** | Vision–Language Model：视觉语言模型（VLA 的底座） |
| **Qwen3-VL-4B** | 使用的 VLM 骨干 |
| **LoRA** | 低秩适配微调，不全量重训大模型 |
| **Kino-Tokens** | 学习到的身体窗口 latent 摘要，注入 VLA |
| **Kino-Projector** | 把 proprio 窗口投到 VLM 维度的模块 |
| **SFT** | Supervised Fine-Tuning，监督微调 |
| **CoT** | Chain-of-Thought，带 `<Thought>` 的推理文本 |
| **Hindsight CoT** | 事后用 oracle 写的正确思维链数据 |
| **ApiOracle** | 用真实 API 写 CoT 的 oracle（非 Scripted 模板） |
| **Truth filter** | 用特权信息过滤胡编理由，只保留 grounded CoT |
| **Monitor / monitor_abaware** | 异常检测器；A/B 感知版用于稳健性，非主贡献 |
| **Oracle trigger** | 用仿真特权状态在正确时刻触发评价 |
| **CBF-QP shield** | 控制障碍函数二次规划安全层（基础设施） |
| **Isaac / Isaac Lab** | NVIDIA 机器人仿真栈 |

## A.3 故障算子与 taxonomy

| 英文 | 中文解释 |
|------|----------|
| **Operator O1…O11** | 故障注入算子（冰、泥、粘、负载、effort 衰减等） |
| **O4 / adhesion** | 粘性故障 |
| **O2 / compliant terrain** | 软地面 / 可变形地面 |
| **O7 / O8** | 视觉欺骗 / 视觉盲类 |
| **O10 / effort decay** | 执行器努力衰减类 |
| **Taxonomy T1–T5** | 冲突类型学格子（谁有信息、谁误导） |
| **T2** | 视觉真、需冲突消解的一类 |
| **T3** | proprio 真、视觉误导/盲 |
| **T4** | 细结构主要在身体时序里 |
| **Matched pair O4↔O2** | 身体流匹配的粘 vs 软对 |
| **Ambiguity regime** | 仅靠视觉不够、必须靠身体的区域 |
| **Appearance / appearance OOD** | 外观纹理类；外观分布外 |
| **Held-out** | 训练未见的留出集 |

## A.4 统计与认证

| 英文 | 中文解释 |
|------|----------|
| **C2ST** | 分类器双样本检验；≈0.5 表示两分布难分 |
| **Byte-identical** | 字节级完全相同（最强匹配） |
| **Wilson CI** | 比例置信区间 |
| **McNemar** | 配对二分类准确率检验 |
| **AUROC** | ROC 曲线下面积，排序质量 |
| **ECE** | 期望校准误差；A7.1 作诊断，不宣称校准好 |
| **Seed** | 随机种子，用于可重复与多 seed 稳健性 |
| **n=…** | 样本数 |

## A.5 实验编号与 agent 名

| 英文 | 中文解释 |
|------|----------|
| **A0** | 公共基础设施（确定性、语料、注册表） |
| **A1** | C1 认证与扩展 |
| **A2** | C2 三行主结果（scale-up） |
| **A3** | 双向冲突电池 |
| **A4** | 标签置换后果矩阵（C3） |
| **A5** | 泛化与弃权（C4） |
| **A6** | 干预边界 θ\* 与 O2 模态依赖（C4） |
| **A7** | 方法消融（C2 方法） |
| **B1** | 最强纯 proprio 归因器 |
| **B-V** | 视觉-only VLA（mask proprio） |
| **B-T** | 文本摘要注入身体信息的 VLA |
| **B-F** | 无 VLM 的融合分类器 baseline |
| **B5-unshaped** | 有视觉、无 conflict 课程 |
| **B5-conflict** | + 单向 conflict 样本 |
| **B5-conflict-bi** | + 双向 conflict 样本 |

## A.6 后果、弃权与边界

| 英文 | 中文解释 |
|------|----------|
| **Label-swap / interventional** | 只改标签测因果后果 |
| **Canonical success** | 正确标签 + 规范恢复 的成功率 |
| **Regret** | 相对最优标签的超额代价 |
| **ERS** | 期望恢复代价 |
| **Abstention / Hold** | 弃权 / 请求停住，不硬猜 |
| **Risk–coverage** | 覆盖率–风险（期望代价）曲线 |
| **Always / never intervene** | 永远干预 / 永不干预的代价基准 |
| **θ** | 特权物理严重度等参数 |
| **θ\*** | base 无干预时的成功/失败边界 |
| **OOD-θ residual** | θ 相对训练支撑的残差，作弃权信号 |
| **Safe coverage** | 弃权后仍安全的覆盖 |

## A.7 A7 方法专用

| 英文 | 中文解释 |
|------|----------|
| **Route ablation** | 路由消融：latent / text / vision-only |
| **Latent injection** | 用 Kino-Tokens 注入身体 |
| **Text / binned / rich schema** | 用文本写身体统计的不同详细程度 |
| **Conflict dose** | 冲突样本剂量（0/5/10/20/40） |
| **Protocol-matched / protocol-clean** | 主曲线统一配方，禁止混配方刷数 |
| **Sensitivity-only** | 敏感性分析，不进 headline |
| **Grounded-correct / GC** | 对且理由 grounded 的比例 |
| **Encoder grid** | 编码器训练设定网格（privileged/contrastive/scratch × 窗口 × gate） |
| **Privileged distillation** | 用特权 θ 蒸馏的训练 |
| **A7.1 selective prediction** | 用后验熵等做选择性预测 / 弃权 |

## A.8 工程红线用语

| 英文 | 中文解释 |
|------|----------|
| **Real stack** | 真仿真/真模型/真 oracle，禁止用 surrogate 撑主结果 |
| **Surrogate** | CPU 简化替身、ScriptedOracle 等，仅限单测脚手架 |
| **Frozen corpus** | 内容哈希锁定的评价集 |
| **Deep reset** | 深度清零仿真状态，消残留 |
| **Load-bearing** | 支撑主 claim 的数字/实验 |
| **Claim bridge** | 结果如何推出 claim 的一句话 |
| **Falsifier** | 若出现则 claim 被推翻的观测 |

---

# 附录 B · 时间紧时的 12 分钟精简版

1. **1 页（1′）** 对象是 cross-modal attribution，不是导航系统；货币是 frozen snapshots + label-swap。  
2. **3 页（1′）** C1–C5 各一句。  
3. **5 页（2′）** C1：byte-identical + C2ST 0.5 + CLIP 1.0；T3 mirror。  
4. **6 页（2′）** C2：0.50 / 0.80 / 0.90；bi 0.92；C5 fair B1。  
5. **7 页（2′）** C3：630-episode matrix；conflict 相对 unshaped ΔERS CI<0，但不是所有 baseline 中最好。
6. **8 页（2′）** C4：A5.6 两套 final 均 Δsafe<0，六外观簇复现、放行精度 1.00；同时 residual 单分数与 learned boundary 仍失败。
7. **9+11 页（2′）** A7 dose10；entropy/MSP Δsafe CI 触零；强调 C4 正结论仅限 supported structured recovery。

---

# 附录 C · 可能的提问与简答提纲

**Q：和完整导航 paper 有何不同？**  
A：我们不 claim 系统导航 SOTA；claim 的是 **归因操作** 本身 + 其对安全后果的因果。

**Q：为什么 B1 在 conflict 上只有 0.5 还叫 fair？**  
A：C5 看的是它在新版 A2/A3 的整体表现仍约为 **0.97–0.98**，所以它不是弱 baseline。
旧 controlled matched 构造可单独报告；新版不能在 A1 equivalence gate 未通过时把
matched 差异直接解释成信息论必然。

**Q：dose 5 很差会不会打脸 “small data”？**  
A：主结论是 **protocol dose 10** 的 knee；dose 5 不稳定是诚实范围，不混配方刷 5。

**Q：sim 到 real？**  
A：当前主证据全部来自 Isaac Sim 的 Go2 资产，没有物理机器人验证，也不做 closed-loop 硬件导航 claim。

**Q：monitor 是不是贡献？**  
A：Detection 不作贡献；oracle-primary。monitor_abaware 是基础设施 + 少量稳健性。

---

*文件路径：`paper/PaperA_group_meeting_speech_script.md`*  
*配套：`paper/PaperA_claims_group_meeting.pptx`*
