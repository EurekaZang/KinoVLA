## 我的判断

导师提出“找一个 benchmark 跑评分”是对的，但这个实验的角色需要定准：

> **它应该证明 external validity，而不是重新证明你内部已经证明得很强的 C1–C3。**

你现在已经有三条相当完整的证据链：

* O4↔O2 在本体通道上不是“比较像”，而是逐字节相同；视觉却可分。
* 在冻结语料和外观留出上，B1、普通视觉融合、冲突训练模型形成 0.50 → 0.80 → 0.90 的显著三档。
* A4 又通过标签干预矩阵证明了归因改变会因果地改变成功、安全和物理代价。

所以现在缺的不是“再造一套内部场景，多跑一点 accuracy”，而是：

> **离开你自己的 Kino-Fail、Go2、Isaac Lab、标签词表和恢复原语后，这个“融合视觉与身体证据做失败归因”的操作是否仍然成立？**

这也与论文当前边界一致：它不是导航系统论文，导航结果只是归因后果的测量，不应该拿普通导航成功率或 SPL 当外部 benchmark。

---

# 最推荐的方案：Guardian/FailCoT 官方评分 + RoboFail 多模态扩展

我建议把它作为一个组合实验，但在论文里仍可写成一个统一的 **External Failure-Reasoning Benchmark**：

## 第一层：跑官方 Guardian/FailCoT benchmark，获得可引用的公开分数

Guardian/FailCoT 目前非常适合承担“公开 benchmark 跑分”的角色：

* 官方已经发布了代码、数据、模型和评测脚本。([École Normale Supérieure][1])
* 它覆盖：

  * `RLBench-Fail`：仿真、分布内；
  * `BridgeDataV2-Fail`：真实机器人数据；
  * `UR5-Fail`：真实 UR5、多视角；
  * `RoboFail`：真实 UR5；
  * `RoboVQA`：三种 embodiment。([GitHub][2])
* 官方任务正是 planning/execution failure verification，评测器会输出预测、标签和解析后的 failure category，方便接入你自己的模型。([GitHub][2])
* 公开测试规模也比较合适：RLBench-Fail 和 BridgeDataV2-Fail 各有 1000 个 execution、500 个 planning 测试样本；UR5-Fail 为 140/140，RoboFail 为 153/30。([arXiv][3])

这层实验回答：

> “我们的 failure reasoning 模块离开自建数据后，在公开、跨任务、跨环境、跨 embodiment 的失败识别任务上是否仍然有合理表现？”

但必须明确：**官方 Guardian track 基本是 image/text VQA，本身不能证明 Kino-Tokens 有用。** 它只能证明你的视觉语义归因模块或完整训练方法没有在外部数据上崩掉。

### 建议跑的官方表

| Model                         |     Backbone | Input          | RLBench-Fail Exec / Plan | BDV2-Fail Exec / Plan | RoboFail Exec / Plan | UR5-Fail Exec / Plan |
| ----------------------------- | -----------: | -------------- | -----------------------: | --------------------: | -------------------: | -------------------: |
| Guardian official             | InternVL3-8B | RGB multi-view |                reference |             reference |            reference |            reference |
| Qwen3-VL zero-shot            |           4B | RGB            |                          |                       |                      |                      |
| Qwen3-VL + FailCoT SFT        |           4B | RGB            |                          |                       |                      |                      |
| Kino-VLA-General, Kino masked |           4B | RGB            |                          |                       |                      |                      |

官方 headline metric 可以沿用 accuracy，同时补：

* macro-F1；
* per-failure-category recall；
* execution / planning 分开；
* ECE 或 AURC；
* 95% paired bootstrap CI。

这里不要把 Guardian-8B 和 Qwen3-VL-4B 的差异说成方法差异。Guardian 只是 external reference；真正的消融比较必须全部在同一个 Qwen3-VL-4B backbone、相同 LoRA budget、相同训练数据下进行。

---

## 第二层：在同一个 benchmark 上恢复 proprio，真正验证你的核心方法

这层才是承重实验。

原始 RoboFail 数据本来就包含 RGB-D、声音、robot states/proprioception，并且面向失败解释与纠正；真实部分包含 UR5e 失败轨迹，仿真部分包含 AI2-THOR 失败轨迹。([arXiv][4])

因此可以把第一层中的 RoboFail 进一步做成：

> **RoboFail-Multisensory / RoboFail-CM：使用公开 episode，但把原始 proprioception 重新接入 Kino encoder。**

这样不是另造一套私有 benchmark，而是在公开 benchmark 上增加一个预注册的多模态 track。

### 应该比较的模型

| Arm                       |     Vision |      Proprio | 训练过 conflict | 目的                |
| ------------------------- | ---------: | -----------: | -----------: | ----------------- |
| V-only                    |          ✓ |              |              | 视觉基线              |
| P-only                    |            |            ✓ |              | 身体状态基线            |
| V+P concat                |          ✓ |            ✓ |              | 普通融合是否足够          |
| V+P text                  |          ✓ | text summary |            ✓ | REFLECT 式注入       |
| V+P latent                |          ✓ |  Kino-Tokens |              | 有 latent 不等于会消解冲突 |
| **V+P latent + conflict** |          ✓ |  Kino-Tokens |            ✓ | 完整方法              |
| Oracle modality           | privileged |   privileged |            — | 数据可解性上界           |

这与现有 A2/A3/A7 的矩阵是完全同构的，不需要发明新的叙事。

---

# 最关键的一步：不要只报整体准确率，要构造公开数据上的冲突子集

普通 failure benchmark 的绝大多数样本可能是“视觉和状态都同意”的 easy cases。完整模型即使整体 accuracy 高，也可能只是继续看图。

因此建议把 RoboFail/RLBench-Fail 样本重新按你论文的证据结构分组，而不是照搬 quadruped operator 名称：

### E1：Agree

视觉和 robot state 都支持同一个结论，例如明显错抓、明显未完成。

预期：V-only、P-only、fusion 都应当不错。它是 non-regression control。

### E2：Vision-true / proprio ambiguous

机器人做出了相似的运动和夹爪状态，但由于目标物体、目标位置或任务语义不同，正确判断依赖视觉。

例如：

* 运动轨迹相似，但抓的是 wrong object；
* 相似搬运动作，但放到了错误区域；
* 状态变化看起来正常，但与语言指定对象不一致。

预期：P-only 失败，vision/fusion 成功。

### E3：Proprio-true / vision ambiguous

画面本身不能可靠揭示接触是否成功，但 gripper、joint、contact 或 force history 可以区分。

例如：

* gripper 视觉上遮挡，实际没有闭合；
* 物体看起来仍在夹爪附近，但已经 slip/drop；
* 机械臂看起来在运动，实际 blocked/no-progress；
* 插入外观接近，但接触力或位置误差表明 jam/misalignment。

预期：V-only 失败，P-only/fusion 成功。

### E4：Nominal / benign

任务确实完成或无需纠正，检测“不要误介入”。

预期：测试 calibration 和 abstention。

这会把你当前 T1–T5 的科学结构迁移到一个完全不同的 embodiment 上，而不是只迁移标签名字。你论文的主张本来就是“证据结构”，不是“黄色粘鼠板”本身。

---

# 更强的版本：在 RLBench-Fail 生成器上建立 certified matched pairs

Guardian 已经公开了 RLBench-Fail 的完整 failure-generation pipeline，可以重新生成 trajectory，而不仅仅是下载图片。([GitHub][5])

这给了你一个很强的机会：

## RLBench-Fail-CM

在生成失败时同时落盘：

* joint position / velocity；
* gripper width / command；
* end-effector pose；
* contact state；
* simulator wrench 或 interaction force；
* RGB-D multi-view；
* task/subtask instruction。

然后仿照 A1，构造两类 matched pair：

### P-matched pair

本体窗口经过匹配后不可分，但失败原因由视觉目标决定。

要求：

* 对 proprio window 做 C2ST；
* AUC 的置信区间覆盖 0.5；
* vision embedding 可以分开；
* 双向 balanced accuracy 作为指标。

### V-matched pair

相同或高度相似的视觉状态，但接触/力/夹爪状态不同。

要求：

* 对 image embedding 做 C2ST 或相似度 matching；
* proprio 可以分开；
* vision-only 应被构造性封顶。

这一步很重要，因为它会让 external benchmark 不只是“我们在另一个数据集上也有 73.4%”，而是：

> **同一套构造性信息必要性论证，在另一个 simulator、另一种机器人、另一类任务和另一组 failure taxonomy 上复现。**

这比单纯 leaderboard score 强很多。

---

# 推荐的主指标

## 1. Official benchmark score

沿用官方：

[
\text{Accuracy}*{exec},\quad \text{Accuracy}*{plan}
]

再补 macro-F1，防止 failure category 不均衡。

## 2. Conflict-balanced attribution

不要让 agree 样本淹没 conflict：

[
\text{CBA}
==========

\frac{1}{2}
\left(
\text{Acc}*{V\text{-true}}
+
\text{Acc}*{P\text{-true}}
\right)
]

## 3. Fusion gain

[
\Delta_{\text{fusion}}
======================

## \text{CBA}_{V+P}

\max(\text{CBA}*{V},\text{CBA}*{P})
]

这是最接近“方法论 make sense”的单一数字。

必须与 **较强的单模态模型** 比，而不是与随机或弱 baseline 比。

## 4. Conflict-training gain

[
\Delta_{\text{conflict}}
========================

## \text{CBA}_{V+P+\text{conflict}}

\text{CBA}_{V+P\text{ unshaped}}
]

它对应当前 C2：“有两个通道不等于会正确权衡两个通道”。

## 5. Selective risk / calibration

在公开 benchmark 上复现 A5：

* confidence–error AUROC；
* risk–coverage curve；
* AURC；
* abstention 后的 failure cost。

这样 C4 的校准弃权不再只依赖 Kino-Fail。你当前 A5 已经表明 θ residual 对完整融合模型错误有较强检测能力；外部 benchmark 可以测试这种 residual 是否跨 embodiment 仍有排序能力。

## 6. Recovery consistency

对具有 benchmark-native correction labels 的样本，输出：

```text
failure_present
failure_cause
recommended_recovery_class
confidence
```

不要强行输出 quadruped 的 `Backstep` 或 `Switch_Gait`。建立一个固定、预注册的 benchmark-native compiler，例如：

* `retry_grasp`
* `re_align`
* `release_and_regrasp`
* `retry_subtask`
* `replan_object`
* `replan_sequence`
* `hold_or_abort`

评价：

[
\text{Correct Attribution}
\land
\text{Recovery}\in\mathcal{A}_{\text{true cause}}
]

仍沿用你论文的 attribution-gated correct recovery，而不是 exact-string match。

---

# 一个现实问题：目前的 B5-conflict-bi 不适合直接拿去跑整体榜单

你自己的分析已经指出，B5-conflict-bi 是 T2/T3 的 conflict specialist，不是一个全格 Pareto-dominant 的通用归因器；它在 T4/T5 上明显较弱。

因此直接把它扔进 Guardian 全测试集，可能得到一个不好看的 overall accuracy，但这并不意味着冲突消解方法失败，只意味着模型发生了 curriculum specialization。

建议先形成一个统一模型：

## Kino-VLA-General

1. 先用 hindsight/general failure data 做通用 SFT；
2. 再用 conflict data 精调；
3. conflict 精调时混入一定比例的 general replay，避免 catastrophic forgetting；
4. 保留 conflict-only specialist，只用于机制分析。

最终论文中：

* **Kino-VLA-General**：报公开 benchmark overall score；
* **Kino-VLA-CM specialist**：报 conflict subset 和机制上界；
* 不把 specialist 包装成通用 detector。

---

# 数据泄漏要特别小心

RLBench-Fail 的公开 metadata 中已经包含：

* `failure_mode`
* `failure_reason`
* `reward`
* start/end captions
* start/end gripper state
* primitive 等字段。([Hugging Face][6])

其中有些几乎就是答案。

主模型只能读：

* 原始图像；
* task/subtask instruction；
* 原始 observable robot-state history。

以下字段只允许用于 evaluation、stratification 或 oracle upper bound：

* failure reason；
* reward；
  -人工或模型生成的 start/end caption；
* failure mode；
* ground-truth object identity；
  -任何由 failure label 派生的字段。

最好在 appendix 中列一个 **Allowed Inputs / Evaluation-only Metadata** 表，否则 reviewer 很容易怀疑 benchmark leakage。

---

# 其他候选为什么不作为第一选择

## ForceVLA-Data：非常适合补充，不适合当主 benchmark

ForceVLA 已公开代码和数据，包含多视角视觉、proprioception 和六轴 force/torque，覆盖插拔、按压、擦拭、削皮等五类真实接触任务。([Google Sites][7])

它在“身体信号是否有价值”上非常贴近你的方法，但它的主要任务是端到端 action prediction 和真实任务成功率，而不是 failure attribution。直接迁移会混入：

* action head；
* low-level control；
* robot embodiment；
* dataset imitation quality；
* force-control controller。

所以它更适合做一个 appendix representation probe：

* force/contact phase classification；
* failure onset prediction；
* force regression；
* occluded condition 下的 success prediction；
* latent vs text vs no-force。

不要把它作为唯一的外部 benchmark。

## FailSafe/ManiSkill：恢复结果最贴近，但工程和 scope 风险最大

FailSafe 生成 failure–executable recovery action pairs，并在 ManiSkill 中通过 VLA task success 衡量恢复价值，概念上与 A4 很接近。([arXiv][8])

但它要求把你的归因器接入 manipulation VLA 和 7-DoF recovery action，容易重新变成一个完整 manipulation system comparison。你已经有 A4 的干预式物理后果矩阵，不必为了一张外部表再背一个完整控制栈。

---

# 我建议形成的论文新增部分

可以把当前 optional A8 拆成：

## A8a — Public Benchmark Transfer

* Guardian 官方协议；
* RLBench-Fail、BDV2-Fail、RoboFail、UR5-Fail；
* standard visual track；
* official accuracy + macro-F1；
* same-backbone baselines。

## A8b — Cross-Modal RoboFail Transfer

* raw RGB-D + proprio；
* V-only / P-only / concat / text / latent / conflict-trained；
* conflict-balanced attribution；
* calibration；
* benchmark-native recovery admissibility；
* sim→real 或 leave-one-task-out。

## A8c — Certified RLBench-Fail-CM，可选强化

* 重新生成状态轨迹；
* P-matched 和 V-matched pairs；
* C2ST certification；
* 外部 taxonomy heatmap。

你原先的 A8 是 Go2 真实快照迁移，目标是同步 1 kHz proprio 和 RGB-D，做 zero-shot 与 few-shot calibration。 这可以保留为 A8d，或者硬件资源不足时由公开 multisensory benchmark 替代。

---

# 最终建议

**只选一个公开 benchmark 时，我会选 RoboFail，但使用 Guardian 的官方评测管线，并恢复 REFLECT 原始数据中的 proprioception。**

这会给你两张互补的表：

1. **官方表**：在公认公开 split 上有可复现分数；
2. **方法表**：在同一批公开 episode 上，latent fusion 是否显著超过 vision-only、proprio-only 和普通融合。

最理想的论文结论不是：

> “我们在某 benchmark 上达到 82.1%。”

而是：

> “在公开、跨 embodiment 的机器人失败 benchmark 上，模型保持有竞争力的标准 failure-verification 表现；更重要的是，在预注册的跨模态冲突子集上，完整融合模型显著超过两条单模态分支及普通融合，增益集中出现在理论预测的冲突单元，并且没有牺牲 agreement/nominal 单元。”

这才真正把 benchmark 分数接回你这篇论文的科学命题，而不是为了有表而有表。

[1]: https://www.di.ens.fr/willow/research/guardian/ "Guardian"
[2]: https://github.com/paulpacaud/Guardian-FailCot/blob/main/docs/Offline_VQA_Evaluation.md "Guardian-FailCot/docs/Offline_VQA_Evaluation.md at main · paulpacaud/Guardian-FailCot · GitHub"
[3]: https://arxiv.org/abs/2512.01946 "Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models"
[4]: https://arxiv.org/abs/2306.15724 "REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction"
[5]: https://github.com/paulpacaud/Guardian-FailCot "GitHub - paulpacaud/Guardian-FailCot: Codebase for the paper \"Scaling Cross-Environment Failure Reasoning Data for Vision-Language Robotic Manipulation\" · GitHub"
[6]: https://huggingface.co/datasets/paulpacaud/rlbenchfail_test_dataset "paulpacaud/rlbenchfail_test_dataset · Datasets at Hugging Face"
[7]: https://sites.google.com/view/forcevla2025 "force-vla"
[8]: https://arxiv.org/abs/2510.01642 "FailSafe: Reasoning and Recovery from Failures in Vision-Language-Action Models"
