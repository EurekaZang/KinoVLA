# F42 负结果后的方法与确认集恢复方案

## 结论

F42 是有效的一次性负结果，不是可通过调阈值修补的近失误。当前 learned router 在 Scale
上相对 late average 为 +0.04068，但在 Conflict 上为 -0.06179；T2 视觉路由忠实度只有
0.08160，T3 本体路由忠实度为 0.9460。A6 的 O9 准确率为 0.0375，16 个直接接触参数点
最低为 0。当前架构不能作为扩大版数据上的论文主结果。

F42 从现在起只能承担两种角色：原样报告的独立负结果，以及后续架构开发数据。任何使用
F42 标签、预测、门槛结果或图像抽检结论做出的结构选择，都不能再在 F42 上称为独立确认。

## 失效机制

### T2：局部视觉证据被全局表征稀释

新 T2 中 O4 绳索经常只占画面底缘的少量像素。相对六场景开发集，O2/O4 配对的全局视觉
L2 距离由 0.538 降到 0.369，ground-ROI HOG 距离由 3.727 降到 2.602；30 个场景代表
图像对中，阈值 0.05 的变化像素比例由 0.299 降到 0.126，变化重心也更靠近画面底部。
旧 structured-v5 的 HOG 增强分支在 F42 上仍能取得 0.722 T2 accuracy，明显高于当前
统一路由器，但低于其旧三场景确认集的 0.989。这说明局部高分辨率视觉分支是必要方向，
同时数据生成端还需要真正的语义可见性门。

### T3/O9：绝对本体统计不具备跨物理域不变性

F42 本体特征平均行范数为开发集的 0.487 倍；旧 formal test 为开发集的 1.041 倍。
旧 structured-v5 在 F42 上把全部 T3 样本路由到 T2 family，T3 accuracy 为 0。当前
random-forest experts 虽然在 T3 上大多选择 proprio 路由，却无法把新的低幅度物理响应映射
到正确类别。O9 在全 16 点连续参数区间上失败，排除了单一异常参数点的解释。

### 不是主要问题的因素

- 三个外观视角的 pooled accuracy gap 上界为 0.001265；
- 同时最坏算子的三视角 gap 上界为 0.01067；
- Scale、T3、Conflict 和 overall attrition 均低于 5%；
- 五个 checkpoint、特征代码、η=0.8 和统计程序均保持冻结。

因此，当前主要问题不是 texture randomization、样本数、attrition 或统计功效，而是表示与
数据语义的跨域稳健性。

## 推荐架构：KINO conflict-aware invariant router

### 1. 双尺度视觉专家

保留冻结的全局 ViT/CLIP 表征，同时加入不使用 operator/scene 元数据的 bottom-ground ROI
分支。ROI 分支直接消费五帧同步图像的 patch tokens、HOG/edge map 和帧间差分，通过
multi-instance attention 让少量 tether、塌陷边缘和底盘横梁像素不被全图平均。全局与局部
分支分别校准，并输出预测熵、局部显著性质量和 OOD 距离。

### 2. 相对本体时序专家

不再将 80-D 绝对统计直接交给树模型。对每个 episode 使用 encounter 前历史建立在线基线，
构造 `(x - median_pre) / (MAD_pre + eps)`、一阶差分、短时积分和接触相位残差，再由小型
TCN/Transformer 编码 21 帧窗口。该编码只使用部署时可观测历史，不需要 nominal
counterfactual、truth、operator 或 outcome。训练中按 scene、domain 和参数区间执行 group
normalization/domain randomization。

### 3. 冲突专门学习

路由器输入专家后验、局部视觉质量、本体 OOD 分数和跨模态 disagreement；训练使用：

- modality dropout，防止长期退化成默认 proprio；
- T2/T3 matched conflict hard negatives；
- 专家正确性监督和 route-fidelity 辅助损失；
- worst-battery/group-DRO 目标，避免 Scale 提升掩盖 Conflict 退化；
- 低可靠度时显式 abstain，而不是用固定默认专家强制作答。

### 4. 数据语义门

这些门必须在训练/确认预测前冻结，且只判断输入是否实现预定实验语义：

- T2：前置相机投影中的 cue pixel length、宽度、对比度、离底边距离、ground-ROI 占比；
- T2：独立 2AFC 视觉语义抽检，判断 O2/O4 是否能由画面而不是标签命名区分；
- T3：决策窗口内相对 pre-contact baseline 的 effect SNR 与 O7/O8 可分性下限；
- O9：持续底盘接触之外，增加窗口内 chassis unloading/support redistribution 的可观测门；
- 所有门对 nominal/anomaly、三视角和 30 个场景完整留痕，不按模型正确率筛选。

## 实验分期

### 开发阶段 D1

F42 全部转为开发数据，按 scene/domain 分组做 train/validation/internal-test；任何 split 都不再
称为确认集。先验证：

- T2/T3 family route fidelity 均不低于 0.90；
- Conflict balanced accuracy 不低于 late average，目标差值至少 +0.01；
- Scale 相对当前 learned router 的下降不超过 0.01；
- 11 个 anomaly-only operator 中最差点准确率至少 0.80；
- O9 16 点中最差点准确率至少 0.75，再进入冻结候选。

### 确认阶段 F43

只有 D1 达标后才冻结模型、路由、abstention、所有语义门和统计程序。F43 必须重新使用：

- 全新 scene geometry、PBR material、seed；
- 未在 F42 出现的 operator 参数区间；
- 至少三个生活/生产/野外 domain；
- 完整 T2/T3 matched conflict、11-operator Scale 和直接接触 O9；
- 一次预测、一次揭盲、无结果导向补采。

为控制采集时间，可先进行模型盲的物理功效计算；规模不得由 F43 模型效果决定。推荐保留
15--30 个新场景，并优先保证每个 scene/material/operator 与 T2/T3 cell 的独立物理单位数，
而不是重复渲染同一 episode。

## 明确排除的做法

- 在 F42 上重新搜索 η、类别阈值或路由规则并继续称 F42 为 confirmation；
- 只替换 T2 失败图像或 O9 失败样本；
- 因模型错误率高而删除场景、参数点或 operator；
- 用 texture view 一致性替代 cause cue 的视觉语义可见性；
- 只重训现有 random forest 而不处理局部视觉和相对本体表示。

## 当前状态

F42、A6 和 post-hoc diagnosis 已完成并封存。A4-v8 仍按原 seal 独立采集；其结果无论正负
都进入 A0--A7 v8 ledger。新的方法训练和 F43 采集尚未启动，需在该方案获批后执行。
