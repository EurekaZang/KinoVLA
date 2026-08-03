# KiNO v4 方法改进与开发证据（2026-08-03）

## 状态

- 实验阶段：五折场景不相交开发评估。
- 数据：F42 的 30 个场景、30 个材质簇；F42 已被本轮架构选择使用，因此不再构成独立确认性证据。
- 当前候选：proprioception-default evidence router + high-resolution terrain visual specialist。
- 独立确认：最终结构冻结后仍需在未查看的新场景、新材质、新随机种子和新算子参数区间上一次性评分。

## 当前最优结构

1. 默认本体专家使用 80-D 旋转不变量与 171-D 窗内相对动态，共 251 维；分类器为 512 棵 ExtraTrees。
2. evidence router 只读取同一 251-D 本体特征；阈值为 0.11。场景、材质、算子、battery、标签和结果均不是部署输入。
3. 视觉 specialist 读取五帧 Go2 前视 RGB 的固定下部 55% 区域。图像以 448×448 输入冻结的 `facebook/dinov2-large`，使用 terrain patch mean，并与原有 CLIP mean/final 上下文拼接。
4. 视觉分类器为带标准化的线性 Logistic Regression，`C=0.03`。只有 router 判定视觉证据具有决定性时才覆盖默认本体预测。

## 五折开发结果

| 指标 | 旧冻结 KiNO | KiNO v4-Large | 绝对变化 |
|---|---:|---:|---:|
| Scale balanced accuracy | 68.80% | 97.11% | +28.31 pp |
| Conflict balanced accuracy | 3.70% | 95.07% | +91.37 pp |
| T2 balanced accuracy | — | 90.13% | — |
| T3 balanced accuracy | — | 100.00% | — |
| 等权 battery 宏平均 | 36.20% | 96.09% | +59.89 pp |
| 最差 battery | 3.70% | 95.07% | +91.37 pp |

Conflict router 在 17,580 个样本上的 evidence-route fidelity 为 100%。Scale 中有 174 个保守性误路由，因而最终 Scale 比本体默认值 97.22% 低 0.11 个百分点；所有被误路由的 Scale 真值均位于 T2 specialist 支持集之外，因此当前分类分数不依赖缺失的 Scale DINO cache。

## 视觉瓶颈定位

- 旧 CLIP+DINOv2-Base 全图 specialist 的 T2 为 85.57%。
- 错误集中于第三个合法相机标定：三个 camera profile 的错误率约为 9.1%、6.0% 和 29.8%。
- DINOv2-Base 的默认 224×224 预处理会压缩薄膜褶皱与软地形形变。
- 将固定 terrain 区域以 448×448 输入 DINOv2-Base 后，T2 提升到 88.70%。
- 将冻结编码器替换为 DINOv2-Large 后，T2 进一步提升到 90.13%，最难折达到 86.83%。

## 已排除的方案

| 方案 | 结果 | 决策 |
|---|---|---|
| 在 gate 中加入 CLIP/DINO | 低于 251-D 本体 gate | 排除；视觉不用于判定证据路线 |
| HistGradientBoosting gate | 路由错误多于 ExtraTrees | 排除 |
| RGB ground/route crop CLIP、HOG | T2 约 70%–79% | 排除 |
| pairwise BCE、ranking、centering、GroupDRO | 未稳定超过线性基线 | 排除 |
| frozen patch attention | T2 82.83%，最差折 76.28% | 排除 |
| nonlinear MLP | Base/terrain 均未超过线性 classifier | 排除 |
| camera-profile GroupDRO | T2 87.66%，最差折 81.61% | 排除 |
| 解冻 DINO 最后一层 | 最难外层折 79.78% | 排除；跨场景稳定性下降 |
| 双尺度三模型 ensemble | 仅比单模型约高 0.21 pp | 排除；复杂度与收益不成比例 |
| 只用 Scale 训练本体专家 | T3 降为 0% | 排除；T3 样本对跨分布本体识别必要 |

## 产物

- 当前候选预测：`outputs/eval/kino_conflict_invariant_v4_development_dinov2large_terrain448_eta011/scene_disjoint_predictions.jsonl`
- 当前候选报告：`outputs/eval/kino_conflict_invariant_v4_development_dinov2large_terrain448_eta011/report.json`
- Large T2 特征：`outputs/eval/kino_t2_dinov2_large_448_terrain_v4_development/features.npz`
- Base 公平基线：`outputs/eval/kino_v4_fair_baselines_terrain448_detailonly_eta011_development/report.json`
- Base 配对统计：`outputs/eval/kino_v4_paired_statistics_terrain448_detailonly_eta011_development/report.json`

Large 公平基线和配对统计完成后，才能冻结最终候选及独立确认协议。
