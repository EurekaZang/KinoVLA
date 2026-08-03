# A0–A7 中 O9 High-Centering 的语义审计与修复

日期：2026-07-27  
范围：仅 O9 `High-Centering`；A8 不在范围内。

## 结论

旧 O9 的物理准入条件错误地把“进入障碍区域并产生测量”当作
High-Centering。High-Centering 实际要求横向障碍持续承托 Go2
底盘/腹部并造成足端部分卸载；头部或腿部正面撞击不属于该异常。

因此，旧 O9 不能继续支撑 11 算子总体结论。问题被限定在 O9：
A1 不使用 O9，A6 的主实验仅使用 O1/O2/O4/O5/O10，二者不受影响；
其余使用 11 类总体指标或 O9 单元格的实验，需要在替换 O9 后重算。

## 旧数据全量审计

机器可读报告：

- `outputs/eval/realistic_a0_a7_v6/o9_semantic_audit_v1.json`
- `outputs/eval/realistic_a0_a7_v6/o9_recollection_required_v1.jsonl`

审计结果：

| 数据来源 | O9 单元 | 直接语义通过 | 处理 |
|---|---:|---:|---|
| A0/A2/A3/A5/A7 共用 frozen scale snapshot | 90 physical pairs / 540 records | 无法直接证明 | 90 pairs 全部重采 |
| A4 actual action | 10 cases / 20 action rows | 0 | 排除并重采 |
| A5 alternate realization | 90 anomaly manifests | 0 | 排除并重采 |
| A5 C4 direct | 15 source episodes / 30 branches | 0 | 排除并重采 |

scale-v8 的 180 个原始 O9 manifest 已按此前磁盘清理要求删除，只保留
pair summary 和 frozen snapshot。snapshot 中的 19-D proprioception
不能区分接触 body link，因此不允许用 support ratio 代理腹部接触。
代理检查还显示：90 个 anomaly 中 23 个在 decision window 内从未出现
`support_ratio <= 0.5`，86 个少于 5 帧，进一步说明不能把它们当作
High-Centering 的直接证据。

保留了 privileged telemetry 的实验给出一致结论：

- A4：20/20 action rows 没有腹部接触，18/20 是头部/肢体单独碰撞；
- A5 alternate realization：90/90 没有持续腹部接触；
- C4 direct：30/30 没有持续腹部接触。

## 新语义门控

实现：

- `kino_vla/data/o9_semantics.py`
- `tests/test_o9_semantics.py`

必须同时满足：

1. base/belly 最大接触力不少于 20 N；
2. 连续腹部接触不少于 10 control steps；
3. 腹部接触 duty cycle 不少于 0.10；
4. 测得的最小足端支撑比例不高于 0.75；
5. 不能是 head/limb-only collision；
6. 有位姿证据时，障碍必须位于底盘纵向中心下方。

单元测试覆盖“持续腹部承托应通过”和“头部单独碰撞应拒绝”，结果
为 `2 passed`。

## Fig.2 修复

Fig.2 的 O9 已重采为横梁位于 Go2 底盘下方的 `pallet_edge`，并通过
上述语义门控。图像 manifest、aggregate audit 和论文图 generator
之间增加了逐文件哈希绑定，避免图源更新后沿用旧审计结论。

论文产物：

- `/home/eureka/KiNO-Paper/figures/fig2_operator_gallery.pdf`
- `/home/eureka/KiNO-Paper/figures/fig2_operator_gallery.png`
- `/home/eureka/KiNO-Paper/figures/fig2_operator_gallery_provenance.json`

## 最终重采协议

最终版本为 v4。所有 pilot 均被显式排除，不进入训练或评测；pilot
只使用 body-link contact 和 nominal route-stability telemetry，
没有查看归因模型预测。

冻结文件：

- `outputs/kinofail_realistic/design_o9_semantic_repair_v4/schedule.jsonl`
- `outputs/kinofail_realistic/design_o9_semantic_repair_v4/design_audit.json`
- `configs/data/kinofail_realistic_o9_semantic_repair_formal_v4.json`
- `scripts/isaac_collect_kinofail_realistic_o9_semantic_repair_v4.py`

固定设计：

- 90 个原 scale-v8 O9 counterfactual pairs 全量替换；
- 原场景、split、domain、材质、三 appearance views、相机、随机种子和
  五组 physical nuisance 身份保持不变；
- nominal/anomaly 使用相同起点、姿态和 `0.08 m/s` 诊断爬行指令；
- 障碍为与行进方向正交的 `pallet_edge`；
- moderate：高度 0.36 m、宽度 0.12 m、residual target 0.46；
- severe：高度 0.36 m、宽度 0.14 m、residual target 0.22；
- 物理准入必须通过新 O9 semantic gate。

v4 moderate pilot 已通过：nominal 360 steps 内未跌倒；anomaly 通过
完整腹部接触/足端卸载门控。剩余 89 pairs 已由两个 shard 开始采集，
运行入口为：

`scripts/run_kinofail_realistic_o9_semantic_repair_v4.py --shards 2`

完成后，必须用 v4 O9 覆盖旧 O9，重新生成 snapshot，并重跑
A0/A2/A3/A4/A5/A7；在此之前，不再把旧 O9 或旧 11 类 macro 指标
写成 publication-ready 证据。
