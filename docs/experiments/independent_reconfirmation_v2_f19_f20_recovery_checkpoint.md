# Independent Reconfirmation v2：F19/F20 恢复检查点

> 快照时间：2026-07-29 04:12 EDT（08:12 UTC）。  
> 本文件记录 `production_scene_04` 后处理故障的模型盲恢复，以及从
> `production_scene_05` 继续采集的可恢复边界；它不是最终实验结果。

## 当前状态

- 30 个确认场景中已有 **15/30** 完成完整流水线：
  `confirm_v2_life_scene_00`–`09` 与
  `confirm_v2_production_scene_00`–`04`。
- 15 个完成场景的 prune receipt、conflict capsule 和 F13 validation
  集合一致，均为 complete/pass。
- 当前活动场景为 `confirm_v2_production_scene_05`。
- v4 all-scenes pipeline 正在运行；它根据 receipt 跳过前 15 个场景，从
  scene 05 开始继续。
- v6 finalizer 已通过 F19/F20 预检，当前处于
  `waiting_for_30_model_blind_scene_shards`。
- 冻结模型、预测、评分和最终统计尚未执行；当前状态不得作为确认性结果解读。

## scene 04 的故障与 F19 恢复

scene 04 的物理采集、Scale 后处理和 T3 后处理已经完成，但冻结 T2 特征入口
把 corpus 符号链接解析到 `/data` 后，尝试相对
`/home/eureka/KinoVLA` 计算 provenance 路径，因而在
`t2_features` 阶段触发 `ValueError`。故障发生在载入模型或预测之前。

F19 只增加存储感知的路径适配：

- 保持逻辑路径与物理路径具有相同的 repo-relative path；
- 直接复用冻结 T2 特征实现和特征顺序；
- 不改变特征值、模型、路由、阈值、统计、仿真、传感器或 schedule；
- pruner 仅接收解析后的 `/data` corpus 路径。

一次性恢复只执行了缺失的 T2 特征、冲突证据胶囊和安全裁剪，没有重新执行
scene 04 物理采集、Scale 后处理或 T3 后处理。终态审计为：

- `state: terminal`
- `passed: true`
- `physical_acquisition_reexecuted: false`
- `scale_or_t3_postprocess_reexecuted: false`
- `model_or_prediction_loaded: false`
- `scientific_content_changed: false`
- `result_dependent_retry_or_selection: false`

scene 04 已形成：

- T2：50 cases / 300 samples；
- T3：100 retained anomaly episodes，0 excluded；
- conflict capsule：3,400 files / 748,594,440 bytes；
- prune receipt：complete/pass；
- 原始 scene 04 scratch shard 已从 `/data` 安全裁剪。

权威文件：

- F19 manifest：
  `outputs/freeze/unified_moe_v3_reconfirmation_f19_storage_postprocess_amendment1/amendment_manifest.json`
- F19 manifest SHA-256：
  `af9211d8fe1ad876b8aa7a3e3941f7d60b731879d4fb5fb77a7a006a28ed722a`
- scene 04 recovery audit：
  `outputs/kinofail_reconfirmation_v2/orchestration/confirm_v2_production_scene_04/f19_storage_postprocess_recovery.json`
- scene 04 pipeline state：
  `outputs/kinofail_reconfirmation_v2/orchestration/confirm_v2_production_scene_04/pipeline_state.json`
- scene 04 prune receipt：
  `outputs/kinofail_reconfirmation_v2/prune_receipts/confirm_v2_production_scene_04/completed.json`

## F20 finalizer 包装修复

首次启动 F19-aware v5 finalizer 时，包装层错误引用了不存在的
`finalize_v4.base` 属性。异常发生在调用 predecessor finalizer 的
`main()` 之前，因此没有进入 30-scene 等待、特征合并、模型推理、truth join、
评分或统计阶段。

F20 新增 v6 包装器，只把模块引用修正为冻结链中的
`finalize_v4.predecessor.base`，继续调用原 v4 finalizer 的 `main()`。
它不修改任何科学步骤，也不修改已封存的 F19/v5 文件。

权威文件：

- F20 manifest：
  `outputs/freeze/unified_moe_v3_reconfirmation_f20_finalizer_reference_amendment1/amendment_manifest.json`
- v5 失败日志：
  `outputs/kinofail_reconfirmation_v2/orchestration_logs/f19/finalizer_v5.log`
- 活动 v6 finalizer 日志：
  `outputs/kinofail_reconfirmation_v2/orchestration_logs/f19/finalizer_v6.log`

## scene 05 现场健康检查

快照时：

- 1 个 all-scenes runner、1 个 scene pipeline；
- T2 与四个原始 Scale logical runners 均存在；
- FIFO 三槽位池维持最多 3 个 Isaac Python process；
- 快照实例为 1 个 T2 collector + 2 个 Scale pair collectors；
- 已观察到 ticket `0`–`8` 按 FIFO 顺序放行；
- 新 pair summary、launcher audit、RTX 图像和 proprioception 数据持续写入
  `/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_v2/corpus_ext4/confirm_v2_production_scene_05`；
- GPU utilization 约 81%，显存约 21.1/32.6 GB；
- `/home` 可用约 211 GB，`/data` 可用约 1.6 TB。

该调度仍保留四个逻辑 partition、partition 内顺序、每个 counterfactual pair
独立启动 Isaac、Scale→T3 barrier，以及最多 3 个 Isaac 实例的冻结约束。

## 验证

F13、F15、F17、F18、F19 和 F20 相关测试合计 **20/20 passed**：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/eureka/miniconda3/envs/kinovla/bin/python -m pytest -q \
  tests/test_reconfirmation_f19_storage_postprocess.py \
  tests/test_reconfirmation_f18_finalizer_restart.py \
  tests/test_reconfirmation_f17_work_conserving.py \
  tests/test_reconfirmation_f13_launcher_eligibility.py \
  tests/test_reconfirmation_f15_partition_supersession.py \
  tests/test_reconfirmation_f20_finalizer_reference.py
```

## 当前入口与恢复规则

主流水线：

```bash
/home/eureka/miniconda3/envs/kinovla/bin/python \
  scripts/run_kinofail_reconfirmation_all_scene_pipelines_v4.py
```

日志：

`outputs/kinofail_reconfirmation_v2/orchestration_logs/f19/remaining_scenes_pipeline.log`

等待型 finalizer：

```bash
/home/eureka/miniconda3/envs/kinovla/bin/python \
  scripts/finalize_kinofail_reconfirmation_v6.py --poll-seconds 30
```

日志：

`outputs/kinofail_reconfirmation_v2/orchestration_logs/f19/finalizer_v6.log`

恢复前必须先确认没有第二套 all-scenes pipeline、scene pipeline、Isaac
collector 或 finalizer。已封存的 F19/F20 哈希绑定脚本不得原地修改；若再次
发现操作性错误，必须在没有查看模型结果的边界追加后继 amendment。
