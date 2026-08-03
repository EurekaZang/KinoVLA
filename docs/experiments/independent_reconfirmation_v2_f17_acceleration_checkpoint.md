# Independent Reconfirmation v2：F17 无损加速切换检查点

> 快照时间：2026-07-28 16:49 EDT（20:49 UTC）。  
> 本文件记录模型盲确认集从旧三 worker 调度切换到 F17 三槽位 FIFO
> 调度的可恢复边界；它不是最终实验结果。

## 当前状态

- 30 个确认场景中已有 **14/30** 完成完整流水线：
  `confirm_v2_life_scene_00`–`09` 与
  `confirm_v2_production_scene_00`–`03`。
- 14 个完成场景的 prune receipt、conflict capsule 和 F13 validation
  集合一致，均为 complete/pass。
- `confirm_v2_production_scene_03` 是旧 v2 pipeline 完成的最后场景，
  receipt 时间为 2026-07-28 16:42 EDT。
- `confirm_v2_production_scene_04` 是 F17 v3 pipeline 的第一个场景，
  于 2026-07-28 16:44 EDT 启动。
- 冻结 checkpoint 尚未运行；prediction、score 与
  `confirmatory_report.json` 仍为 0。
- v4 finalizer 已通过预检并处于
  `waiting_for_30_model_blind_scene_shards`。

## F17：保持科学协议不变的工作保守调度

旧 pipeline 将 T2 和四个 scale runner 放入三个长生命周期 worker。
任务队尾的第四个 scale partition 经常只能在其他 partition 完成后启动，
从而产生长时间单路尾段。F17 改为：

- 保留冻结的四个逻辑 partition 及其 membership；
- 保留每个 partition 内的 pair 顺序；
- 保留每个 counterfactual pair 一个 fresh Isaac process；
- 保留 scale→T3 stage barrier；
- 最大同时运行的 Isaac process 仍严格为 3；
- 五个初始逻辑 runner 和四个 T3 runner 通过进程共享的 FIFO 三槽位池
  逐 pair 获得执行机会。

没有减少 episode、物理步数、相机视角、RTX/PBR 渲染、proprioception
时间序列或统计样本；没有新增失败重跑。scene 04 的现场核验显示：

- 四个 scale partition 均已取得 FIFO ticket；
- 实际 Isaac Python process 数为 3；
- ticket 按 0、1、2、3、4、5… 顺序放行；
- 新样本持续写入 `/data`；
- GPU 在三进程下保持稳定。

F17 manifest：

`outputs/freeze/unified_moe_v3_reconfirmation_f17_work_conserving_storage_amendment1/amendment_manifest.json`

SHA-256：

`4323597aebcb2b494fa7d1d0d343c897ba75966a008b729287af4eea94eb6a68`

封存状态：

`sealed_after_scene03_before_scene04_acquisition`

## Scratch 迁移

scene 03 完成后，原始 35.8 GB shard 已由既有 seal-and-prune 流程合法清理。
在没有活动 writer 且 scene 04 尚未创建的边界执行了最终 rsync：

- 源和目标 dry-run difference count：0；
- 边界时两端均为 0 个 scratch 文件；
- `/home/eureka/KinoVLA/outputs/kinofail_reconfirmation_v2/corpus_ext4`
  已切换为指向
  `/data/eureka/KinoVLA/outputs/kinofail_reconfirmation_v2/corpus_ext4`
  的符号链接；
- scene 04 的 collector 命令与实际文件路径均解析到 `/data`；
- 切换后 `/home` 可用约 213 GB，`/data` 可用约 1.6 TB。

冻结的 derived shards、capsules、receipts、inventories 和 schedules 没有移动
或改写。

## F18：finalizer 重启预检恢复

旧 finalizer 在 scene 01 prune 之前启动，因此已在内存中读过 F14 recovery
audit。F17 切换时重启 finalizer 后，v2 入口重新读取该已被合法 prune 的临时
audit，因而在推理前触发 `FileNotFoundError`。

scene 01 的 `raw_inventory.jsonl` 保留了该文件的精确认证信息：

- bytes：16,279；
- SHA-256：
  `88f36e352eb94a7fbea26e28bb86ac25157b871f6e71f3edafd25c4abf9761e0`。

本地执行记录保留了完整 JSON。按原 writer 的
`json.dumps(..., indent=2, sort_keys=True) + "\n"` 规范化后，字节数和
SHA-256 与 prune inventory 完全一致。F18 只把 finalizer 的 F14 预检读取
路径指向这一认证缓存；后续 merge、blind bundle、唯一一次预测、truth join
与冻结 scorer 均继续调用原 v2/v3 finalizer。

F18 manifest：

`outputs/freeze/unified_moe_v3_reconfirmation_f18_pruned_audit_cache_amendment1/amendment_manifest.json`

SHA-256：

`8e5f78b891ae93e00699a5be124fea0e9d64439b500912fdb1a6347298094eaa`

封存状态：

`sealed_during_scene04_acquisition_before_any_prediction`

## 验证

以下测试合计 **12/12 passed**：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/eureka/miniconda3/envs/kinovla/bin/python -m pytest -q \
  tests/test_reconfirmation_f18_finalizer_restart.py \
  tests/test_reconfirmation_f17_work_conserving.py \
  tests/test_reconfirmation_f13_launcher_eligibility.py \
  tests/test_reconfirmation_f15_partition_supersession.py
```

覆盖：

- 三槽位并发上限；
- FIFO admission；
- 四个原 partition 的 membership/count；
- 原科学 runner 的直接复用；
- 生产命令入口与 live broker 连接；
- F14 cache 的 byte-for-byte 认证；
- F17/F18 manifest 及脚本哈希链。

## 恢复入口与日志

主流水线：

```bash
/home/eureka/miniconda3/envs/kinovla/bin/python \
  scripts/run_kinofail_reconfirmation_all_scene_pipelines_v3.py
```

日志：

`outputs/kinofail_reconfirmation_v2/orchestration_logs/f17/remaining_scenes_pipeline.log`

finalizer：

```bash
/home/eureka/miniconda3/envs/kinovla/bin/python \
  scripts/finalize_kinofail_reconfirmation_v4.py --poll-seconds 30
```

日志：

`outputs/kinofail_reconfirmation_v2/orchestration_logs/f17/finalizer_v4.log`

恢复前必须先确认没有第二套 all-scenes pipeline、scene pipeline、Isaac
collector 或 finalizer。不要修改 F17/F18 manifest 中哈希绑定的脚本。
