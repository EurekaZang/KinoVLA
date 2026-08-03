# Independent Reconfirmation v2：可恢复进度检查点

> 本文件是切换到 website / PPT presentation 工作前的持久化实验记忆。  
> 快照时间：2026-07-28 04:27 EDT（2026-07-28 08:27 UTC）。  
> 后台采集会继续推进；活动场景的瞬时计数只用于恢复定位，最终状态必须重新读取权威 artifact。

## Material Passport

- `schema_version`: `kinofail.independent-reconfirmation-progress-checkpoint.v2`
- `verification_status`: `RUNNING_MODEL_BLIND`
- `scope`: 独立确认性扩展的采集、后处理、封存和恢复状态
- `authoritative_roots`:
  - `outputs/kinofail_reconfirmation_v2`
  - `outputs/eval/unified_moe_v3_reconfirmation_v2/shards`
  - `outputs/freeze/unified_moe_v3_reconfirmation_*`
- `predictions_or_scores_used`: `false`
- `checkpoint_is_final_result`: `false`

## 1. 当前稳定边界

- 30 个全新确认性 scene shard 中，已有 **11/30** 完成完整流水线。
- 已封存场景为：
  - `confirm_v2_life_scene_00` 至 `confirm_v2_life_scene_09`
  - `confirm_v2_production_scene_00`
- 当前活动场景为 `confirm_v2_production_scene_01`，于
  2026-07-28 04:26 EDT 启动。
- 已完成集合的 capsule、prune receipt 和 F13 validation 均为
  **11/30**，场景集合完全一致；11 个 F13 validation 全部
  `status: complete, passed: true`。
- 冻结模型尚未加载；确认性目录中 prediction、score 和最终 report
  artifact 计数仍为 **0**。
- O9 high-centering semantic repair 仍被有意暂停，避免与主确认集争抢显存。

## 2. 已完成场景

| Scene shard | Capsule | Prune receipt | F13 validation |
|---|---:|---:|---:|
| `confirm_v2_life_scene_00` | complete | complete | pass |
| `confirm_v2_life_scene_01` | complete | complete | pass |
| `confirm_v2_life_scene_02` | complete | complete | pass |
| `confirm_v2_life_scene_03` | complete | complete | pass |
| `confirm_v2_life_scene_04` | complete | complete | pass |
| `confirm_v2_life_scene_05` | complete | complete | pass |
| `confirm_v2_life_scene_06` | complete | complete | pass |
| `confirm_v2_life_scene_07` | complete | complete | pass |
| `confirm_v2_life_scene_08` | complete | complete | pass |
| `confirm_v2_life_scene_09` | complete | complete | pass |
| `confirm_v2_production_scene_00` | complete | complete | pass |

权威位置：

- Capsule：
  `outputs/kinofail_reconfirmation_v2/conflict_capsules_ext4/<scene>/capsule_manifest.json`
- Prune receipt：
  `outputs/kinofail_reconfirmation_v2/prune_receipts/<scene>/completed.json`
- F13 validation：
  `outputs/kinofail_reconfirmation_v2/f13_validations/<scene>/validation.json`
- Pipeline state：
  `outputs/kinofail_reconfirmation_v2/orchestration/<scene>/pipeline_state.json`

## 3. production_scene_00 的最终封存状态

`confirm_v2_production_scene_00` 已于
2026-07-28 04:26 EDT 完成 acquisition、model-blind postprocess、
conflict capsule 和 prune。

| Battery | 冻结计划 | 已产生完整 summary | Retained / passed | 预注册 attrition |
|---|---:|---:|---:|---:|
| Scale | 352 pairs | 352 | 296 | 56 |
| T2 | 50 cases | 50 | 50 cases / 300 samples | 0 |
| T3 | 100 pairs | 100 | 98 | 2 |

这些 attrition 是 launcher eligibility / operator-local QA 的模型盲结果，
没有根据模型预测、分类得分或效果强弱重试。全局 attrition gate 仍由最终
finalizer 在 30/30 场景完成后执行。

权威 capsule：

`outputs/kinofail_reconfirmation_v2/conflict_capsules_ext4/confirm_v2_production_scene_00/capsule_manifest.json`

其状态为 `status: complete, passed: true`，包含 T2 的 50 cases / 300
samples，以及 T3 的 98 retained anomaly episodes。

## 4. F16 T2 活锁恢复记录

`confirm_v2_production_scene_00` 的原 T2 Isaac 进程在连续写出 30 个通过
manifest 后发生模型盲活锁。恢复前状态严格划分为：

- 30 个连续、已通过且有 manifest 的 case；
- 1 个已创建目录但文件数为 0 的下一顺序 case；
- 19 个从未尝试的 case。

F16 在停止旧进程前封存上述状态，只允许：

- 逐字节保留原 30 个 manifest；
- 重新进入唯一的零观测 case；
- 继续 19 个从未尝试 case；
- 使用冻结 collector 已存在的 `--resume` 路径；
- 在 T2 恢复通过前暂停 scene pipeline，避免提前并发启动 T3。

恢复结果：

- T2 从 30/50 补齐至 50/50；
- summary `passed: true`；
- 原 30 个 manifest 哈希全部保持不变；
- recovery collector return code 为 0；
- scene pipeline 恢复后完成 T3、后处理、capsule 和 prune。

权威 F16 amendment：

`outputs/freeze/unified_moe_v3_reconfirmation_f16_t2_production00_liveness_amendment1/amendment_manifest.json`

原 recovery audit 和 recovery log 已随场景 prune 被清理，但其路径、字节数和
SHA-256 被保存在：

`outputs/kinofail_reconfirmation_v2/prune_receipts/confirm_v2_production_scene_00/raw_inventory.jsonl`

其中：

- `c2_t2/launcher_audits/confirm_v2_production_scene_00_f16_resume.json`
  - bytes: `20682`
  - SHA-256:
    `b04bac8824286a7028331ba236edaa6d851330b5296cc2d178c6f5df3ce436da`
- `c2_t2/launcher_logs/confirm_v2_production_scene_00_f16_resume.log`
  - bytes: `89304`
  - SHA-256:
    `63358a6159e73597508eb6f7a65650f306f5c82b497c8c4b79279c8f2fe91ff8`

恢复实现与测试：

- `scripts/recover_kinofail_reconfirmation_t2_f16.py`
- `tests/test_reconfirmation_f16_t2_recovery.py`
- 3 项恢复状态测试全部通过。

## 5. 当前活动场景

Scene：`confirm_v2_production_scene_01`

快照时状态（2026-07-28 04:27 EDT）：

| Battery | 已完成 | 通过 | 计划总量 | 状态 |
|---|---:|---:|---:|---|
| Scale | 0 | 0 | 352 pairs | 两个 partition collector 已启动 |
| T2 | 2 | 2 | 50 cases | 正在持续采集 |
| T3 | 0 | 0 | 100 pairs | 等待 Scale/T2 阶段结束 |

快照时 scene pipeline 的 `model_checkpoint_loaded` 为 `false`，stage ledger
尚未写入完成阶段。活动进程结构为：

- 1 个 scene pipeline；
- 1 个 T2 Isaac collector；
- 2 个 Scale pair Isaac collectors；
- 总 Isaac 并发数为冻结上限 3。

快照资源：

- GPU 显存：约 `22.97 / 32.61 GB`
- GPU utilization：约 `88%`
- `/home`：819 GB 总量，682 GB 已用，96 GB 可用，使用率 88%
- `outputs/kinofail_reconfirmation_v2`：约 15 GB
- `outputs/eval/unified_moe_v3_reconfirmation_v2/shards`：约 647 MB

## 6. 冻结边界

- 当前确认性扩展仍是 model-blind acquisition / postprocess。
- `outputs/kinofail_reconfirmation_v2` 下没有 prediction、score 或
  `confirmatory_report.json`。
- 预测、评分、五个冻结 checkpoint 和统计检验必须等 30/30 shard 完成后由
  finalizer 一次性执行。
- 在 finalizer 之前不得修改 architecture、routing features、默认
  proprioception policy、阈值 η、评价指标或统计分析。
- 不得因单个 pair 未通过 eligibility/QA 而结果依赖重试。

## 7. 会话切换后的核验与恢复

首先执行只读核验：

```bash
ps -eo pid,ppid,stat,etime,%cpu,rss,cmd |
  rg 'run_kinofail_reconfirmation|isaac_collect_kinofail_confirmatory'

find outputs/kinofail_reconfirmation_v2/prune_receipts \
  -mindepth 2 -maxdepth 2 -name completed.json -type f | wc -l
```

处理规则：

1. 若 `run_kinofail_reconfirmation_all_scene_pipelines_v2.py` 和活动 scene
   pipeline 仍存在，只监控，不启动第二套 collector。
2. 读取最新未完成 scene 的
   `outputs/kinofail_reconfirmation_v2/orchestration/<scene>/pipeline_state.json`。
3. 核对 capsule、prune receipt、F13 validation 的已完成 scene 集合一致。
4. 确认 prediction/score/report artifact 仍为零。
5. 只有确认整个主 orchestration 已退出、且没有任何 Isaac collector 后，才可用
   下列幂等入口恢复未完成场景：

```bash
/home/eureka/miniconda3/envs/kinovla/bin/python \
  scripts/run_kinofail_reconfirmation_all_scene_pipelines_v2.py \
  --exclude-scene confirm_v2_life_scene_00
```

不要在主确认性扩展完成前恢复 O9 双 shard 采集。

## 8. Website / PPT presentation 的使用边界

- 可准确表述为：
  “A fully frozen, model-blind independent confirmation extension is in
  progress; 11 of 30 new scenes were sealed as of 2026-07-28 04:27 EDT.”
- 不能把 11/30 的中间数据当作最终确认性结果，也不能据此宣称新扩展已经支持或
  否定论文结论。
- Website/PPT 可使用已经冻结的 A0–A7 论文结果；若提到本扩展，应明确标注
  `ongoing independent confirmation`。
- F16 是实验完整性记录，不应占据主叙事版面；只有在方法学审计、补充材料或
  审稿质疑独立性时才需要展开。
