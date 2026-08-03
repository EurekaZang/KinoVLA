# Independent Reconfirmation v2：F21–F23 无人值守恢复检查点

> 本文件记录 2026-07-29 对 `confirm_v2_production_scene_05` 的模型盲
> T2 活锁根治和可恢复边界。它不是模型结果，也没有读取预测、标签或分数。

## 当前状态

- 30 个确认场景中已有 **16/30** 个通过 prune receipt 完整封存。
- 当前活动场景是 `confirm_v2_production_scene_06`。
- Scale 的四个分区已完成；T2 已从 47/50 无损恢复到 **50/50**。
- F21 恢复新增的 3 个 T2 case 均在第一次运行中通过。
- 事发前 47 个成功 manifest 的 SHA-256 全部保持不变。
- scene 05 已完成并产生通过的 receipt。
- scene 06 的四个逻辑 scale 分区已启动，共享 FIFO broker 同时只准入
  3 个 Isaac 进程，F21 T2 runner 在同一 broker 上等待可用槽位。
- v8 finalizer 已启动，继续等待 30 个 receipt 和 30 个 conflict capsule。
- 后续所有场景都由 F23 v6 pipeline 启动，T2 继续使用 F21 原子采集器。

## 根因

旧 T2 collector 在一个 Isaac application 中连续执行 50 个 case。scene 01、
scene 00 和 scene 05 均曾在靠后的 case 出现进程仍存活、CPU 忙转、日志和 artifact
停止增长的活锁。旧 F14/F16 只能在事件发生后终止整个长驻进程并从冻结前缀恢复，
没有消除长生命周期状态累积，因此同一失效模式会复发。

scene 05 的冻结事发边界是：

- 47 个连续、完整且 `passed=true` 的正式 case；
- 下一个 case 只有空目录，没有 RGB、proprioception、telemetry 或 manifest；
- 其余 2 个 case 从未启动；
- source log 在干预前超过 600 秒没有写入；
- 没有 reconfirmation prediction、score 或 report。

## F21 根治

F21 不修改冻结 collector 的仿真与科学内容，只改变进程生命周期和写入事务：

1. 每个冻结 T2 case 启动一个全新的 Isaac 进程；
2. collector 仍接收并验证完整冻结 schedule、registry、asset lock 和 protocol；
3. 只在内存中把 schedule 过滤到当前唯一 case；
4. 输出先写入 `/data` 上的 bounded same-filesystem staging；
5. 只有 observables hash、30 个 image hash、case ID 和 scene ID 全部通过后，
   case 目录才以 `os.replace` 原子提交到正式 corpus；
6. 无 manifest 的运行故障在 300 秒硬超时后回收整个进程组，最多进行 3 次
   operational attempt；
7. 已正式接纳的 pass 或 fail 永不重跑，重试与物理结果、模型结果无关；
8. 每 15 秒原子更新 audit heartbeat，并保持原三槽位资源上限。

这把原来的“一个失效可阻塞整场数小时”收敛为“单 case 最迟 300 秒被回收并自动
续接”，同时防止半写 artifact 污染正式数据。

F21 manifest：

`outputs/freeze/unified_moe_v3_reconfirmation_f21_atomic_t2_amendment1/amendment_manifest.json`

SHA-256：

`b2fabf4d8d40c3c691c85c04751b19df3783a54766dbe8c803d58b8b8b0cf624`

## F22 接管边缘修正

第一次 F21 接管已正确终止活锁的 bash/Python Isaac 子进程，但旧
`run_kinofail_reconfirmation_slotted_t2_v1.py` wrapper 仍睡眠等待。F21
controller 按失败关闭原则停止，没有继续采集，也没有修改 47/空/2 的冻结边界。

F22 在该状态下再次封存 corpus inventory 和进程拓扑，精确终止孤儿 wrapper，
然后运行完全相同的 F21 atomic runner。F22 不改变未来 v5 路径，因为 v5 已不再
产生旧 slotted T2 wrapper。

F22 manifest：

`outputs/freeze/unified_moe_v3_reconfirmation_f22_orphan_wrapper_amendment1/amendment_manifest.json`

SHA-256：

`4ba1c106a393650e556385920d8c8828076b11d27bf2eb387f3e48dd806e0a4e`

## F23 successor preflight 顺序修正

scene 05 完成后，F22 controller 按设计启动 v5 all-scenes successor。v5 在
scene 06 启动前先把旧 `SLOTTED_T2` 常量替换为 F21 atomic entrypoint，随后才
调用 F17 的旧环境哈希校验。F17 因而把预期的新入口误判为其冻结依赖被修改，并在
任何 scene 06 采集动作前失败关闭。

F23 在该边界封存以下事实：

- scene 06 没有 pipeline state；
- `/data` 中没有 scene 06 corpus 目录；
- 没有 scene 06 receipt、episode 或正式 observation；
- 没有 prediction、score 或 report；
- 错误来自 successor preflight 的调用顺序，而非 F17/F19 文件变化。

F23 v6 pipeline 先在原始全局状态下执行精确 F17 校验，再缓存该通过结果并安装已
封存的 F21 T2 entrypoint。F19 校验、采集、postprocess 和 prune 链保持不变。

F23 manifest：

`outputs/freeze/unified_moe_v3_reconfirmation_f23_preflight_order_amendment1/amendment_manifest.json`

SHA-256：

`2ec8ec82707f5220befce8afe0819d418c5465605bff784d7ee33bc425af7560`

F21–F23 successor 测试合计 **10/10 passed**。scene 06 恢复后观测到 3 个
fresh Isaac scale pair 满槽运行，GPU 显存约 22.5/32.6 GiB，短窗 SM 利用率为
40%–91%。

## 验证

`tests/test_reconfirmation_f21_atomic_t2.py` 共 **8/8 passed**，覆盖：

- 冻结 schedule 的单 case 唯一选择；
- 正式 corpus 必须是连续 manifest 前缀；
- admitted failure 禁止重跑；
- observables 和 30 张图片的完整 hash 校验；
- 同盘原子提交与源 manifest hash 保持；
- 对假活锁进程组的 TERM/KILL 回收；
- 接管时只匹配真实脚本 token，避免把 shell command string 当成目标进程；
- F21 对全部 successor source 的 hash 认证。

## 无人值守健康判据

正常运行应同时满足：

- `f21_scene05_recovery.json` 在 scene 05 完成前每 30 秒更新 heartbeat；
- 当前 T2 audit 在执行 T2 时每 15 秒更新 heartbeat；
- 同时运行的 Isaac Python 进程不超过 3；
- T2 正式 case 目录只出现完整 manifest，不出现有文件但无 manifest 的半写目录；
- 每个完成场景产生 `prune_receipts/<scene>/completed.json` 和通过验证的 capsule；
- scene 05 receipt 产生后，进程列表中出现
  `run_kinofail_reconfirmation_all_scene_pipelines_v5.py`，后续 T2 子进程为
  `isaac_collect_kinofail_confirmatory_t2_case_f21.py`；
- prediction 和 score 只允许在 30/30 receipt、30/30 capsule 后由 v7 finalizer
  单次生成。

不要重新执行 F21/F22 的一次性 scene 05 recovery controller，也不要原地修改
F21/F22 manifest 中哈希绑定的脚本。若机器掉电，先以 receipt、atomic T2 audit、
launcher audit 和 staging inventory 判断边界，再以新的追加 amendment 恢复。
