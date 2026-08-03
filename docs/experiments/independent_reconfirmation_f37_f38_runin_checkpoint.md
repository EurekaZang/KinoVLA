# 独立确认 F37–F38 run-in 恢复检查点

更新时间：2026-08-01 20:16 EDT（2026-08-02 00:16 UTC）

## 当前结论

扩大版 Scale、direct-O9 与原始 T2 数据已经完成；最终 A0–A7 尚未完成。当前正在重新采集与冻结
C2-T3 的物理确认数据。原因不是模型结果不理想，而是在任何预测或 truth key 生成前发现：旧的
reachable-region T3 collector 从首帧即把 Go2 footprint 放进异常区，使原冻结 v5 特征所要求的
21-sample、接触前后时间窗无法成立。

F36 与 F37 均保留为 result-blind 基础设施/输入合同失败，不作为科学结果，也不覆盖或删除：

- F36：`Path.rglob` 不进入 episode-directory symlink；
- F37：修复 symlink 枚举后，2,864/2,864 个有效 T3 anomaly source episode 均因首帧已接触而无法
  形成冻结时间窗；
- F36/F37 均没有 prediction、truth key、score、confirmatory report 或 finalization audit。

F37 失败审计：
`outputs/kinofail_reconfirmation_f37_failure_audit_v1/audit.json`。

## 模型盲开发 pilot

所有 pilot 都是 development-only，`counts_as_confirmatory_evidence=false`，且永久从正式 successor
中排除。

1. P1：三域各 1 case，区域中心 1.25 m、半长 0.55 m。6/6 anomaly 均可提取冻结时间窗，但只有
   2/6 pair 通过算子 QA；区域对部分轨迹过远。
2. P2：三域各 1 个新 case，近边界冻结为 0.52 m。O7 三域 3/3 通过；O8 实际在约 0.18 m 被透明
   障碍挡住，但旧 QA 以 base-center + 0.25 m 判断接触，3/3 被误记为未触发。
3. P3：三域各 1 个新 case；物理区域不变，只把 O8 exposure QA 对齐到 Go2/frozen-v5 共用的
   0.35 m footprint margin。O7 3/3、O8 物理 engagement 3/3；唯一失败是 production 的一个既有
   RGB swap effect 门，按原规则计为数据质量 attrition，不是物理或时间窗失败。

正式冻结合同因此为：region center 0.82 m、half-length 0.30 m、near edge 0.52 m、O8 footprint
margin 0.35 m；v5 特征函数、模型、router、阈值、统计检验、仿真算子、碰撞几何、相机与外观干预
均不改。

## 正式 F38 与并发恢复

最初正式 seal 错误地冻结到 6 个并行 Isaac 进程。32 GB GPU 首批出现 OOM；在第 13 个 pair 启动前
停止队列。已触及的 12 pairs 恰好属于 6 个完整 case，全部排除且不重试。失败批次完整保留于：

- `outputs/kinofail_t3_runin_f38_formal/`
- `/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_formal/corpus/`
- `outputs/kinofail_t3_runin_f38_concurrency_failure/audit.json`

三并发 successor S1 已在采集前冻结：

- seal：`outputs/freeze/kinofail_t3_runin_f38_formal_s1/seal_manifest.json`
- formal output：`outputs/kinofail_t3_runin_f38_formal_s1/`
- corpus：`/data/eureka/KinoVLA/outputs/kinofail_t3_runin_f38_formal_s1/corpus/`
- watchdog：`outputs/kinofail_t3_runin_f38_s1_watchdog/state.json`
- planned design：1,500 cases；
- development-excluded：9 cases；
- six-process-touched excluded：6 cases；
- S1 selected：1,485 cases / 2,970 counterfactual pairs / 5,940 physical episodes；
- minimum accepted：1,426 cases，即总 attrition 仍必须严格小于 5%；
- fresh Isaac process per pair，terminal pair 不重试。

截至本检查点，S1 已终止 6 pairs，6/6 通过；另有 3 pairs 在运行。平均每 pair 约 74.5 s，三进程
峰值显存约 20.8–22.2/32.6 GB。按当前吞吐，物理采集约需 20–21 小时。`/data` 仍有约 1.3 TB
可用空间。

## 采集完成后的固定顺序

1. 依据 S1 final audit 构建模型盲有效 T3 design 与原 3,000-case conflict truth schedule；
2. 在任何预测前冻结新的 F39 seal；
3. 只运行一次 F39 blind prediction 与 confirmatory scoring；
4. 使用同一 F39 prediction/truth 更新 A6；
5. 串行执行已冻结的正式 A4-v8（O4/O5，50 cases、350 action episodes）；
6. 汇总同一证据哈希下的 A0–A7 v7 fail-closed ledger；
7. 统计门失败时保留负结果，不调参、不重跑、不以旧数据替换。

