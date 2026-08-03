# Kino-Fail realistic O4 v31 冻结确认与失败归因

更新时间：2026-07-23

## 结论

v31 是对 v27 phase-conditioned Continue/Backstep 架构的首次全新、operator-blind realistic
cohort 确认。六个场景覆盖 Bedroom、House、LivingRoom、Office 四个 EmbodiedGen room family，
场景准入在 O4 运行前冻结；每个场景只运行一次，全部结果均保留。

- 证据完整性：`6/6`，`integrity_passed=true`。
- 严格 outcome：`1/6`，`outcome_passed=false`。
- 封存状态：`sealed=true`。
- 该结果不属于 A0–A7；realistic A0–A7 readiness 仍为 `0/8`。
- v31 六个场景不得重跑、调参、删去或转作下一次确认集。

封存入口：

- 配置：`configs/data/kinofail_o4_action_consequence_v31_heldout_confirmation_v1.json`
- preflight：`outputs/kinofail_realistic/operator_confirmation/o4_action_v31_preflight/preflight_audit.json`
- runner：`outputs/kinofail_realistic/operator_confirmation/o4_action_v31_heldout_confirmation/runner_audit.json`
- postrun：`outputs/kinofail_realistic/operator_confirmation/o4_action_v31_heldout_confirmation/postrun_audit.json`

对应 SHA-256 为：

- config：`dc8a707046984b184433db36937a59ece1149cc885c32036a6a8414ae028f665`
- preflight：`9f026741bc74e5c88ead62ef8b4d1142caae1569a1df5153f9e55307a90493bf`
- runner：`26d88af9f761d74216b13f5bf723db4372904644e056e96e4185c13bad099410`
- postrun：`9657f2a9ac97d59f42ad18232bfebe756a33c12ec890bb2d435d895fb6b9caa4`

## 逐场景结果

| 场景 | 触发 | 决策 step | Backstep 后退 | Continue | Backstep | 主要失败门 |
|---|---:|---:|---:|---:|---:|---|
| Office90 | maximum dwell | 76 | 0.267 m | 跌倒 | 安全且 clearance 完成 | Continue 倒地后的 outcome 图像未过通用纹理门 |
| LivingRoom91 | posture urgency | 83 | 0.000 m | 跌倒 | 跌倒 | 分支过晚；无有效撤离并越出 lateral budget |
| Office96 | maximum dwell | 76 | 0.247 m | 跌倒 | 安全且 clearance 完成 | 无，严格通过 |
| Bedroom100 | maximum dwell | 94 | 0.209 m | 跌倒 | 安全 | 后退上限后仍有 1 只 active foot，clearance 未完成 |
| Office102 | maximum dwell | 95 | 0.279 m | 跌倒 | 安全 | 后退上限后仍有 1 只 active foot；Continue 倒地 outcome 低纹理 |
| House105 | posture urgency | 87 | 0.000 m | 跌倒 | 跌倒 | 分支过晚；无有效撤离并越出 lateral budget |

所有六例均满足同 seed、同 USD、同 operator、同首次 attachment、同 decision step、决策前
telemetry byte-identical；Continue 也均在 decision 之后跌倒。因此负结论不是反事实不匹配，而是
恢复策略和通用图像门在新场景上的真实泛化失败。

## 根因分解

### 1. 紧急分支仍然过晚

v27 要求 attachment 后至少等待 5 steps，并要求姿态紧急连续 2 steps。LivingRoom91 和
House105 都在首次 attachment 后 6 steps 才分支，此时机身已经进入不可由当前速度命令挽回的
下坠/倾覆状态。两例均触发真实 peel，但 Backstep 仍跌倒。这说明“检测到了 urgency”不等于
“仍处于可恢复域”。

### 2. 固定 120-step 上限不足以保证离开局部 adhesion patch

Bedroom100 与 Office102 分别已经后退 0.209 m 和 0.279 m，未跌倒且路线偏差合格，但在上限
到达后进入 hold。前足仍可能位于长 0.96 m 的局部 adhesion region 内，随后继续发生 peel /
reattach cycle；最终 `active_feet=1`，连续 5 steps 的真实 clearance 从未成立。正确终止条件应
由“目标撤离距离 + active feet 连续清零”决定，并允许在剩余 fixed horizon 内持续后退；不能
以与 hazard 几何和 Go2 footprint 无关的固定时长提前停止。

### 3. 通用纹理门错误惩罚了真实倒地视角

Office90 与 Office102 的 initial、decision 和安全 Backstep outcome 均为非退化 RTX 帧。
失败只来自 Continue 已确认跌倒后的 body-fixed camera outcome：相机自然朝向近距离地面或墙面，
其亮度标准差分别约 0.0148、0.0172，低于通用 0.025 门槛；但 black fraction 为 0，仍有
43/51 个 5-bit 颜色。该帧可以作为动作后果证据，却不是归因模型的 decision snapshot。
后续视觉门必须按角色拆分：initial/decision 与安全恢复 outcome 使用严格场景信息门；已跌倒
outcome 使用传感器有效性、曝光和物理状态一致性门，不能要求它保留正常导航视角的纹理丰富度。

Isaac 日志中的 EmbodiedGen mesh UV primvar warning 需要继续作为资产编译债务追踪，但本批两例
低纹理失败不能归因给 UV：其 decision snapshot 指标正常，失败严格发生在倒地 outcome。

## v32 开发合同

v32 只允许在 v31 之外的新 development scene 上开发；v31 结果永久冻结。

1. decision 改为首次 attachment 后 1 step 的 action-agnostic 确定性分支，不再等待姿态已经
   进入紧急域。Continue 与 Backstep 仍必须在 branch 前 byte-identical。
2. Backstep 在剩余 horizon 内持续执行，直到后退距离达到 0.20 m 且 `active_feet=0` 连续
   5 steps；只有真实 clearance 完成后才允许 hold。
3. 保留 command-agnostic contact adhesion、force cap、peel、cooldown、route controller 和
   Continue 速度，不通过读取未来 action 或命令直接释放 adhesion。
4. 图像 QA 拆成 inference-state strict gate 与 consequence-state gate；不删除任何原始帧和指标。
5. 新开发场景从 candidate 113 以后按 nominal-only scene/RTX/Go2/stack gate 获取；先冻结场景，
   再运行 O4。v30 未处理的 106–112 suffix 继续保持 untouched。
6. 开发通过后还要再生成一批更晚、从未运行 O4 的 operator-blind cohort 做一次性确认；只有该
   确认通过，才恢复 O5–O11 realistic coverage 与 canonical corpus 扩量。

以上工作仍只是 operator capability gate。它不能将 realistic A0–A7 从 `0/8` 改为已完成；
正式目标仍是 registry-bound corpus、重新训练/推理以及在新版数据上独立复现 A0–A7。
