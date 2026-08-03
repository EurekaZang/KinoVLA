# 室内场景 + adhesion 异常算子：ICRA Demo 审查说明

日期：2026-07-20  
场景版本：`indoor_workstudio_adhesion_v1`  
配置：`configs/demo/indoor_adhesion_icra.yaml`  
成品目录：`outputs/demos/indoor_adhesion_icra/`

## 1. 结论与使用边界

本垂直切片已经达到“可供导师审查的 ICRA demo 工程门”：真实 Isaac Sim/PhysX 执行、Go2 学习策略闭环、固定在机身前部的 RTX RGB 相机、足端局部 adhesion 力、同 seed 配对对照、第三人称审查视频、逐步遥测和可复现清单已经连通。

它还不是可直接写入论文主表的统计证据。当前只有一个场景和一个 seed，视觉层是按照 EmbodiedGen 接口合同编写的 Kino 自建 fallback，并非 EmbodiedGen 生成结果；没有真实 Go2 锚点，也没有完成 A0–A7 realistic replication。正确表述是“首个新框架物理/视觉垂直切片通过”，不是“新数据集已经完成”。本工作不引入或恢复 A8。

## 2. 场景成品

场景是 10 m × 7 m 的室内工作室/生活空间，包含：

- 木地板、墙体、踢脚线、窗户、玻璃、顶灯和吸音构件；
- 开放书架、书、储物箱、工作台、工具柜、长凳、地毯、沙发和植物；
- 保持 Go2 中央路线无刚体障碍的同时，在路线两侧提供生活/生产语义；
- 66 种 USD PreviewSurface 材质、218 个环境 primitive、30 个碰撞 primitive；
- 1.73465 m² 的不规则 adhesion footprint；视觉上使用哑光浅灰色粘附式地面保护膜，不用黄色诊断贴片；
- 已关闭策略速度箭头，并消除了默认 Isaac 蓝色网格从木板接缝泄漏的问题。

场景由五层 USD 组成：

1. `visual.usda`：房间、家具、灯光和保护膜外观；
2. `collision.usda`：墙体、家具等碰撞；
3. `physics.usda`：地面材料与视觉—碰撞分离声明；
4. `operators.usda`：O4/adhesion 类型、fidelity 和精确 footprint；
5. `episode.usda`：可替换组合入口。

静态审计结果：无重复 prim、无缺失材质、中央走廊无碰撞阻挡、异常区域与路线相交，全部通过。各层 SHA-256 在 `manifest.json` 中冻结。

## 3. adhesion 物理

本实现不是在机身上施加匿名阻力，而是对 `FL_foot`、`FR_foot`、`RL_foot`、`RR_foot` 分别维护状态机：

- 足端进入不规则区域、足高在表面容差内且法向接触力 ≥ 5 N 时，建立该足的世界系 anchor；
- 每个控制步用真实足端位姿/速度计算三轴弹簧—阻尼力，并通过 PhysX 对对应 foot body 施力；
- 同时最多激活两足，单足施力封顶 55 N，前向破坏阈值 105 N；
- 明确的反向控制阶段才允许按 16 N 低阈值 peel；不能用支撑脚的瞬时反向速度判断，否则正常前进步态会被误判；
- peel 或 break 后本 episode 不自动重粘，事件、anchor、接触力、伸长量和施力均写入遥测。

当前 fidelity 名称为 `foot_local_force_constraint_v1`。它已经满足“力作用于具体足端、可审计、可产生局部姿态后果”的 demo 要求，但不能冒充胶层 FEM、材料撕裂或原生粘着接触模型。

## 4. 配对协议

nominal 与 adhesion 使用相同 seed=42、相同初态和相同控制序列；唯一干预变量是 adhesion 算子：

- 0–4.7 s：`vx = +0.55 m/s`；
- 4.7–7.7 s：`vx = -0.32 m/s`，作为明确 peel 请求；
- 7.7–8.7 s：停止；
- nominal 无算子并完成全过程；adhesion 在跌倒后按真实终止条件提前停止。

前置视角与第三人称视角是同 seed 的独立 replay。原因是本机 Isaac Sim 5.1 在同一环境同时初始化两个 Replicator Camera 不稳定；两次 replay 的摘要和逐步遥测完全一致，视频帧数也一致。

## 5. 关键结果

| 指标 | nominal | adhesion + peel |
|---|---:|---:|
| seed | 42 | 42 |
| 仿真时长 | 8.72 s | 5.26 s（跌倒终止） |
| 峰值 adhesion 力 | 0 N | 51.56 N |
| 最大同时粘附足数 | 0 | 2 |
| attach / peel / break | 0 / 0 / 0 | 2 / 2 / 0 |
| 最大机身倾角 | 0.0568 rad | 0.8452 rad |
| 是否越过 0.8 rad 跌倒阈值 | 否 | 是，5.26 s |

事件链为：

- 4.18 s：`FR_foot` attach；
- 4.42 s：`FL_foot` attach；
- 4.82 s：总足端 adhesion 力达到 51.56 N；
- 4.84 s：`FL_foot` peel；
- 5.20 s：`FR_foot` peel；
- 5.26 s：累积姿态扰动使倾角达到 0.8452 rad，超过 0.8 rad 阈值。

nominal 最终净位移只有 1.62 m 是因为配对协议包含 3 s 反向运动，不代表 nominal 无法前进。

## 6. 建议审查顺序

1. 先看 `nominal/review_contact_sheet.jpg` 和 `adhesion_peel/review_contact_sheet.jpg`，检查房间、Go2 路线、保护膜外观和姿态差异；
2. 再看两个 `review_third_person.mp4`，确认 attach 后的局部腿部受力和跌倒不是后处理动画；
3. 看两个 `front_body_fixed.mp4`，确认相机严格随 Go2 六自由度运动，没有自动瞄准；
4. 看 `adhesion_peel/telemetry_front.png`，核对反向命令、足端力与倾角越阈值的时间关系；
5. 最后看 `manifest.json`、`review_summary.json` 和 `scene_usd/episode.usda`，审计参数、哈希和 USD 分层。

## 7. 成品索引

- 第三人称对照：`nominal/review_third_person.mp4`
- 第三人称异常：`adhesion_peel/review_third_person.mp4`
- Go2 前置相机对照：`nominal/front_body_fixed.mp4`
- Go2 前置相机异常：`adhesion_peel/front_body_fixed.mp4`
- 对照审查图：`nominal/review_contact_sheet.jpg`
- 异常审查图：`adhesion_peel/review_contact_sheet.jpg`
- 异常诊断曲线：`adhesion_peel/telemetry_front.png`
- 峰值受力状态八机位截图：`multiview_peak/contact_sheet.jpg` 与 `multiview_peak/01_*.png`–`08_*.png`
- 多机位冻结状态清单：`multiview_peak/multiview_manifest.json`
- 逐步足端遥测：`adhesion_peel/telemetry_front.jsonl`
- 可复现与来源清单：`manifest.json`、`manifest.yaml`
- USD 组合入口：`scene_usd/episode.usda`

四个视频均为 H.264/20 fps：前置相机 640×360，第三人称 960×540；nominal 为 175 帧，adhesion 为 106 帧。`manifest.json` 中给出了 front/review 两条完整复现命令。

## 8. 请重点拍板的审查项

- 视觉叙事：当前“室内工作室 + 浅灰粘附式地面保护膜”是否足够自然，还是需要换成厨房溢胶、仓储胶膜或维修区树脂泄漏；
- 异常强度：当前 consequence 是“反向 peel 已发生，但累积扰动仍导致跌倒”，是否符合目标，还是要降到“明显失速但不跌倒”；
- 外观真实度：是否接受 USD-native demo 作为框架验收，随后再替换为 EmbodiedGen/扫描资产视觉层；
- 相机：当前是严格 body-fixed RTX RGB，但没有可信 depth；是否将 depth 修复列为进入 pilot 前的硬门；
- 论文边界：是否同意只把此成品作为 pipeline figure/demo，不把单 seed 数值并入 A0–A7 主结论。

## 9. 进入 A0–A7 realistic replication 前的硬门

- 接入并冻结真实 EmbodiedGen V2 输出或有许可证的扫描/高质量资产，替换当前 authored visual layer；
- 增加材质纹理、法线、粗糙度和照明域随机化，并进行视觉泄漏审计；
- 完成前置相机内外参标定、畸变与可靠 depth；
- 用多场景、多 seed、严重度梯度和 bootstrap CI 验证 adhesion 后果；
- 用真实 Go2 的低风险 adhesion fixture 校准力—位移范围；
- 通过以上门后再按总计划把 realistic subset 加入 A0–A7，而不是先写论文结论。
