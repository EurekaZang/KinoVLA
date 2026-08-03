# Kino-Fail O4 v34 held-out confirmation analysis

## 结论

v34 是一次完整、不可重跑的负确认：artifact/causal integrity 为 **6/6**，预注册严格结果为
**5/6**，因此 `counts_as_realistic_o4_operator_confirmation=false`，realistic A0–A7 readiness
仍为 **0/8**。失败不能通过修改审计合同、删除场景或重跑 seed 消除。

## 冻结边界

- acquisition config SHA-256：`0f8bf7daef1accde52c94c56137c855015674324f4b25af2b1e712acaeb2ef78`
- acquisition postrun SHA-256：`2501393ebc192caa444d89a5f78bcb62342b11d06298ef491ced455b60f7e108`
- action config SHA-256：`64cc72a9957dbae80a64154f236fea7296de97dd83b76effdfb4864ce94cfd35`
- action preflight SHA-256：`44567d13a1f5e4f2a7cb0bce895e1017454367dda3d6aa44c6d3f3e4e99af8d7`
- runner audit SHA-256：`9bfbb0e74a6574e26c88f7dbeea0c4ed029fc31aa5dbeb1d032ab51295dfc4b2`
- action postrun SHA-256：`ad772be27917ceadac9cffa86f782740ee29159b0aa7a33119b20329a58bfae0`

acquisition 按冻结顺序处理 candidate 137–157 共 21 条，首个满足 quota 的前缀入选 6 条，
覆盖 LivingRoom、Kitchen、House、Office 四个 family；candidate 158–160 未生成。场景准入仅看
source geometry、route/corridor、静态与运动 RTX、Go2 nominal 和 stack admission，O4 outcome
在 cohort 冻结前不可见。

动作执行严格继承 v32 的完整 runtime 与 state-aware visual contract。v34 唯一新增的是在执行前
已经冻结并测试的 phase-aware nuisance gate：matched prefix 与 Backstep 全时域必须在 0.3 m
route budget 内；Continue 只要求截至由 tilt≥0.08 rad 且 height-drop≥0.02 m 定义的物理失稳
起点受限，失稳后的偏移继续报告但不作为混杂门。失稳定义不使用 route deviation。

## 六场结果

| scene | family | strict | Backstep retreat (m) | peel events | Continue fall step | failure onset step |
|---|---|---:|---:|---:|---:|---:|
| indoor_livingroom_139 | LivingRoom | PASS | 0.2747 | 1 | 178 | 143 |
| indoor_kitchen_140 | Kitchen | PASS | 0.2733 | 1 | 179 | 144 |
| indoor_livingroom_151 | LivingRoom | PASS | 0.2357 | 2 | 149 | 111 |
| indoor_house_153 | House | PASS | 0.2634 | 3 | 124 | 85 |
| indoor_office_156 | Office | **FAIL** | 0.0000 | 0 | 128 | 83 |
| indoor_livingroom_157 | LivingRoom | PASS | 0.2403 | 2 | 157 | 126 |

五个正例均满足 matched-state 分支、Continue 分支后跌倒、Backstep 完整 horizon 无跌倒、真实
contact-driven peel、至少 0.15 m 后撤、全 Backstep route containment 和 state-aware RGB 门。
LivingRoom 157 的原始通用 `all_frames_non_degenerate=false` 来自倒地后的 Continue outcome；
预注册 v34 state-aware gate 保留 initial/decision/Backstep outcome 的严格视觉门，并只对已经跌倒
的 Continue outcome 使用传感器有效性门，因此该例被正确判为通过，而非事后改判。

## Office 156 为什么是真失败

Office 156 的 Continue lane 正常产生目标后果：第 80 步 FL 首次粘附，第 81 步冻结 decision，
第 82 步动作分支，第 83 步达到物理失稳起点，第 128 步跌倒。matched prefix 最大横向偏差
0.0128 m，Continue 到失稳起点最大偏差 0.0128 m，因此没有路线混杂。

Backstep lane 与 Continue 在 decision 前逐字节一致，但固定 `-0.24 m/s` 命令未克服该步态相位
的反转延迟：

- decision progress：0.3702 m；最终 progress：0.5891 m；分支后继续前侵 0.2188 m；
- backward recovery distance：0.0000 m；
- FL 在第 80 步粘附后始终没有 peel；FR 在第 113 步再次粘附；
- 两足同时承受粘附后第 115 步跌倒；最大 tilt 0.8218 rad；
- Backstep 全轨迹最大横向偏差仅 0.0621 m，所以不是路线越界导致；
- Backstep outcome 的低纹理是跌倒后的结果，不是主失败原因。

因此 v34 识别的是动作架构对 gait/contact phase 泛化不足，而不是 O4 接触定律、相机、场景 QA
或审计阈值错误。

## v35 候选与后续门

v35 只作为新的 development candidate，不对 v34 任何场景重跑。它保持 adhesion v3 接触物理、
一拍分支、基础 `-0.24 m/s` Backstep、相机与 300-step horizon 不变；当测得物理紧急态
（沿用 tilt/height-drop 阈值）或 decision 后继续前侵达到 0.025 m 时，锁存为 `-0.48 m/s`
紧急后撤。该策略不读取 scene identity，也不能命令粘连释放；peel 仍必须由真实卸载和抬足产生。

实现与纯逻辑测试：

- `scripts/isaac_collect_o4_action_consequence_v35.py`
- `tests/test_o4_action_consequence_v35.py`

下一步必须使用全新的 operator-blind development cohort 验证 v35，再冻结另一批从未运行过 O4
的 held-out cohort。只有新 held-out 全通过后，才能进入 O5–O11 realistic coverage；随后仍需
registry-bound corpus、新训练/推理和 A0–A7 全量复现。v34 的 5/6 不是 A0–A7 数据。
