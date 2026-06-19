# Drift-Guarded Localized Reflect — 设计 SPEC（局部信用分配，杠杆②）

*2026-06-18 · 分支 `acceleration` · **本文单独可读，可冷启动。** 设计哲学杠杆②（局部信用分配 /
localized credit assignment），生成侧、与 Method C 正交。背景链：`PHILOSOPHY-FLAWS-REPORT.md` §2/§7（杠杆②
的研究依据 + 严格门 caveat）→ `DECOUPLED-SELECTION-COLD-START.md`（C 主线，本杠杆与其正交可叠）。*

---

## 0. 一句话

> SkillOpt 的 reflect 盲目整块改、只看聚合失败、不查"改完是否打坏了原本做对的样本"（StraGo 命名的
> **prompt drifting**）。我们加一层**生成侧 drift-guard**：用 beam/gate **本就要算**的逐-item D_sel 信号度量
> 每个 candidate 的**回归率 Acr**，把标量 accept 门升级为 **drift-aware**（净正但高 Acr 也拒/降权），
> 从而把更干净的轨迹交给 Method C 去选 ship。**等预算起步**，效果不好可升级到最大档（额外评测做真 per-edit
> 裁剪）。

**与主命题不冲突**：我们修的是生成**缺陷**（regression / drift），不是"更聪明地生成"。它**与 C 正交**——dg
在训练期用 **D_sel** 清洁**轨迹**，C 在事后用 **held-out val60** 选 **ship**：不同时机、不同数据、不同对象。

---

## 1. 研究依据（来自 `PHILOSOPHY-FLAWS-REPORT.md` §2/§3②）

- **缺陷（可引用，high 3-0）**：EMNLP'25 survey——self-reflection 会 "incorrect error identification,
  prior biases, semantic invalidity"；**StraGo**（EMNLP'24, 2410.08601）命名 **prompt drifting**：对聚合失败
  整块改会**打坏原本做对的样本**，用 **Acr/Bcr** 度量翻车率，解法=Analyzer 把对/错分层。
- **SkillOpt 正中靶心**：reflect 聚合失败、改整块 skill、只看失败不看成功——三毛病全占（见 §3 现状核实）。
- **替代技法**：ProTeGi（2305.03495）方向性 textual gradient（先总结缺陷再"往反方向"改）；TextGrad
  （**Nature 2025**, 2406.07496）逐组件反传；Text2Grad（2505.22338）span 级。
- **严格门 caveat（诚实）**：强证据多在**多组件 pipeline**；"局部信用在**单块 prompt** 上等预算赢"是
  **开放问题**——又一个我们能填的空。Text2Grad 改权重（口径不符）、REVOLVE 仍整块改。
- **用户决定（2026-06-18）**：C 主线先行；②作为正交生成侧清洗并行设计/实现。**效果实在不好，等预算可扩到
  最大档（③）。**

---

## 2. 现有机制核实（读码已证，2026-06-18）

| 事实 | 位置 | 含义 |
|---|---|---|
| **对/错分层已半存在** | `gradient/reflect.py:run_minibatch_reflect` 分 `run_error_analyst_minibatch`（hard=0）/ `run_success_analyst_minibatch`（hard=1），各自独立产 patch | StraGo 的"分层"算子已有；缺**对照**（成功用来约束失败的改动）与**drift 度量** |
| **edit 已是锚定算子** | `prompts/analyst_error.md` schema + `optimizer/skill.py:_apply_edit_with_report` | op ∈ {`append`,`insert_after`(target),`replace`(target),`delete`(target)}；replace/insert/delete 锚到 exact text → **段定位算子层已半成品** |
| **归因是整 minibatch common pattern** | `analyst_error.md`："identify the most prevalent... COMMON failure patterns" | 粗粒度；非 per-failure-cluster；改动可落 skill 任意处 |
| **完全无 drift-guard** | `optimizer/skill.py` 盲 `replace/delete`；`append` 最常用 | 无任何"改完别打坏对的"检查；append 滚大 blob |
| **slow_update 保护区单独管** | `optimizer/skill.py:SLOW_UPDATE_*`；analyst 被禁改该区 | 就是污染 shipped best_skill 的那块；dg 不碰它 |
| **gate 是标量净分门** | `gate.py`（accept ⟺ cand_mean > cur_mean）；strict 下 `current==best` 恒成立 | dg 把它升级为 drift-aware；λ=1=逐字节等于它 |
| **beam 已逐-item 评 candidate** | `_propose_rg` 等 beam 模式 + scored 后处理（trainer ~2363） | **L3 的免费信号源**：candidate vs incumbent 逐-item D_sel 现成 |

---

## 3. 方法 — 两条正交轴（L3 门 + L1/L2 生成）

**架构精化（读码后，2026-06-18，优于初版"全塞进 beam_proposal=dg"）**：L3 与 L1/L2 在**不同轴**上，
独立可组合：
- **L3 drift-guard = 新 gate 模式 `gate_select="drift"`**（驱动 `--gate drift --dg-lambda/--dg-tau`），
  **复用 pairgate 已铺好的 `sel_per_item` 逐-item 管线**（trainer 2489 paired 分支旁加平行 `elif`）；
  `λ=1,τ=1` 在 hard 0/1 判分上 = strict 的同一 accept 动作。**已实现 + 测试 + 接线**（见 §8）。
- **L1+L2（对照证据 + 段定位）= 生成 / prompt 轴**（`beam_proposal=dg` 或 analyst prompt），独立于门。

复刻 `rg`/`pairgate` 的落地方式：**新模式、零触碰默认路径、可直接 A/B**。

| 层 | 机制 | 对应技法 | 成本 |
|---|---|---|---|
| **L1 对照证据** | error-analyst 同时收到「要修的失败簇」+「必须保住的成功轨迹」("以下现在是对的，你的 edit 不许伤它们") | StraGo Analyzer（对照升级） | prompt，免费 |
| **L2 段定位 edit** | 每条 edit 必带 `section` 标签 + `direction`；优先 replace/insert_after 命中相关段；抑制盲目 append | ProTeGi 方向性 + 段定位 | schema/prompt，免费 |
| **L3 drift-guard（新核心）** | 免费逐-item 信号算 `b`（回归）/`c`（修复）→ **drift-aware accept 谓词** 决定下一 incumbent | StraGo 的 Acr/drift 度量 | 零额外 eval |

**为什么 L3 是招牌**：现有标量门只看 net D_sel（几分，常在噪声地板下），**分不清 "+0净" 和 "+3修−3坏"**；
L3 把被 net 掩盖的"坏了3个"翻出来。即便 net 在 held-out 上 ns（支柱③预期），**pooled Acr 显著下降本身
就是干净、可发表的机制结论**。

### 3.1 一步数据流（★=新增）
```
rollout 当前 skill 于 train minibatch ──▶ failures + successes        (已有)
  ★L1 error-analyst 收到 failures(要修) + successes(要保住,对照)        (免费)
  ★L2 每条 edit 带 section/direction;优先 replace/insert_after          (免费)
apply ──▶ candidate(s)                                                (已有)
beam/gate 把 candidate 在 D_sel 上逐 item 评 0/1                       (已有=免费信号)
  ★L3 算 b/c → drift-aware accept 谓词决定下一 incumbent               (零额外 eval)
  ★(可选,仍等预算) 宽度换消融:部分 beam slot = base 去掉第 j 条 edit    (用宽度换 per-edit 定位)
所有 skill_vXXXX 照常存盘 ──▶ C 事后在 held-out val60 选 ship          (与 dg 正交)
```

### 3.2 drift-aware accept 谓词（核心公式）
D_sel 上 candidate vs incumbent 配对（hard 0/1，n 项），McNemar 不一致格：
- `b` = incumbent 对 & candidate 错 = **回归数**
- `c` = incumbent 错 & candidate 对 = **修复数**

| 谓词 | 公式 | 含义 |
|---|---|---|
| 现有 vanilla 门 | accept ⟺ `c > b` | drift-盲 |
| **dg 谓词** | accept ⟺ **`c − λ·b > 0`**（λ≥1）**且** `Acr = b / (incumbent对数) ≤ τ` | 回归比修复贵 λ 倍 + 灾难性 drift 硬上限 |

- **λ=1 且 τ=1 ⟹ 与 vanilla 门同一 accept 动作（hard 0/1 判分）**（忠实 no-op 默认；回归测试对 `pairgate.compare_paired().better` 断言之）。λ>1 = drift-averse。
- 防噪（可选）：叠 `pairgate.sign_test_one_sided`（已有）——仅当 c 对 b 不对称**过单边 sign 检验**才动手。
- **反 oracle + 防事后调参**：λ、τ **只在非 test 数据（train/D_cal）标定**，test 只报告（沿用 RR-Boost 的
  D_cal 纪律）；默认 λ=1=vanilla。
- **复用**：dg 谓词 = `qd/pairgate.py` 配对谓词 + 回归格非对称代价 λ，不重造。

**押的假设（可证伪，正是要测的）**：被 dg 拒的"net +1 但高 Acr"candidate，那 +1 多半 val-overfit 噪声、掩着
会在 test 现形的真 drift；跳过它让轨迹留更干净分支 → C 能选到更高 test 的 ship。

### 3.3 L2 段定位 edit schema（在 analyst_error 的 patch 上扩字段，向后兼容）
```json
{"op":"replace","target":"<exact text in the relevant section>","content":"...",
 "section":"<heading this edit addresses>","direction":"<add|remove|tighten 的一句方向>"}
```
- `section`/`direction` 缺省可空（兼容旧 prompt）；dg 模式下要求填、并据此记录定位统计。
- 抑制 append：dg prompt 要求"能 replace/insert_after 命中相关段就不要 append"；记录 append 占比作诊断。

---

## 4. 升级到最大档③（用户授权的 fallback）

**触发**：V3（满档②）在 LiveMath ≥3 seed 等预算下，既**无 pooled-Acr 显著下降** vs V0，又**无 C-on-test 增益**
→ 升级③。
**③机制**：对"原本做对的 protect-set"逐条 **leave-one-edit-out 重跑** → 定位惹祸 edit → **裁掉** → 重评裁后
candidate。额外 eval **计入预算**；为公平，相应减 candidate 数/步数对齐总预算；**显式 log 花的预算**（不静默截断）。
**③仍守**：反 oracle（只 D_sel/protect-set，绝不碰 test）、隔离模式、λ/τ 非 test 标定。

---

## 5. 实验 / 消融设计（诚实口径）

### 5.1 指标
- **主（机制，灵敏）**：**pooled Acr** = Σ回归 / Σincumbent对，跨 step 跨 seed 池化（单步噪、池化稳；这正是
  支柱③的体现，非 bug）。+ pooled fix 率。
- **次（奖品，等预算 head-to-head）**：**C-on-held-out-test**，dg-轨迹 vs vanilla-轨迹的 C-selected ship。
  多种子 + **seed-level sign test** + McNemar（我们的诚实门）。
- **诊断**：accept 率、轨迹长度、anchored vs append 占比、平均 edit blast radius。

### 5.2 消融梯（隔离每层，仿 RG 的 A−B）
| 档 | 配置 | 隔离 |
|---|---|---|
| V0 | vanilla greedy（= 现 SkillOpt k1） | baseline |
| V1 | +L1 对照证据 | StraGo 分层的增量 |
| V2 | +L1+L2 段定位 | 定位的增量 |
| **V3** | +L1+L2+L3 drift-aware accept = **满档②** | drift-guard 的增量 |

务实：先 **V0 vs V3** go/no-go；V3 赢再 decompose V1/V2 归因。

### 5.3 benchmark 顺序
**LiveMath 先**（最便宜、确定性判分=无 grader 噪声地板、C 已在此做完 → §3 结果表是干净对比锚）→ SSB / SearchQA
（headroom 已确认的 cell）→ OfficeQA 最后（24-turn 慢）。

---

## 6. 忠实 / 隔离 / 错误处理（贯穿）
- **λ=1,τ=1 默认 = vanilla 的同一 accept 动作（hard 0/1 判分）**（回归测试 fixture 断言相同 accept 决策）。
- **反 oracle**：drift-guard 只读 D_sel，绝不碰 test；λ/τ 只在 train/D_cal 标定。
- **隔离**：新增 `beam_proposal=dg`，白名单加 `"dg"`；默认路径零改动；`--beam-proposal dg` opt-in。
- **fallback 不崩**：仿 `_propose_rg`——too-few-failures / 空对照 / 解析失败 → 退回 `_propose_one`，只能持平或收窄，
  绝不崩步。
- **resume-safe / JSON-robust**：仿 reflect.py 的 4-retry 读 conversation.json。

---

## 7. 代码落点
| 文件 | 内容 | 测试 |
|---|---|---|
| `qd/drift_guard.py`（新，纯函数 zero-API） | `regression_fix_counts(cand_items, incumbent_items)`→(b,c)；`acr(b, n_incumbent_correct)`；`drift_aware_accept(b,c,*,lam,tau,...)`→bool；`format_contrastive_evidence(failures, successes)`；`parse_localized_edit(edit)`（section/direction 校验）；`leave_one_edit_out_subsets(edits)`（消融子集，③用） | `qd/tests/test_drift_guard.py` zero-API |
| `SkillOpt/.../engine/trainer.py` | **L3 done**：`gate_select="drift"` 分支（镜像 paired，调 `drift_gate_action`，复用 `sel_per_item`）+ `dg_lambda/dg_tau` 配置校验。**L1/L2 pending**：`_propose_dg(k)`（镜像 `_propose_rg`）+ `beam_proposal` 白名单加 `"dg"` | qd suite |
| `SkillOpt/.../prompts/analyst_error_dg.md`（或运行时 system 注入） | L1 对照 + L2 段定位 + direction 的 prompt | — |
| driver `repro/official/run_ssb_official.py` | **L3 done**：`--gate drift` choice + `--dg-lambda/--dg-tau` → `evaluation.{gate_select,dg_lambda,dg_tau}`。**L1/L2 pending**：`--beam-proposal dg` | --help 验过 |
| `tools/` | dg-vs-vanilla 的 pooled-Acr 报告 + C-on-test 对比（可复用 `decoupled_ship.py`/`mcnemar_pooled.py`） | — |

---

## 8. 实现顺序（TDD，先零-API 安全增量）
1. ✅ **`qd/drift_guard.py` 纯函数 + `qd/tests/test_drift_guard.py`**（22 tests green）：`regression_fix_counts`
   (b/c)、`acr`、`fix_rate`、`pool_drift_counts`、`drift_aware_accept`（λ=1/τ=1 no-op、λ>1 drift-averse、τ cap、
   可选 sign-test）、`drift_gate_action`（(action,diag) 包裹）。commit `710a295`。
2. ✅ **vanilla 等价守卫**：`drift_aware_accept(λ=1,τ=1)` == `pairgate.compare_paired().better`（已断言）。
3. ✅ **L3 trainer/driver 接线**：`gate_select="drift"` 分支（镜像 paired，复用 `sel_per_item`）+ `dg_lambda/dg_tau`
   校验 + driver `--gate drift --dg-lambda/--dg-tau`。trainer/driver compile + 全 qd 套件 401 green + `--help` 验过。
4. ⏭ **2-task 真路径 smoke**（box2，仿 preflight）——任何付费 run 之前。
5. ⏭ **首个实验 = L3-only（最外科）**：vanilla proposal + `--gate drift --dg-lambda 2.0`（可叠 `--dg-tau`）vs
   strict baseline，LiveMath ≥3 seed 等预算；看 **pooled-Acr↓ + C-on-test**。赢→加 L1/L2（V1→V3）做归因 + 扩 bench；
   不赢→按 §4 评估升级③。
6. ⏭ **L1/L2 生成轴**（`beam_proposal=dg`）：对照证据 + 段定位 prompt/schema + `_propose_dg`（仅在 L3-only 见效后）。

---

## 9. STOP（纪律）
- 不碰 test 选 / 不放宽反 oracle / λ-τ 不在 test 标定。
- v1 **不**上 proxy verifier（RR v1>v2 被烧过：会误杀）。
- 不重建 exploration（cold-start 已证 relax 探索是 NO-OP）；dg 改的是 accept **形状**不是放宽。
- 不碰 slow_update 保护区。
- 不静默截断预算（③ 的额外评测必须显式 log）。
