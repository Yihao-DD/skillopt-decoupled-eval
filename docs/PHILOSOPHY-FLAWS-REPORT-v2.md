# SkillOpt 哲学缺陷 v2 — 深层缺陷 + 范式级改进 + 可抢的 open gap

*2026-06-18 · 分支 `acceleration` · **承接 v1 `PHILOSOPHY-FLAWS-REPORT.md`(3 缺陷:贪心搜索/盲目信用/噪声点估计选择),深化一轮。** 深度研究(deep-research workflow,108 agents / 26 源 / 25 claim 三票对抗验证,**24 confirmed / 1 killed**)融合我方 4-benchmark×5-seed 实证。可冷启动:外部 agent 读本文即可接手哲学线。*

---

## 0. 一句话结论(可发表)

> **「在含噪 proxy 反馈下迭代优化单个 NL 工件」本质是 EVALUATION-BOUND(评测受限)+ SELECTION-LIMITED(选择受限)的——而文献证明这是结构性的、不是调参 bug。** 我方「瓶颈在选择不在生成」**从猜想升级为被证明的定理(Optimizer's Curse)**。三个范式级补救都存在文献根基,但**没有一个被证明在「单 NL 工件」设定下、等预算、held-out、显著地打赢调好的 greedy** —— 这是项目能「第一个填」的空白。

---

## 1. 排序的深层缺陷(文献 + 我方数据双重落地)

### ① 最强:FLAW (E) = Optimizer's Curse / 选择性过拟合 = **被证明的定理**
- **Smith & Winkler 2006**(*Management Science* 52(3):311-322,《The Optimizer's Curse》):选 noisy 估计的 argmax → 选中者真实值在**期望上低于其估计,即便每个估计无偏**(Prop 1:E[μ_{i*}−V_{i*}]≤0,只要可能选到非最优就严格<0)。**元凶是选择算子本身,非估计偏差。** worked example:3 选项真值 0、独立 N(0,1) 估计 → 期望失望 0.85σ。
- **Cawley & Talbot 2010**(*JMLR* 11:2079-2107,~2100 引):**模型/候选选择本身会过拟合**;"选择准则的**低方差与无偏一样重要**";其量级"**常与算法间真实性能差相当**";适用于"**任何**在有限样本上优化选择准则"——**显式授权迁移到 NL-skill 选择**。
- **我方落地**:「val 最高却 test 最差」+ OfficeQA ship 比 baseline 差 17pt 的不可察觉退化 = 这个定理的**预期结果**,非意外。
- **🔑 关键 nuance(我方已部分缓解)**:SkillOpt 在**同一 D_sel** 上评所有 candidate → 噪声**正相关** → **缩小** curse 量级(Smith-Winkler Table 2)。**这正是我方 pairgate(McNemar 配对门)在做的** → 方向对,但还差更深一步(§2 方向③)。
- 源:`pubsonline.informs.org/doi/10.1287/mnsc.1050.0451` · `jmlr.org/papers/v11/cawley10a.html`

### ② FLAW (A)+(B) = evaluation-bound(评测成本是绑定约束 / bitter-lesson 反转)
- **Miller/Anthropic 2024**(arXiv **2411.00640**):eval 分是点估计,SE=√(Var/n);分辨 0.03 效应需 **~969(≈1000)个独立 item**("new evals should contain at least 1,000 questions");我方 60-item val 的 binomial SEM ≈ **±6.5pt**。**→ 3pt 单步增益结构性低于可检测地板,贪心门基本在选采样噪声。**

### ③ FLAW (A) 机制 = 有限「最大有用候选池」;搜索过头反伤
- **Dalal et al. 2026**(arXiv **2603.15377**,《More Test-Time Compute Can Hurt》):argmax over n 个 noisy 分注入 winner's-curse 偏差 ≈σ√(2log(n−1))(EVT);**最大有用池 n̂=1+exp(Δ²/2σ²)**,由 SNR=Δ/σ 决定。高噪打分器 k=4 **降** held-out 5.4pt(Llama)/3.9pt(Mistral);低噪打分器同宽度 **+6.2/+8.9**。**→ 打分器/评测可靠性(非生成)决定能爬多高;噪>增益时多搜索主动有害。**(beam-search 迁移,非直接 SkillOpt 测量。)

### ④ FLAW (B) = 无可靠进度信号 = 没指南针的优化
- **Atil et al. 2024**(arXiv **2408.04667**,Eval4NLP'25):**temp-0 根本不确定**——同配置准确率摆动达 15%、最好-最差差 **70pt**;单题分两次采样间摆 **44pt**;"单 run 比较不可靠,建议报 max-min"。我方 ~19pt 同-skill 摆动与之一致。
- **Madaan et al. 2024**(arXiv **2406.10229**,COLM/ICLR,FAIR/Stanford):**进度可检测性由 metric 决定**——离散 exact-match MMLU 训练单调性 **0.09**(SNR 52)vs 连续 cloze **0.95**(SNR 303),**同任务、仅打分方式不同**。**→ 在离散含噪 proxy 下进度信号可「根本不存在」** → OfficeQA 退化不可察是预期。

### 旁证(medium):「生成够好、败在选择/评测」
- **Zhang et al. 2026**(arXiv **2604.14585**,《Prompt Optimization Is a Coin Flip》,CTB@ICML'26):Haiku 4.5 上 72 run(6 法×4 任务×3 重复),**对 zero-shot 增益与抛硬币不可区分**(binomial p=0.91)、**49% 低于 zero-shot**、3/4 任务平均增益为负;**只在有可利用输出结构时才有用**。**→ 完美对应 Method C 只在 LiveMath 有用、SSB 中性/负。**(模型特定 Haiku;Nova Lite 更差;迁移类比非同款。)

(**被杀 1 条**:字符串级可复现性 TARr@10 50%/7%,对抗验证 1-2,已剔除——仅准确率级非确定性存活。)

---

## 2. 真·范式改进方向(非微调)+ 我们能抢的 open gap

**三个范式级补救,均无人证明在「单 NL 工件」等预算 held-out 显著打赢调好的 greedy ⇒ 项目第一个填。**

| # | 方向(范式级) | 文献根基 | 严格门状态 |
|---|---|---|---|
| 1 | **评测中心重构**:连续/cloze metric 替离散 exact-match(0.09→0.95)、配对/聚类 SEM(McNemar)、功率分析、~1000-item 评测 → **先拉 SNR 再优化** | Miller / Madaan / Atil | 机制立,NL-skill 等预算胜未证 |
| 2 | **收缩/「有纪律的怀疑」gate**:Bayesian/经验贝叶斯把 candidate 估计**向先验收缩后再选**(Optimizer's Curse 原版解药) | Smith & Winkler 2006 | 同上;需可标定先验/噪声模型 |
| **3 ⭐** | **inference-aware / 显著性感知选择**:不按"估计均值最高"接受,而按 **"该增益在新 held-out 上通过诚实显著性检验的概率"最高** 接受 —— **为「能复现」优化,不为「点估计最高」** | **Bastani et al. 2025**(arXiv **2510.18161**,《Beating the Winner's Curse via Inference-Aware Policy Optimization》) | 仿真-only(contextual bandit,IPW);**LLM/NL-skill 迁移未证 = 我们的空白** |

**THE OPEN GAP（项目最强定位,文献已确认缺失,= v1 §2 元发现的延续）**:
> 没有任何 selection-aware / shrinkage / inference-aware accept-gate 被证明在**单 NL 工件迭代-edit + val-gate** 设定下、**等评测预算**、**held-out test**、**显著**打赢**调好的** greedy。Bastani 是仿真 bandit、Smith-Winkler 是决策论、Zhang 是整-prompt 替换——**都不是这个范式。**

---

## 3. 推荐下一步(实打实、接在已建之上)

**主方向 = 方向③ inference-aware 选择门 + 方向① 评测功率前沿** —— 范式级、非 trick、且自然接在 pairgate 上:

1. **把 accept-gate 从「argmax 均值」升级为「argmax P(该增益过 held-out 显著性检验)」**(Bastani 2025 在 LLM-skill 的首次实例化)。我方已有 McNemar/配对统计基建(`qd/pairgate.py`)→ 自然下一步,且是 **open gap = 可发表的「第一个」**。
2. **预算分配前沿**(深 open question):固定评测预算,钱花在**"每候选更多 item 降 SEM"** vs **"更多候选"**?Dalal n̂=1+exp(Δ²/2σ²) 给了 beam 闭式答案——**NL-skill 有无类似前沿、且单步增益<噪声地板时降-σ 是否胜过多-candidate?** 我方数据(噪>增益)正好能答。

**为何「哲学级」而非取巧**:把优化目标**从「点估计最优」整个换成「可复现/经得起检验」**——重新定义「什么叫一次成功的 edit」,直击 Optimizer's Curse 根因。

**两个更深但证据空白、动手前需各自再研究的**:Flaw C(模块化/检索表示替单 NL-blob——深研 **0 条存活 claim**)、Flaw D(分布鲁棒/per-cluster/CVaR 目标替均值——**0 条存活 claim**)。真未探区。

---

## 4. 开放问题(研究方向,见 deep-research openQuestions)
1. **等预算 head-to-head(中心 gap)**:任何 selection-aware 门等预算 held-out 显著打赢 tuned greedy?(项目最强 first-to-fill)
2. **预算分配前沿**:exploration vs evaluation-power 的最优分;NL-skill 上有无 Dalal-类闭式/经验前沿?
3. **连续进度信号**:能否为 SkillOpt benchmark 造连续/软 surrogate(gold log-prob / judge 分布 / 部分分),把 0.09→0.95 那种单调性带来,且等预算迁移到更高离散 held-out?(闭 Flaw B)
4. **表示变更(Flaw C)**:模块化/检索 competence store 替单 blob,能否可证降 drift/干扰 + 等预算 held-out 胜?(需先补研究)
5. **分布鲁棒目标(Flaw D)**:per-cluster/CVaR/worst-group 替 mean-on-D_sel,能否更保 held-out 增益、防 OfficeQA 式不可察退化?
6. **abstention/非承诺范式**:ship running-best 是含噪下不可逆承诺(curse 触发点);ensemble-of-skills / 推迟选择到功率够 / CI 跨 0 就 abstain,等预算 held-out 是否胜?

---

## 5. 诚实 caveat(别 oversell)
- 最强三条(eval-noise、optimizer's-curse、shrinkage)是**领域通用统计定理**(经典 ML/决策论),证**原理**非 SkillOpt 直接测量;我方数据补 LLM 域实例。
- **transfer-not-direct**:Dalal(beam-search)、Zhang(整-prompt 替换)是**佐证类比**,非单-工件-val-gate 范式。
- **预印本**:Atil/Zhang/Dalal/Bastani 多为 arXiv(部分 workshop-accepted);现象被独立佐证但未全过传统同行评审。
- **worst-case vs typical**:70pt/44pt/12%-flip 是最坏 cell(诚实标 "up to");969-item/±6pt 是量级锚非硬阈(绑定方差假设)。
- **model-specificity**:coin-flip 是 Haiku 特定;beam 退化是 scorer 特定。「换模型/打分器一切变」本身是发现的一部分。
- **remedy 效力未证**:三方向严格门(等预算 held-out 显著胜、单 NL 工件)在文献**全未达** = 这正是项目空白,但也意味无保证胜、是有强理论背书的假设。
- **正相关噪声上行(被低估)**:同 D_sel 评所有 candidate → 噪声正相关 → **缩小** curse + 正是配对/聚类 SEM(McNemar)恢复可分辨的 regime → 我方 pairgate 已部分对路,方向①应前置之。

## 6. 引用清单(全 primary)
Smith & Winkler 2006 *Mgmt Sci* [mnsc.1050.0451] · Cawley & Talbot 2010 *JMLR* [v11/cawley10a] · Miller 2024 [2411.00640] · Dalal et al. 2026 [2603.15377] · Atil et al. 2024 [2408.04667] · Madaan et al. 2024 [2406.10229] · Zhang et al. 2026 [2604.14585] · Bastani et al. 2025 [2510.18161] · SIREN [2605.05973] · Bayesian Hybrid Shrinkage [2511.06318]。表示/分布角度(Flaw C/D,无存活 claim,待补研究):AWM [2409.07429] / Dynamic-Cheatsheet [2504.07952] / Voyager [2305.16291] / ExpeL [2308.10144]。

*全量验证日志见 session transcript 的 deep-research 输出(task wtjo182z7;25 claim 逐条三票记录)。*
