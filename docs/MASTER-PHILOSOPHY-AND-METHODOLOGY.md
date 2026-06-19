# SkillOpt 增益器 — 完整哲学与方法论(MASTER 文档)

*2026-06-19 · 分支 `acceleration` · **本文是全项目哲学 + 方法论的单一权威合集,可冷启动。** 力求详尽、不漏细节。
英文 submission 版定位稿见 §VI 对应的 （内部定位稿,未公开）;定位判决的对抗验证溯源见
（内部对抗验证记录,未公开）。配套单点文档:`DECOUPLED-SELECTION-COLD-START.md`
(Method C)、`INFERENCE-AWARE-SELECTION-RESULT.md`(siggate/lcb)、`DRIFT-GUARD-DESIGN.md`(drift-guard)、
`PHILOSOPHY-FLAWS-REPORT-v2.md`(理论根基)。*

---

## 文档地图(怎么读)
- **§I 背景与问题** —— SkillOpt 是什么、贪心 gate 的精确机制、噪声从哪来。
- **§II 核心命题与三支柱** —— 哲学骨架(选择问题 ≠ 生成问题)。
- **§III 我们如何发现"瓶颈在选择"** —— 发现的证据链(不是假设,是被实验逼出来的)。
- **§IV 三个增益器** —— 逐个的 philosophy + methodology + 公式 + 代码 + 结果 + 边界。
- **§V 三者如何拼成整体** —— 两条轴、暗线、时间线。
- **§VI 学术定位** —— (内部材料,未公开)。
- **§VII 结果汇总** —— 全 benchmark × 全方法的诚实表。
- **§VIII 诚实纪律** —— 贯穿全项目的方法论原则(反 oracle / 等预算 / seed-level sign test / never-lose)。
- **§IX 文献地图** —— 每篇引用论文与我们的关系。
- **§X 术语表 + 代码地图 + 复现**。

---

# §I 背景与问题

## I.1 SkillOpt 是什么(base 系统)
SkillOpt(microsoft)是一个 **test-time 自我改进**框架。被优化的对象是一份**自然语言「skill」**(一份解题攻略 /
playbook / 系统提示)。一个 **optimizer LLM** 迭代改写它,循环如下:

```
每一步(step):
  1. rollout:用当前 skill 在一个 train minibatch 上做题 → 得到成功/失败轨迹
  2. reflect:optimizer 看聚合失败 → 反思 → 产出对 skill 的一条/多条 edit(append/insert_after/replace/delete)
  3. apply:把 edit 应用到 skill → 候选 skill(candidate)
  4. gate:候选 skill 在一个固定的小选择集 D_sel 上打分 → 决定接受/拒绝
  5. 接受则候选成为新 incumbent;循环
训练结束 → ship 滚动 best_skill
```

关键概念:
- **skill** = 被迭代优化的单一自然语言工件(单 blob)。
- **optimizer / target**:optimizer 是写 edit 的模型;target 是执行 skill 做题的模型(常同一个、temp=0)。
- **D_sel**:gate 用的固定选择集(原版 n=40;我们的解耦实验里 gate 用 `val18`)。
- **slow_update**:skill 里一块被保护的区域,analyst 被禁止改它——它正是污染 shipped `best_skill` 的那块。
- **edit-budget / cosine schedule**:每步可保留的 edit 数,按 cosine 从 4→2 衰减(`optimizer/scheduler.py`)。

## I.2 贪心 gate 的精确机制(载荷性,读码已证)
- gate 谓词(`skillopt/evaluation/gate.py`):**accept ⟺ `cand_score > current_score`**(候选在 D_sel 上均值严格高于当前)。
- **strict 贪心下 `current_skill == best_skill` 恒成立**:gate 只接受严格优于"当前"的候选 → "当前"只升不降 →
  永远等于"至今最好"(应用在 `trainer.py:2526`)。**这条事实是后面一切的支点(见 §III.4、§IV.1)。**
- gate 是**标量净分门**:它只看两个均值之差,看不见这个差底下"修了几个 / 坏了几个"的构成(见 §IV.3)。

## I.3 噪声从哪来(为什么这把尺子是抖的)
- **D_sel 小**:几十道题,binomial SEM 大。val60 的 SEM ≈ **±6.5pt**(val18 更大)。
- **判分非确定**:即便 target temp=0 也非逐位确定;同一 skill 重测能摆 ~19pt(与 Atil 2024 的"temp-0 根本不确定"
  一致)。
- **后果**:单步 edit 的真实增益只有 **~2–4pt**,**结构性地落在噪声地板之下** → gate 基本在选采样噪声。

---

# §II 核心命题与三支柱(PHILOSOPHY)

## II.1 命题(一句话,可发表)
> Self-improving LLM skill-optimization **不是一个生成问题**(找更好的 edit),**而是一个选择问题**
> (在含噪、昂贵的评测下,从你已经找到的 edit 里,可靠地挑出该 ship 的那一个)。贪心 accept-gate 把
> **探索**与**选择**融成一次**含噪、不可逆的承诺**;我们**解耦**它们——自由探索,最后用高功率 held-out
> 的最佳臂识别可靠地承诺一次。

为什么是哲学不是 trick:**它重新定义了问题是什么。** 领域默认"问题=怎么生成更好的 edit",于是研究更聪明的
反思/textual-gradient/进化/MCTS。我们的数据**反转**了它:
- **生成已经够好** —— optimizer 确实造出了比初版好 +14pt 的 skill(oracle headroom 跨 4 benchmark × 2 模型存在,§III.5)。
- **选择浪费了它** —— 噪声点估计的门 ship 了一个被 slow-update 污染的 `best_skill`,**val 最高却 test 最差**(val 过拟合)。
- **生成 ≫ 选择。**

## II.2 支柱① 解耦 —— 选择是一种独立的认识论行为
贪心门把两个**性质完全不同**的动作焊死成一个:

| | 探索(produce) | 选择 / 承诺(commit) |
|---|---|---|
| 本性 | 生产信息 | 对信息下注 |
| 代价 | 廉价、可逆 | 后果昂贵、不可逆 |
| 应当 | 尽量广、不急着定论 | 攒够证据再做,而且只做一次 |

贪心门每一步**同时**做这两件事——把"这条 edit 是不是有用信息?"和"我要不要拿它当地基、断掉其它路?"混为一谈。
拆开后,博弈论里有现成的名字:**explore-then-commit / best-arm identification**。

## II.3 支柱② 含噪 + 不可逆 ⇒ 应当推迟承诺
这不是风格问题,是最优决策的结论(**optimal stopping / value of information**):当每次观测含噪、决定不可逆时,
最优策略是**推迟那个不可逆决定,直到再等的价值 < 等待的代价**。贪心门反着来——在信息最少、噪声最大的每一步,
做最不可逆的决定。我们把所有承诺**推到最后,用最大证据量(val60,benchmark 能给的最大复评集)一次做完**。

## II.4 支柱③ 真天花板是评测成本(最深、最诚实,bitter-lesson 反转)
**即便把解耦做到完美,能兑现多少仍被"那把尺子有多准"卡死。** 单步真实增益 ~2–4pt < val60 噪声地板 ±6.5pt →
哪怕选对方向,也只能**部分地、含噪地**兑现 headroom。→ test-time 自改进的真天花板 = **"你能多便宜可靠地评一个候选"**,
而非"你能多聪明地生成一个"。整个领域往"生成"堆算力的本能(bitter lesson)被**反转**:杠杆在评测/选择侧。

## II.5 暗线 —— 「聚合数字是个骗子」
贯穿三个增益器的一条线:**平均分/净分会撒谎。**
- ① 不信**每步**的点估计 → 推迟到最后用高功率 val 一次定;
- ② 不信**含噪 val 上的 argmax** → 只 ship 能复现的;
- ③ 不信**单步净分** → 翻开底下"修了几 / 坏了几"的构成。
三者都在拒绝被一个聚合数字糊弄——这是它们"同一套哲学的三个面"的根本原因。

---

# §III 我们如何发现"瓶颈在选择"(发现的证据链)

**这不是假设,是被实验逼出来的。** 两个相反方向的证据撞在一起,唯一自洽的解释只剩"瓶颈在选择"。

## III.1 起点:本想打败贪心(在生成侧)
最初思路和领域一致:贪心不够好,搜得更宽/更聪明/更深总能超过它。best-of-K 宽度(k4none)确实 **+5.4**
(3-seed,p=0.011)——**但那是"额外预算"赢的**(每步 4× sel-eval)。等预算下它 ≈ 贪心。真问题:**等预算能不能赢?**

## III.2 第一波证据:每种"搜得更好"都饱和或失败
| 尝试 | 想法 | 结果(源:`GAIN-SPRINT-FINAL.md` / Round-2 / selection-noise) |
|---|---|---|
| k4-structured(4 角色多样性) | 结构化角色代替随机采样 | **0.539,McNemar p=0.025 显著更差** |
| k8-to-top4(过采再筛) | 生成 8、廉价 probe 筛 top-4 | 0.568,ns 更差(8-item probe 太吵≈随机) |
| 宽度扫描 K2 vs K4 | 加宽度还能不能涨 | **K2≡K4,pooled 106:106 p=1.000** → 宽度 **K=2 即饱和**,K4 纯浪费 |
| K1-extra(深贪心,32 步等算力) | 把宽度预算改成搜更深 | **0.564 vs 贪心 0.554,p=0.80 ns**,仍 < k4none 0.607 → 赢的是**每步宽度**非算力;贪心早 plateau |
| QD-over-Skills / 蒸馏 / adaptive / 编辑级 EES | 各种"在空间里搜得更好" | 全部未破平台(0.567/0.607);train_ees seed1 直接被杀 |

→ **不管宽度、结构、深度、QD 多样性哪个角度,等预算下都搜不出比调好的贪心更高的东西。生成/搜索侧,饱和了。**
这本身是个悖论:若贪心非最优,搜得更狠理应能超——可它就是超不过。悖论逼我们怀疑"问题根本不在搜索"。

## III.3 反复出现的症状:val 高 / test 低
每个失败臂都带同一指纹:**训练信号上看起来最好的,test 上最差。** 两个失败臂(k8top4 val→0.725、k4struct→0.650)
是全场 val 最高、test 最低。**这是"选择"的病、不是"生成"的病——它在说:你拿来选的那个信号在骗你。** 我们被这个
背离烧过多次,定了铁律(先零成本 held-out 探针再花钱)。

## III.4 代码层面的硬证明:松开探索是 NO-OP
由 §I.2 的 `current==best`:"一边继续探索、一边单独记录至今最好"产生的版本序列**和贪心完全相同** →
**松开探索 = NO-OP**。即:**贪心已经把它能探索的都探索了;那条贪心轨迹本身就是完整的已探索集合。**
"搜索侧没油水"从"实验上超不过"升级成"构造上动不了"。

## III.5 决定性一击:oracle headroom —— 好 skill 早就在轨迹里
换个问法:**贪心自己走过的轨迹里,有没有一版其实比它 ship 的更好?**
测法(`tools/diag_eval_candidates.py` / `_oracle_headroom.py`):**允许偷看 test(oracle)**,在轨迹每版打 test 分,
找最高的那版,看它比贪心 ship 的高多少。

> 结果:**+2.5 ~ +14pt,4/5 cell,跨 4 benchmark × 2 模型**(Qwen headroom 均值 **+6.10,3/3**,比 DeepSeek 还大)。

这把解释翻了过来:之前"超不过贪心 → 也许贪心近最优 → 死路";之后"**贪心早就生成出了更好的 skill,就躺在轨迹里,
只是门没 ship 它**"。

### 必须分清:诊断 vs 实得增益
| | 诊断(证明瓶颈在选择) | 实得增益(可部署方法) |
|---|---|---|
| 问 | 轨迹里**是否存在**被门扔掉的更好 skill? | **不偷看 test**、只用 val60,能不能真捞回来? |
| 量 | 允许偷看 test(oracle headroom) | argmax-val60 选出的版在 test 上比门高多少 |
| 结果 | **+2.5~+14,4/5 cell、2 模型**(极稳健) | **SNR-gated**:headroom 大就捞到,≈噪声就捞不动 |
两者之差(oracle − C)= 没捞回的部分 = 被评测噪声吃掉的 headroom = 支柱③活生生的样子。

## III.6 精确结论:两条轴
"贪心把增益吃完了"**必须分轴说**:
| 轴 | 贪心吃完了吗 | 证据 |
|---|---|---|
| **生成 / 搜索轴** | **吃完了**(等预算搜不出更多) | 宽度饱和 K2≡K4 p=1.0、深贪心≈贪心 p=0.80、结构化更差 p=0.025、QD/蒸馏/adaptive 全平台、松探索 no-op |
| **选择轴** | **没吃**(还剩 +2.5~14) | oracle headroom 4/5 cell、2 模型 |
→ **唯一没被吃掉的杠杆 = 选择。** 这就是整个方向从"更聪明生成"掉头到"更可靠选择"的全部由来。
**尾注(诚实):** 桌上那点 selection headroom 也被噪声地板卡着(支柱③)→"吃完搜索增益"是铁的,"选择增益全拿回"是 SNR-gated 的。

---

# §IV 三个增益器(METHODOLOGY 全展开)

## IV.1 增益器① — Method C(解耦选择 / Decoupled Selection)

### Philosophy
直接落实支柱①②:把"收集证据(轨迹)"与"做出承诺(ship 哪版)"分开。探索时随便探索,最后**只承诺一次、
且在高功率证据下**——explore-then-commit / best-arm identification。

### Methodology — 三步流水线
```
[explore]  贪心训练(run_ssb_official.py) → 存下整条轨迹 skill_v0001..vN(含被拒的)
[select]   posthoc_select.py  ★我们唯一动的一步 → 每版在 held-out val60(⊥test)重评 → argmax-val → ship
[compare]  decoupled_ship.py → 门(best_skill) vs C(argmax-val),逐 seed + sign test
```
载荷性设计:
1. **探索 = 贪心,逐字节相同(NO-OP 证明,见 §III.4)** → 我们没碰探索;那条贪心训练**身兼两职**(baseline + 探索阶段)→
   **等预算公平对比**,唯一差别是最后 ship 哪版。
   - 门 ship 的 `best_skill` 是在又小又吵的 val18 上胜出(且被 slow-update 污染);C 把**整条轨迹**(含被拒候选)拿到
     **更大更干净的 val60** 重打再选 → 常选到不同的版。
2. **切分设计**:`train35 / val18(gate) / val60(C 选) / test82`,5 seeds。**val60 ⊥ test82(反 oracle 唯一硬要求);
   val60 与 val18 嵌套重叠无妨。** 其它 benchmark 镜像此设计(`data/<PRE>_v18` + `data/<PRE>_v60`)。
3. **ship 规则 = argmax-mean-val**(`qd/decoupled_select.py:argmax_mean`,平局取 id 最小,确定性)。
   - **试过更花哨的 Copeland**(半数据集对决投票,抗 winner's curse)→ **没帮上忙**(LiveMath argmax +6.5 vs Copeland +4.9)
     → **argmax 胜出(YAGNI)**。
4. **反 oracle(支点)**:选择**只碰 val**,test 只报告(code-review 验过无泄漏);`common_ids` 保证所有候选同题比较。
5. **诚实统计**:**pooled-McNemar 反保守**(82 test 题跨 seed 复用、不独立 → 虚高),**不信它**;诚实单位 = **seed-level sign test**
   (每 seed 当一个独立观测)。

### 代码
`qd/decoupled_select.py`(`argmax_mean`/`copeland_robust`/`common_ids`/`select`)+ `tools/posthoc_select.py`
(Method C,支持 `--target-backend qwen_chat` 跨模型)+ `tools/decoupled_ship.py`(逐 seed + pooled-McNemar + seed-level sign)
+ `tools/mcnemar_pooled.py` + `tools/diag_eval_candidates.py`(oracle headroom,`--limit` 子采样)+ `tools/make_experiment_splits.py`
(nested-val 切分,扩展支持 CSV envs)。`qd/tests/test_decoupled_select.py` 13 绿。

### 结果(诚实)
- **LiveMath × DeepSeek 5-seed**:s1+2.4 / s2+2.4 / s3+14.6(McNemar p=0.012 SIG)/ s4+1.2 / s5−1.2;**均值 +3.9,4/5 正**;
  **pooled 不显著**(seed-sign p=0.1875)。诚实读法:方向性赢、s3 又大又显著、其余温和。**主动降级过一次过度声称**(早先 3-seed
  +6.5/p=0.06 是 s3 离群带来的小样本乐观,被 code-review 的 seed-level 检验照出)。
- **OfficeQA × DeepSeek = 最干净的赢**:greedy 0.302(门栽最惨、轨迹有 0.45+ 被扔);**argmax +10.2,5/5,seed-sign p=0.031 SIG**
  (pooled-McNemar p=0.0004)。**但单模型。**
- **Qwen 跨模型**:诊断 3/3 复现(headroom +6.10),但实得增益 SNR-gated(argmax 均值 −2.44,1/3)→ **+10.2 跨模型嫌疑未洗清。**

### 边界
实得增益受**选择功率**限制,而功率受 **benchmark 大小**限制(见 §IV.2 的硬上限);headroom 大才捞得动。

---

## IV.2 增益器② — Inference-Aware Selection(siggate / lcb)

> ②和①是**同一条轴**(都在 ship 时选,代码同在 `decoupled_select.py`)。②升级的是①最后那一记 argmax。

### Philosophy
①的 argmax-mean **正是赢家诅咒惩罚的那个动作**(选最高点估计 → 选中者多半走运 → test 失望;argmax 在 3 个低 headroom
benchmark 都有负 seed、SSB 净负 −1.5)。②的范式转变:**别 ship 点估计最高的,ship 最可能在新 held-out 上复现的那版。**
文献根基:Smith & Winkler 2006(收缩是赢家诅咒原版解药)、Cawley & Talbot 2010(选择准则低方差和无偏一样重要)、
Bastani 2025(按"过显著性的概率"而非均值接受 = inference-aware,LLM-skill 上首次实例化 = open gap)。
接回支柱②:"偏离稳妥老选择"本身是有下行风险的承诺 → 证据不够强就不偏离、弃权 → **永不严格输(never-lose)**。

### Methodology — 共享地基 + 两条规则
**共享地基:逐题配对比较**(`qd/pairgate.py`,而非比平均分)。候选与 incumbent 在**同一批 val 题**打分 → 逐题看差值。
该 benchmark 逐题噪声**正相关 ρ≈+0.56**("难题对谁都难")→ 配对差值方差远小于均值之差(尺子变稳);这也正是
Smith-Winkler nuance:同题评所有候选使噪声正相关 → **缩小赢家诅咒量级**。配对比较在免费吃这个红利。

**规则一 `siggate`(显著性门,`significance_gated`):** 候选相对 incumbent 的逐题战绩 赢 w / 输 l,
**通过单边精确 sign test(p<α)才"有资格"**;有资格的取 argmax-mean;一个都没资格 → 返回 `INCUMBENT`(弃权)。
- sign test 直觉:若候选与 incumbent 无差别,分胜负的题应像公平硬币 50/50;它在 10 个分胜负题里赢 8 个有多稀奇 = p 值。
- 精确公式(McNemar 精确,纯 `math.comb`):`n=w+l`;`p = Σ_{k=w}^{n} C(n,k) / 2ⁿ`。

**规则二 `lcb`(下界,`paired_lcb`):** 逐题差 `d_i = 候选_i − incumbent_i`;`mean_d`;`SE = sd(d)/√n`(ddof=1);
**悲观分 `LCB = mean_d − z·SE`**;取 LCB 最高且 >0 ship;全 ≤0 → `INCUMBENT`。
- 单题 SE=∞ → 绝不在单题证据上冒进。`z` = 怀疑旋钮;**argmax 是 z=0 端点**,②把固定端点变成一整根可调轴。

两规则共同:可返回 `INCUMBENT` 哨兵(弃权 ship 贪心)→ Δ=0、never-lose;纯/zero-API/确定性;反 oracle(只在 val 选)。

### 可调旋钮直觉
```
大胆 ← argmax(分最高) … lcb(z 小) … lcb(z 大) … siggate(硬门) … 永远弃权 → 胆小
        ①的端点                                                  =退回贪心
```
**旋钮该拧到哪,取决于 SNR,每个任务不同** → 没有一个固定刻度能在所有 benchmark 上既赢又 never-lose。

### 代码
`qd/decoupled_select.py`(`significance_gated`/`paired_lcb`)+ `qd/pairgate.py`(`compare_paired`/`sign_test_p_one_sided`/
`paired_gate_action`)+ `tools/decoupled_ship.py`(路由规则、报 never-lose/abstain)。`qd/tests/test_inference_aware_select.py` 16 绿;
2-agent 审 0C/0H;反 oracle 验过。

### 结果 —— 比"赢"更深的东西(3 bench × 5 seed,反 oracle,离线零新 API)
**面一(低 headroom 3 任务):**
- **`siggate@0.05` 在 15/15 cell 全弃权** —— ~2–4pt 增益在 60 题上只净赢 1–3 题,**永远过不了 p<0.05**。松到 α≈0.2–0.3
  (= 放弃诚实显著性)才 fire,赢家诅咒立刻回来(SSB siggate@0.20 = −1.7)。→ **对"评测受限"支柱③最直接的实证。**
- **lcb 是 Pareto 安全中点**:SearchQA `lcb@1` 拿到和 argmax 一样的 +0.7 但 never-lose 5/5(argmax 4/5);LiveMath `lcb@1`
  +2.4 never-lose 5/5。
- **没有预注册固定刻度既赢又处处 never-lose**:`lcb@1` 在 SSB 小亏(−1.7);`z≥1.5`/`siggate` 处处 never-lose 但捕获 0;
  那个"处处 never-lose 又为正"的 z≈1.5 是**偷看 test 挑的 = 事后 oracle 调参,已披露、不算干净**。
- **天花板是数据集大小硬上限**:LiveMath 全池 **177 = test82+train35+val60**(`_lm_split_probe.py` seed7 复现 test82),
  **val60 已是最大可能 held-out**。且 **headroom ⊥ 池大小**:LiveMath +3.9/池177(封顶)、SearchQA +0.7/池2000、SSB −1.5/池400
  → 有真 headroom 的太小没法扩 val、池大的 headroom 又太小永远过不了显著 → **eval-power sweep 在现有套件解析地无意义,反相关本身就是结论。**

**面二(OfficeQA,headroom≫噪声):** greedy 0.302。argmax +10.2(5/5,p=0.031);**lcb@0.5 +10.2(5/5,p=0.031)** —— 连最诚实
软版都全捞到;lcb@1 +7.8(4/5);**siggate@0.05 仍全弃权 +0.0** —— 一个 test 上好 +10 的版在 val60 也只到 p≈0.1–0.2(val 噪声大)→
纯显著性门即便这里都嫌太严,恰到好处的是**轻度收缩 lcb@0.5**。

**合成(可写进论文):** 解耦选择能兑现的、**可复现的**增益 = headroom × capture,capture 被评测功率界住、功率被 benchmark 大小界住。
argmax 平均捕获多但付赢家诅咒;lcb 用"只捕获统计可分辨那部分"换 never-lose;诚实显著性门什么都捕获不到,因 headroom 在地板下。
**领域的杠杆是更便宜可靠的评测,不是更聪明的选择规则。** ②用一套实验把这个反转的两个方向都演了出来。

### 边界
val60 功率下诚实显著性在低 headroom 任务打不响 → 退化成贪心(这是诚实非失败);"干净 inference-aware 主张"还差一步——
旋钮(z/α)须在**与 test 无关数据**上标定(val60 内 nested-CV),预测会挑保守 z → ≈贪心 → 印证"这预算下连标定都做不动";
要展示 siggate "随功率上升开始打响"需换**同时大池 + 大单步 headroom** 的新 benchmark。

---

## IV.3 增益器③ — Drift-Guard(局部信用分配 / Localized Credit Assignment)

> ③在**另一条轴**(生成/训练时、在 D_sel 上),与 ①② **正交可叠**:③ 清洁轨迹,①② 从干净轨迹选 ship。

### Philosophy(重要澄清)
③**不是**"更聪明地生成",而是**修生成的一个缺陷**——不自打"瓶颈在选择"的嘴。SkillOpt 的 reflect 三毛病(读码核实):
①整块改、②只看聚合失败、③只看失败不看成功 = StraGo(EMNLP'24)命名的 **prompt drifting**(为修聚合失败整块改 → 打坏原本对的)。
核心洞察:**标量净分门分不清「+0 净」和「+3 修 −3 坏」**;③把被 net 掩盖的"坏了 3 个"翻出来。类比 = **软件回归测试**
(修 bug 时跑原来通过的用例,确认没改坏)。哲学动作 = **局部信用分配**:把判断(和将来的生成)拉到更细粒度(逐 item / 逐段)。

### Methodology — drift-aware 接受谓词
D_sel 上候选 vs incumbent 配对(hard 0/1)的 McNemar 不一致格:
- **`b` = 回归**(incumbent 对 & 候选错 = 改坏的);**`c` = 修复**(incumbent 错 & 候选对 = 修好的)。

| 谓词 | 公式 |
|---|---|
| 现有 vanilla 门 | accept ⟺ `c > b`(hard 0/1 上 ≡ `cand_mean > inc_mean`) |
| **dg 谓词** | accept ⟺ **`c − λ·b > 0`** 且 **`Acr = b/(incumbent对数) ≤ τ`** |

- **`λ≥1`** = 一次回归算 λ 次修复(drift-averse);**`τ`** = 回归率硬上限;`Acr` = StraGo 回归率(原本对的题被改坏几成)。
- **忠实 no-op 默认 `λ=1, τ=1`**:`c−1·b>0 ⟺ c>b ⟺ cand_mean>inc_mean`;且 `b ≤ incumbent对数` 恒成立 → `Acr≤1` 永不触发 →
  **逐字节同 vanilla 门同一接受动作**(回归测试断言 `==pairgate.compare_paired().better`)。拧大 λ 才厌恶 drift、拧小 τ 才加硬闸。
- **可选叠 sign test**(`require_significant`):仅当"修 vs 坏"不对称过单边 sign 检验才动手(噪声下弃权)。
- **两级结构**(`drift_gate_action`,镜像标量门):接受进轨迹(drift-aware 胜过**当前**)/ 标记新最佳(还得胜过**至今最佳**);
  记 `would_accept_plain`(纯净分)/`would_accept_net`(带 λ)→ 拒绝时能分清是 λ 拒的还是 τ 闸拒的。
- **反 oracle**:只读 D_sel;λ/τ 只在非 test(train/D_cal)标定。

### 最聪明的一手:成功指标不被噪声地板卡住
- **①② 是 eval-bound 的**(回报 = test 准确率,被 val 噪声地板封顶)。
- **③ 故意量别的东西**:**pooled Acr = Σ回归 / Σincumbent对数**,跨 step/seed 池化(单步噪、池化稳)。
> **即便 C-on-test 增益在 held-out 上 ns(支柱③预期),pooled Acr 显著下降本身就是干净、可发表的机制结论。**
> ③给自己挑了一个**不被那把抖尺子封顶**的成功指标——这是它吸取 ①② 教训后的关键设计。
(主指标 = pooled Acr↓;次 = 等预算 dg-轨迹 vs vanilla-轨迹的 C-on-test,多 seed + seed-level sign。)

### L1/L2(生成轴搭档,与门 L3 正交,pending)+ 升级最大档
- **L1 对照证据**:给 error-analyst 同时喂"要修的失败"+"必须保住的成功"(免费,prompt 级)。
- **L2 段定位 edit**:每条 edit 带 `section`/`direction`,优先 replace/insert_after 命中相关段,抑制盲目 append(免费,schema 级)。
- L1/L2 YAGNI:L3-only 见效后再上。
- **升级最大档(用户已授权 fallback)**:若满档既不降 Acr 又无 C 增益 → 对 protect-set 逐条 **leave-one-edit-out 重跑**定位惹祸 edit
  裁掉、重评;额外 eval **计入预算、显式 log**(不静默截断),仍守反 oracle。

### 代码 + 状态
`qd/drift_guard.py`(`regression_fix_counts`/`acr`/`fix_rate`/`pool_drift_counts`/`drift_aware_accept`/`drift_gate_action`,
26 测试绿,2-agent 审 0C/0H);trainer 接线 `gate_select="drift"`(镜像 paired、复用 `sel_per_item`)+ driver `--gate drift --dg-lambda/--dg-tau`;
全 qd 套件绿。trainer 改动本 gitignored,但 `repro/official/patches/drift_gate.patch` **已 commit**(修过一次缺失的 config.py hunk)→ 部署 box2 不卡。
**❗ 尚未跑过任何真实验**:首个实验 = **L3-only**(`--gate drift --dg-lambda 2.0` vs strict baseline,LiveMath ≥3 seed 等预算,主读 pooled-Acr↓)。

### STOP 纪律
不碰 test 选 / 不放宽反 oracle / λ-τ 只非 test 标定;v1 **不**上 proxy verifier(RR v1>v2 烧过:会误杀);不重建 exploration(NO-OP);
不碰 slow_update 保护区;最大档额外评测必须显式 log;**别 oversell**(单块 prompt 上"局部信用等预算赢"是 open 问题,强证据多在多组件 pipeline)。

---

# §V 三者如何拼成整体

- **暗线**:聚合数字是个骗子(§II.5)。
- **两条轴**:**①② 选择轴**(ship 时、val60、治"选错")、**③ 生成轴**(训练时、D_sel、治"改坏"),正交可叠。
- **时间线**:贪心探索(可选 ③ 守门)→ 轨迹 skill_v1..vN → ③ 已把轨迹擦干净 → ①② 在 val60 上选 ship。
- **关系**:②是①的 ship 规则升级(argmax → inference-aware);③在前面清洁 ①② 要选的池子。
- **总纲**:贪心门把"探索+选择"压成一次含噪不可逆的草率承诺;三个增益器系统性地拆开它;能兑现多少最终诚实地受限于那把尺子有多准
  ——这本身就是对"领域真天花板是评测成本"最有力的论证。

---

# §VI 学术定位

> *(本节为投稿前的学术定位策略,属内部材料,未随本公开仓库发布。)*

---

# §VII 结果汇总(全 benchmark × 全方法,诚实)

## VII.1 原始 SkillOpt K1 基线 + 我们的诊断
- 原始 K1:SSB 0.507 / LiveMath 0.468(注:+14.5 是 SkillOpt 本体、**非**我们的方法)/ SearchQA 0.804。
- **诊断(oracle headroom,门丢更好 skill)**:LiveMath-DS +3.2/+4.0/−2.4(2/3)、SearchQA-DS +2.5、SSB-DS +5.0、
  **Qwen-LiveMath +4.0(跨模型)**、OfficeQA-DS(候选 ~0.10–0.22 → 低)→ **4/5 cell 确认,跨 4 bench + 2 模型。**

## VII.2 Method C(①)实得增益
| benchmark × 模型 | greedy | C(argmax-val) | 读法 |
|---|---|---|---|
| LiveMath × DeepSeek 5-seed | 0.4268 | **+3.9**(4/5;s3 +14.6 p=0.012;pooled ns) | 方向性赢、温和 |
| **OfficeQA × DeepSeek 5-seed** | 0.302 | **+10.2**(5/5,seed-sign p=0.031 SIG) | **最干净的赢、单模型** |
| Qwen-LiveMath 3-seed | 0.451 | argmax **−2.44**(1/3);C+inc **+0.81** | SNR-gated、跨模型嫌疑未洗清 |

## VII.3 Inference-aware(②)前沿(3 bench × 5 seed)
| bench(greedy) | argmax | siggate@0.05 | lcb@1 |
|---|---|---|---|
| LiveMath(0.4268) | +3.9(4/5) | **+0.0 abstain 5/5** | +2.4(never-lose 5/5) |
| SSB(0.5244) | −1.5(net 负) | +0.0 abstain 5/5 | −1.7(lcb@1.5 则 +0.0 never-lose 5/5) |
| SearchQA(0.8171) | +0.7(4/5) | +0.0 abstain 5/5 | +0.7(never-lose 5/5) |
| **OfficeQA(0.302)** | **+10.2(5/5)** | +0.0 abstain 5/5 | lcb@0.5 **+10.2(5/5,p=0.031)** |
→ siggate@0.05 在低 headroom 三任务 **15/15 全弃权**(eval-bound 实证);lcb 处处比 argmax 安全;headroom≫噪声(OfficeQA)时干净大赢。

## VII.4 Drift-Guard(③)
建好、审过(0C/0H)、patch 已 commit、**尚未上机**。首个实验 = L3-only LiveMath ≥3 seed,主读 pooled-Acr↓。

## VII.5 base SkillOpt 在 Qwen 上(sanity,自优化本体强)
s1 0.232→0.488(+25.6)/ s2 0.256→0.415(+15.9)/ s3 0.171→0.402(+23.2)。

---

# §VIII 诚实纪律(贯穿全项目的方法论原则)
1. **反 oracle**:选择/标定**绝不碰 test**;test 只报告。代码审查验泄漏。
2. **等预算**:贪心训练同时充当 baseline + 探索阶段,唯一差别是 ship 选择 → 公平。
3. **诚实统计**:pooled-McNemar 反保守(test 题跨 seed 复用)→ **报 seed-level sign test**,不报好看的池化数。
4. **报保守、不 oversell**:主动降级过度声称(Method C 早先 +6.5 被自己降级);存在性证明(Qwen s2 +6.10)不当 headline。
5. **never-lose 机制**:弃权哨兵保证不严格变差。
6. **off-test 标定**:所有旋钮(z/α/λ/τ)只在非 test 数据标定,防事后调参。
7. **零成本先行铁律**:先零成本 held-out 探针再花钱(被 val/test 背离烧过多次)。
8. **诚实当资产**:文献报膨胀的赢(vs 人类 prompt、单跑、不等预算);我们报 ns/4-5/seed-level sign → 比脆的大数字更难被反驳。

---

# §IX 文献地图(每篇与我们的关系)
- **TextGrad**(2406.07496,Nature 2025)/ **ProTeGi**(2305.03495)/ **DSPy** —— "LLM 优化 = 文本版 BP"既定框架;TextGrad 自标 metaphor、
  选择是贪心循环 → 我们 ①② 填它留空的算子槽。
- **语义 backprop 批判**(2412.03624,Schmidhuber 组)—— 证 TextGrad 违反 reverse-mode AD → 砍"类比 BP"的链式法则保真度,尤其砍 ③。
- **Smith & Winkler 2006**(Optimizer's Curse)—— argmax 含噪估计系统性乐观 = "val 高 test 低"的定理;收缩是原版解药 → ②的根。
- **Cawley & Talbot 2010**(JMLR)—— 选择准则本身会过拟合、低方差和无偏一样重要 → ②/全命题的根;主张边界:只能"at-least-as-binding"。
- **Bastani 2025**(2510.18161,inference-aware policy optimization)—— ②的近乎逐字先例(winner's curse=Optimizer's Curse、按"过显著性概率"接受、Pareto 前沿)。
- **TRIPLE**(2402.09723,NeurIPS'24)—— prompt-opt ≡ 固定预算 BAI-FB、scoped 到选择 → ①最强锚 + "选择成本很少被显式建模"。
- **固定预算 Bayesian BAI**(2408.04869,TMLR'24)—— 误识别界 Õ(√(Hb/n)) → 元命题的形式版(Bayesian 平均情形,慎引)。
- **StraGo**(2410.08601,EMNLP'24)—— prompt drifting + Acr → ③的命名与度量根。
- **Miller/Anthropic 2024**(2411.00640)—— 分辨 0.03 效应需 ~1000 题 → 噪声地板 ±6.5pt,支柱③。
- **Atil 2024**(2408.04667)temp-0 不确定;**Madaan 2024**(2406.10229)进度可检测性由 metric 决定;**Zhang 2026**(2604.14585)
  prompt 优化 ≈ 抛硬币;**Dalal 2026**(2603.15377)更多 test-time compute 可能有害(n̂=1+exp(Δ²/2σ²))—— 旁证支柱③。

---

# §X 术语表 + 代码地图 + 复现

## 术语表
- **skill**:被迭代优化的单一自然语言工件(攻略)。
- **D_sel / val18 / val60 / test82**:gate 选择集 / Method C 选择集 / 全程封闭测试集。
- **incumbent / best_skill**:当前 / 滚动最佳;strict 贪心下二者恒相等。
- **oracle headroom**:偷看 test 找轨迹最优版 − 门 ship 的 = 诊断量(非方法)。
- **b / c / Acr**:回归数 / 修复数 / 回归率(=b/incumbent对数)。
- **siggate / lcb / argmax / Copeland**:四条 ship 规则(②的两条 + ①的两条)。
- **never-lose / abstain / INCUMBENT**:弃权退回贪心 → 不严格输。
- **seed-level sign test**:诚实显著性单位(每 seed 一个独立观测)。
- **SNR-gated**:实得增益随 headroom/噪声(支柱③)。

## 代码地图(我方核心)
- `qd/decoupled_select.py` —— ①② ship 规则(`argmax_mean`/`copeland_robust`/`significance_gated`/`paired_lcb`/`select`)。
- `qd/pairgate.py` —— 配对比较 + 精确 sign test(`compare_paired`/`sign_test_p_one_sided`/`paired_gate_action`)。
- `qd/drift_guard.py` —— ③ 纯函数(`regression_fix_counts`/`acr`/`drift_aware_accept`/`drift_gate_action`)。
- `tools/posthoc_select.py`(Method C,跨模型 `--target-backend qwen_chat`)/ `decoupled_ship.py`(对比 + seed-level sign)/
  `mcnemar_pooled.py` / `diag_eval_candidates.py`(oracle headroom)/ `make_experiment_splits.py`(切分)。
- trainer 接线:`gate_select="drift"`(drift_gate.patch);driver `repro/official/run_ssb_official.py`(`--env`/`--gate`/`--dg-lambda/--dg-tau`)。

## 复现要点
- 切分:`train35 / val18 / val60 / test82`,5 seeds,等预算。`PRE`/`ENV`:ssb_sel/spreadsheetbench、sq_sel/searchqa、
  oq_sel/officeqa、lm_sel/livemathematicianbench。
- 离线规则扫(zero-API):`tools/decoupled_ship.py --tags <PRE>A_v18_s1..s5 --rules argmax,copeland,siggate,siggate@0.20,lcb@0.5,lcb@1,lcb@1.5`。
- 单测:`python -m pytest qd/tests -q`。
- 服务器:box2 `<server>`,repo `/root/skillopt-fullrun-gatesweep`,python `/root/miniconda3/bin/python`(绝对路径)。
