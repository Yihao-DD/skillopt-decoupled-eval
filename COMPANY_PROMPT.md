# 任务 prompt:在 Qwen 上评测 SkillOpt + 解耦选择(argmax / siggate / lcb 三优化器)

> 把下面分隔线**以下的整段**,交给公司内网的执行 agent。它是**自包含**的:照着 0 → 5 步做即可。
> 本文档**刻意写得很细** —— 任何一步报错,先翻文末「**故障排查**」对症;**务必先看「第 2.5 步 · 先冒烟」**,用几毛钱验证整条链路再跑全量。

---

你是公司内网的执行 agent。手上这个仓库是一个**已完全打包、傻瓜式**的评测包。

## 0. 先读懂:这个包在干什么、你要产出什么

**背景(一段话):** SkillOpt 是一个「测试时自我改进」框架 —— 它让一个模型反复 (a) 反思并改写一份叫 *skill* 的自然语言指令(**optimizer** 角色),(b) 把当前 skill 喂给**同一个模型**去答题打分(**target** 角色),贪心地保留分最高的 skill。**它的毛病**:贪心门在一个很小、很吵的验证集上**逐步、不可逆**地挑 skill,常 ship 一个过拟合的(验证集高、测试集反而低)。

**我们加的东西(本包重点):** 三个**解耦选择优化器** —— 不在过程中贪心挑,而是**把整条轨迹里所有 skill 都留下**,跑完后在一个更大的留出验证集上重新挑(**反 oracle**:只用验证集挑,test 只用于报告):
- **`argmax`(= Method-C,增益引擎)**:选留出验证集均值最高的 skill —— 主要的「增益」来源。
- **`siggate`**:只有某 skill **显著**打赢贪心基线才换,否则**弃权**(保平)。「永不显著输」。
- **`lcb`**:按优势的置信下界挑,否则弃权。也是「保守、不输」型。

**你要产出:** 对每个 benchmark,一张「**argmax / siggate / lcb 三优化器 vs 贪心基线**」的增益判决表(见第 4 步怎么读)。

**一个模型干所有事:** 整个流程**自始至终只用 `.env` 里那一个 Qwen 模型** —— 它同时当 optimizer(反思改 skill)和 target(答题)。**绝不掺第二个模型 / 厂商。**

## 已经替你处理好的环境约束
- **内网 / 只能 pull**:代码 + 3 个 benchmark(LiveMath / SearchQA / SpreadsheetBench)的数据**已打包在仓库里**,完全离线、不联网。
- **只有一个 Qwen API**:见上,全程单模型。
- **OfficeQA 是唯一要联网的**:文档太大没打包,用**团队负责人单独发你的** HuggingFace token 在线拉(一次性、几百 M)。内网拉不到 → 它会**自动跳过**,不影响另外 3 个。

## 环境自查(动手前 30 秒)
- **Python ≥ 3.10**:`python --version`。低于 3.10 必须先装新的(SkillOpt 用了新语法,3.9 会语法报错)。
- **磁盘**:代码 + 数据约 50M;若跑 OfficeQA 再留 ~1G 给下载的文档。
- **联网**:前 3 个不需要;OfficeQA 需要内网能访问 `huggingface.co`(或镜像)。
- **能调 Qwen**:确认公司给的 DashScope key 能用、要测的模型已开通(见第 0 步)。

---

## 第 0 步 ——【先问用户】跑哪个 Qwen 模型
**在装任何东西之前,先问用户一句:**
> 「要在哪个 Qwen 模型上评测?(例如 `qwen-plus` / `qwen-max` / `qwen-turbo` / `qwen2.5-72b-instruct` —— 取决于你们 DashScope 账号开通了哪个)」

拿到确切模型名再继续。它会原样填进 `.env` 的 `MODEL_NAME`,也是整条链路唯一用的模型。

## 第 1 步 —— 装环境
```bash
# Windows(cmd 或 PowerShell,在仓库根目录)
python -m venv .venv
.venv\Scripts\python -m pip install -e ./SkillOpt -e . -r requirements-extra.txt

# Linux / Mac 把第二行换成:
# .venv/bin/python -m pip install -e ./SkillOpt -e . -r requirements-extra.txt
```
- 这条会装:**SkillOpt 本体** + **本包的 qd/ 选择器** + 数据物化/测试要的 `datasets` / `huggingface_hub` / `pytest`。
- **后面所有 `python` 都用这个 venv 里的**(Windows:`.venv\Scripts\python`;Linux:`.venv/bin/python`)。下文为简洁写 `python`,请自行替换。
- **验证装好了**(零 API、约 30 秒):
  ```bash
  python -m pytest qd/tests -q          # 期望:41 passed
  ```
  41 passed = 选择器逻辑 + 接线都正常。报错 → 翻「故障排查 · 安装」。

## 第 2 步 —— 写 `.env`(仓库根目录新建,4 行)
```
MODEL_PROVIDER=qwen
MODEL_API_KEY=<你们公司的 Qwen / DashScope API key>
MODEL_NAME=<第 0 步问到的模型,例如 qwen-plus>
HF_TOKEN=<HF_TOKEN_FROM_TEAM_LEAD>     # 仅 OfficeQA 用;团队负责人单独发你;不跑 OfficeQA 可留空
```
逐行说明:
- `MODEL_PROVIDER=qwen` → 自动用 DashScope 的 OpenAI 兼容端点 `https://dashscope.aliyuncs.com/compatible-mode/v1`,**URL 不用你填**。
- `MODEL_API_KEY` → 公司的 DashScope key(形如 `sk-...`)。
- `MODEL_NAME` → 第 0 步那个模型名,**必须是你们账号已开通的**,否则 Qwen 返回 model-not-found。
- `HF_TOKEN` → 只有跑 OfficeQA 才需要。大陆内网若 HF 直连不通,可在 `.env` 再加一行 `HF_ENDPOINT=https://hf-mirror.com`。
- **验证配置**(不花钱):
  ```bash
  python model_config.py        # 打印解析到的 provider / 模型 / base_url(key 打码)
  ```
  配错(缺 key、provider 拼错)会有明确报错。

## 第 2.5 步 —— 先冒烟(强烈建议,几分钟 / 几毛钱)
**别一上来就跑全量**(全量是几小时 + 真金白银)。先用最小代价验证「key 能调通 + 数据在 + 执行器正常」:
```bash
python run.py --env livemathematicianbench --smoke
```
`--smoke` = 1 个 seed、1 个 epoch、最小规模,会真打几次 Qwen。看到它正常跑完、`rc=0`、生成 `SkillOpt/outputs/smoke_livemathematicianbench_s1/summary.json` = 链路通了,可上全量。**冒烟报错就在这里解决掉,别带到全量。**
(想顺带验 SpreadsheetBench 的代码执行器:`python run.py --env spreadsheetbench --smoke`。)

## 第 3 步 —— 全量跑(一条命令跑完所有 benchmark)
```bash
python run_all.py
```
它对 4 个 benchmark **串行**做:
1. **贪心 SkillOpt 跑 3 个 seed(1/2/3)** —— 每 seed 一条轨迹,默认 4 epoch;
2. **每个 seed 做 posthoc 高功率重评** —— 在更大的留出验证集上把轨迹里每个 skill 都打分(零新逻辑,就是多评测);
3. **`decoupled_ship`** 输出 **argmax / siggate / lcb 三优化器**跨 3 seed 的判决;

最后打印一张「**每个 benchmark × 三优化器**」的判决表。

**常用选项:**
- 先不碰 OfficeQA(只跑 3 个离线的):`python run_all.py --skip-officeqa`
- 只跑某几个:`python run_all.py --only livemathematicianbench searchqa`
- 机器弱 / 先快看:`python run_all.py --epochs 2`(默认 4)
- 调并发:`python run_all.py --workers 16`(默认按各 benchmark 配置;太高会被 Qwen 限流)

**各 benchmark 注意:**
- **LiveMath**:选择题、判分确定、最快最省最稳 —— 建议第一个看它。
- **SearchQA**:自由问答、离线、中等速度。
- **SpreadsheetBench**:target 要**写 Python 改 Excel**,本机有**代码执行器**在跑(已确认 Windows 安全:临时文件 + 子进程);比前两个慢。
- **OfficeQA**:**最贵最慢**(多轮、要先下文档),放最后。没 `HF_TOKEN` 或拉不到 → 自动标 SKIPPED,不影响其它。

**容错:** 某个 benchmark 报错**不影响其它** —— 它独立串行,表里会标 `FAILED` / `SKIPPED` / `PARTIAL`,其余照常 `OK`。

## 第 4 步 —— 读判决表
表里每个 benchmark 一段,看三优化器的 `d_seedmean`(= 3 seed 平均、相对贪心基线的增益,单位是准确率百分点):
- **`argmax`(Method-C,增益引擎)**:`d_seedmean > 0` = 解耦选择**比贪心好**,这是主要增益信号。**它可能为负** —— 它会付「优化者诅咒」的代价、在低信噪比 benchmark 上可能输。**如实报,别藏负数。**
- **`siggate` / `lcb`(保守、永不显著输)**:通常 ≈ 0(没显著增益就弃权保平),**只有真有显著增益时才 > 0**。它们 ≈ 0 是**符合预期**(是「安全」,不是「失败」)。
- 直觉:**argmax 博增益、siggate/lcb 保下限**。理想结果 = 「argmax 在有空间的 benchmark 上明显 > 0,siggate/lcb 至少不输」。

## 第 5 步 —— 报告给用户
把判决表整理后报给用户,每个 benchmark 给出:
- 状态(OK / FAILED / SKIPPED / PARTIAL);
- argmax / siggate / lcb 三个 `d_seedmean`;
- 一句话结论(argmax 有没有正增益、siggate/lcb 有没有保住)。

有 FAILED / PARTIAL 就附上你从「故障排查」判断的原因。

---

## 故障排查(遇报错先查这里)

**安装 / 环境**
- `pip install` 报错、找不到包 → 确认 Python ≥ 3.10(`python --version`);确认在仓库根目录;确认用的是 venv 里的 python。
- `pytest` 不是 41 passed / import 报错 → 多半 `-e .` 或 `-e ./SkillOpt` 没装上,重跑第 1 步整条 install。
- `ModuleNotFoundError: skillopt` / `qd` → 同上,install 没成功;或没在仓库根目录跑。

**模型 / API**
- `model_config.py` 报错 / 缺 key → 检查 `.env` 四行拼写、`MODEL_API_KEY` 有没有填。
- **401 / invalid api key** → key 错或没权限,换正确的 DashScope key。
- **model not found / 模型不存在** → `MODEL_NAME` 这个模型你们账号没开通,换一个已开通的(回第 0 步问用户)。
- **rate limit / 429 / 限流** → 降并发 `python run_all.py --workers 8`(甚至 4),或错峰跑。
- 大量 **timeout / 连接错误** → 内网到 DashScope 不稳;确认能访问 `dashscope.aliyuncs.com`;降并发重试。

**OfficeQA**
- 标 `SKIPPED` → 没 `HF_TOKEN` 或内网拉不到 HF。要跑它:填 `HF_TOKEN`(团队负责人给)+ 确认内网能上 `huggingface.co`;大陆内网在 `.env` 加 `HF_ENDPOINT=https://hf-mirror.com`。
- 下载卡 / 慢 → 文档几百 M,首次较久;断了重跑 `run_all.py` 会续传。不想跑就 `--skip-officeqa`。

**结果异常**
- 某 benchmark 标 `PARTIAL`、判决表那段空 / `nothing pooled` → 那个 benchmark 的 posthoc 评测全失败了(通常被 API 错误连累)。往上翻日志里 posthoc 的报错(多半限流/key/超时),解决后单独重跑:`python run_all.py --only <benchmark>`。
- 某 benchmark 标 `FAILED` → 它的贪心 run 非零退出;翻 `SkillOpt/outputs/run_<benchmark>_s1/driver.log` 看具体报错。
- 三优化器 `d_seedmean` 有负数 → **不是 bug**。argmax 在低信噪比 benchmark 上可能输(优化者诅咒);siggate/lcb 该弃权就弃权。如实报全部数字。

**Windows**
- 路径含冒号之类的报错(WinError 267) → 本包已修(`safe_id` 把非法字符换成 `_`),理论上不会再有;若仍遇到,记下报错发回团队。
- `&&` 在 PowerShell 报错 → 第 1 步两行分开运行,或用 cmd。

**重跑某一个 / 省钱**
- 只重某 benchmark:`python run_all.py --only <name>`。
- 只重某 seed 的训练:`python run.py --env <name> --seeds 2`。
- 想先便宜验证整条链:回第 2.5 步 `--smoke`,或 `python run_all.py --only livemathematicianbench --epochs 1`。

## 成本 / 时间预期(心里有数)
- 这是**真实评测**:4 benchmark × 3 seed × 4 epoch + 每个 skill 的 posthoc 重评,**会调很多次 Qwen、跑几个小时**(LiveMath 最快,OfficeQA 多轮最久)。
- **建议节奏**:先 `--smoke`(几毛钱)→ 再 `--skip-officeqa` 跑 3 个离线的看结果 → 最后再决定要不要花更多跑 OfficeQA。
- Qwen 费用按你们 DashScope 计费,本包不控制;`--epochs 2` / `--workers` 可在「省钱」与「跑满」间权衡。

## 一句话总结
**问用户跑哪个 Qwen 模型 → 装环境(`pytest` 41 passed)→ 写 4 行 `.env`(`model_config.py` 验证)→ 先 `--smoke` 冒烟 → `python run_all.py` → 读判决表 → 报告。** 遇错先翻「故障排查」。
