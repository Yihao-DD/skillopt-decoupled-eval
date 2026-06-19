# 任务 prompt:在 Qwen 上评测 SkillOpt + 解耦选择(3 优化器)

> 把下面这整段作为指令交给公司内网的执行 agent。它是自包含的:照着做即可。

---

你是公司内网的执行 agent。手上这个仓库是一个**已经完全打包好、傻瓜式**的评测包。目标:在**一个 Qwen 模型**上跑 4 个 benchmark 的 SkillOpt 自优化 + 我们新加的 **3 个解耦选择优化器(argmax / siggate / lcb)**,得到每个 benchmark 的「三优化器 vs 贪心」增益判决。

## 已经为你处理好的环境约束
- **内网 / 只能 pull**:代码 + 3 个 benchmark 的数据(LiveMath / SearchQA / SpreadsheetBench)**已经打包在仓库里**,这 3 个完全离线、不需要联网。
- **只有一个 Qwen API**:整个流程**自始至终只用这一个 Qwen 模型**(它同时当 optimizer 反思和 target 答题,绝不掺第二个模型)。
- **OfficeQA**:它的文档有 ~788M,太大没打包 → 用**团队负责人单独发你的** HuggingFace token 在线拉(需要内网能访问 huggingface.co;拉不了会自动跳过,不影响其它 3 个)。

## 按顺序做这几步

### 第 0 步 ——【先问用户】跑哪个 Qwen 模型
在动手前,**先问用户一句**:「要在哪个 Qwen 模型上评测?」
(例如 `qwen-plus` / `qwen-max` / `qwen-turbo` / `qwen2.5-72b-instruct` 等,取决于你们 DashScope 账号开通了哪个。)拿到答案再继续。

### 第 1 步 —— 装环境
```bash
# Windows
python -m venv .venv
.venv\Scripts\python -m pip install -e ./SkillOpt -r requirements-extra.txt
# Linux/Mac 把上面第二行换成： .venv/bin/python -m pip install -e ./SkillOpt -r requirements-extra.txt
```
(后面所有 `python` 都用这个 venv 里的 python。)

### 第 2 步 —— 写 `.env`(根目录新建,4 行)
```
MODEL_PROVIDER=qwen
MODEL_API_KEY=<你们公司的 Qwen / DashScope API key>
MODEL_NAME=<第 0 步问到的模型，例如 qwen-plus>
HF_TOKEN=<HF_TOKEN_FROM_TEAM_LEAD>   # 团队负责人会单独把这个 token 发你；公开库里不放真 token
```
- `MODEL_PROVIDER=qwen` 会自动用 DashScope 的 OpenAI 兼容端点(`https://dashscope.aliyuncs.com/compatible-mode/v1`),URL 不用你填。
- 验证配好了：`python model_config.py` —— 会打印解析到的模型(key 打码)。配错会有明确报错。

### 第 3 步 —— 一条命令跑完所有 benchmark
```bash
python run_all.py
```
它会**串行**跑 4 个 benchmark,每个 benchmark：
1. 贪心 SkillOpt 跑 **3 个 seed**（1/2/3）；
2. 每个 seed 做 posthoc 高功率重评；
3. `decoupled_ship` 输出 **3 个优化器(argmax / siggate / lcb)** 的判决；
最后打印一张 **「每个 benchmark × 三优化器」的判决表**。

可选：
- 只跑打包好的 3 个、先不碰 OfficeQA：`python run_all.py --skip-officeqa`
- 机器弱 / 先快速看：`python run_all.py --epochs 2`（默认 4 epoch）
- 单独某几个：`python run_all.py --only livemathematicianbench searchqa`

注意：
- 这是**真实评测**,4 epoch × 3 seed × 4 benchmark 串行，**会跑几个小时**(OfficeQA 是多轮、最久)。
- OfficeQA 第一次会自动拉 ~788M 文档（一次性）。内网拉不了 HF 就会自动跳过它，其余 3 个照常出结果。
- 某个 benchmark 报错不影响其它（独立串行，最后表里会标 FAILED/SKIPPED）。

### 第 4 步 —— 把结果报给用户
把 `run_all.py` 末尾那张判决表整理后报给用户。每个 benchmark 看：
- **`argmax`**（= Method-C,增益引擎）的 `d_seedmean`：**>0** 表示「解耦选择」比贪心好（3 seed 平均的增益）；
- **`siggate` / `lcb`**（保守的「永不输」安全选择器）：通常 ≈ 0（没显著增益就弃权保平），有真增益时才 >0。

## 一句话总结
**装环境 → 写 `.env`（Qwen 模型 + key + 团队负责人单独发你的 HF token）→ `python run_all.py` → 把判决表报给用户。** 就这么简单。
