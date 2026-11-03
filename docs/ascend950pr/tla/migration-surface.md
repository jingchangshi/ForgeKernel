# ForgeKernel × TLA：迁移面测绘

**作者**：trajectory-writer（task-9）。**日期**：2026-11-03。
**性质**：事实测绘，**不含设计建议**。本文回答"TileLang 在 ForgeKernel 里嵌得有多深"，不回答"该不该迁"。

标签沿用全项目约定：**[verified: 路径:行]** / **[source-only]** / **[inferred]** / **[unknown]**。

> **⚠️ 更正（2026-11-03，task-12）：本节初版称"本机没有可安装的 TLA"，该结论是错的。**
> 准确的表述是：**默认的 tilelang 环境**（`s00653124_tilelang`）里没有 `tla`——那 6 个 wheel 文件
> 与 `ls` 的结果都是真的 [verified]，但它们答的是"磁盘上的 wheel 文件里有没有"，
> 不是"本机有没有能用的"。**本机另有专用 conda 环境 `s00653124_tla`**，
> 装有 `triton_ascend-3.6.0.dev20261101+git2a00c605`（21 个 tla .py，RECORD 43 条），
> 可导入、可编译、可在 NPU 上执行，**tla-npu 门已实测 PASS**
> （run `9e8c59f1c356badc32a0e0d396af4659`，11/11）。运行路径见 [`how-to-run.md`](how-to-run.md)，
> 细节见 [`README.md`](README.md) §2。
> 仍成立的是：**我们的 rmsnorm 内核从未以 TLA 编译过**——这是另一件事。
>
> **可复现的证伪：** `python3 docs/ascend950pr/tla/probes/verify-tla-runnable.py`（十项前提全过才 exit 0）。
>
> **因此"从 TileLang 迁到 TLA"是 DSL 家族的更换**（TileLang → Triton + `tla.*` 命名空间），
> 不是同族版本升级。TLA 自己的 `__init__.py` 说得很清楚：
> "Independent `tla.*` programming namespace for Ascend NPU kernels … It does **not** reuse the existing
> `al`/`bl` general interfaces (**different programming paradigm**)."
> [verified: `triton-ascend/third_party/ascend/language/tla/__init__.py:23-25`]

---

## 0. 一句话结论

**ForgeKernel 的 Ascend 支持在"内核编译"这一层是 TileLang 的，在其余各层基本不是。**
但"其余各层"里有两处是真耦合，而不是可插拔的适配点：**注册门（7 个独立断言全部拒绝）**
和 **NPU devicetime 计时基（借 `tilelang.profiler.msprof` 的缓存刷新内核）**。
监管资产——配对同窗口判定机制——**是语言无关的，可以原样带走**；但它的 NPU 计时基不是。

**而迁移将确定性地丢掉 1.103×**，除非一个尚未做的实验证明 `EvictFirst` 能替代 `NOTALLOC_CLEAN`。
详见 §6，那一节是本文对决策最重要的部分。

---

## 1. 两个"language"概念——先分清，否则整张表都会错

这是最容易出错的地方，也是 `"triton"` 陷阱的根源。

| 概念 | 定义处 | 取值 | 用途 |
|---|---|---|---|
| **task language** | `task.py:82` `LANGUAGES` | `("cutedsl", "cuda", "tilelang-ascend")` | task.toml 的 `[task] language` |
| **loop language** | `lang_mode.py:52,58,63` | `CUDA_LANGUAGES ∪ CUTEDSL_LANGUAGES ∪ TILELANG_LANGUAGES` | `loop.toml [loop].language` / `--language` |
| **leaderboard/export language** | `leaderboard.py:1363-1368` `_EXPORT_LANG` | `cutedsl, triton, pytorch, cuda-sol` | 导出到 FlashInfer-Bench / SOL 的 `spec.language` |

三者**不是同一个枚举**，且互不校验。`task.py:84` 的 `LOOP_LANGUAGE` 是 task→loop 的映射表，
只有 3 项；`cli.py:1983-1984` 从三个 mode 集合**重新算**出 `--language` 的 choices。
[verified: `task.py:82-85`, `lang_mode.py:52-63`, `cli.py:1980-1984`, `leaderboard.py:1361-1368`]

**注册一个新的 Ascend 语言，需要同时满足前两套；第三套会静默错标（§2.3）。**

---

## 2. `"triton"` 这个名字：今天是什么，以及为什么它是陷阱

### 2.1 今天的含义：它是 loop 语言，但被登记在 **CuTe DSL 模式**下

```python
CUTEDSL_LANGUAGES = frozenset({"cutedsl", "cuda-cpp", "inline-ptx", "triton",
                               "cutlass-cpp"})
```
[verified: `lang_mode.py:58-59`]

实测后果（本文直接跑 `mode_of` 取得，非推断）：

| 表达式 | 结果 |
|---|---|
| `mode_of("triton")` | `"cutedsl"` |
| `kb_root_name("triton")` | `"hw_kb/cutedsl"` |
| `PROVENANCE_SUBDIR["cutedsl"]` | `""`（即 `docs/kb-provenance/` 根） |
| `PROMPT_OVERLAY_DIR["cutedsl"]` | `None`（读共享 `meta/<prompt>`） |
| `is_tilelang("triton")` | `False` |
| `language_problem("triton", "kernel.py")` | `None` ← **放行** |

[verified: 本文 `python -c` 直接调用 `lang_mode`，18/18 `tests/test_language_registry.py` 通过]

**所以 `"triton"` 是一个可以启动的真实 loop 语言，它的 KB 是 CUDA/CuTe 的 `hw_kb/cutedsl/`，
它的提示词是 NVIDIA 的共享提示词，它不报错。**

### 2.2 名字碰撞的真实形态：不是"重名冲突"，是"**已占用但语义错误**"

`"triton"` 与 `"triton-ascend"` / `"tla-ascend"` 是**不同字符串**，所以不会触发重复注册。
真正的危险是另外两条：

**(a) 一个已登记的名字带着 NVIDIA 语义。** 若有人为 Ascend-Triton 直接复用 `"triton"`
（最自然的命名冲动），`mode_of` **不会报错**，会静默把它路由到 `hw_kb/cutedsl/`——
即 CuTe DSL 知识库、CuTe 提示词、`cutlass.cute` 探测工具。
这正是 `lang_mode.py` 作者在模块 docstring 里点名担心的那件事：

> "Registering `tilelang-ascend` under that rule would have handed a TileLang student the
> CuTe DSL knowledge base, the DSL recipes and the `cutlass.cute` probe tools -- **silently,
> with no line anywhere saying the run was reading a KB for an API its kernel cannot compile**."
> [verified: `lang_mode.py:32-35`]

`UnknownLanguage` 是为此建立的防线（`lang_mode.py:82-93`），而 `"triton"` 是**这道防线上唯一的既存漏洞**：
它合法、放行、且指向错误的 KB。新增 Ascend-Triton 语言时若复用该字符串，
防线不会响——因为它已经在这个集合里了。

**(b) 新名字不会碰撞，但会**静默降级**。** 实测：`mode_of("triton-ascend")` / `mode_of("tla-ascend")`
均抛 `UnknownLanguage`（这是好的，符合设计）。但 `dsl_source.resolve(language="tla-ascend")`
**不抛**，返回 `notes=("language='tla-ascend': no DSL source to default",)`
[verified: `dsl_source.py:253-254`]。即：注册门会拦，但 **DSL 源码解析门会静默给空**——
学生拿不到任何 API 源码，而 run 不报错。这是一个真实的"注册了新语言却没接 DSL 源码"的静默失败点。

### 2.3 第三个错标点：导出会把它写成 `triton`

`infer_language` 先看 `loop_language in ("cutedsl","triton")`，否则**按 import 猜**；
`tla` 源码必然 `import triton`，于是猜成 `"triton"`，导出成 `spec.language="triton"`,
SOL `dependencies=["triton >= 2.3"]`。

实测：`infer_language(<TLA 源码>, loop_language="tla-ascend")` → `"triton"`，
`_EXPORT_LANG["triton"]` → `("triton", ["triton"], ["triton >= 2.3"])`
[verified: `leaderboard.py:1371-1386`, `1363-1368`；本文直接调用]

**一个 Ascend 内核会被导出成 NVIDIA `triton` 解。** `_EXPORT_LANG` 是闭集字典，
`infer_language` 是闭集元组——两处都需要扩，且**都不会自己报错**。

### 2.4 建议的名称检查（事实陈述，非设计）

本文不提议命名。只陈述约束：新语言字符串**不得**是 `"triton"`（已占用且语义错误），
且必须在 `task.py:82` / `task.py:84` / `task.py:90` / `taskcard.py:37` /
`lang_mode.py:52-74` / `leaderboard.py:1363` 六处同时出现，
漏任一处即 §4 表中对应的一行失败。

---

## 3. 组件表：TileLang 到底嵌在哪

`specific?` 取值：**yes** = 该处代码本身为 TileLang 而写；**no** = 完全语言无关；
**indirect** = 逻辑语言无关，但通过别的模块间接依赖 TileLang。

| # | component | file:line | tilelang-specific? | 一个 Triton-based 语言需要什么 |
|---|---|---|---|---|
| 1 | mode 注册表 | `lang_mode.py:52-74` | **no**（纯字符串→路径映射） | 加一个 mode 常量 + 四张表各一项 |
| 2 | `UnknownLanguage` 门 | `lang_mode.py:82-93,96-112` | no | 无需改；新名字自动被覆盖 |
| 3 | kernel 文件后缀规则 | `lang_mode.py:64,123-124,177-197` | **yes**（`.py`→TileLang 硬绑定） | `TILELANG_KERNEL_SUFFIXES=(".py",)` 与新语言**同后缀**，须新增 mode 分支 |
| 4 | task 语言表 | `task.py:82` | no | 加一项 |
| 5 | task→loop 映射 | `task.py:84-85` | no | 加一项；漏则 `KeyError` |
| 6 | arch 正则 | `task.py:90-96` | **indirect**（`"tilelang-ascend": ^ascend\d+[a-z]*$`） | 加一项；否则回落到 `^sm\d+$` 而**拒掉** `ascend950pr` |
| 7 | arch 示例 | `task.py:97-100` | no | 可加，仅影响报错文案 |
| 8 | KB 目录校验 | `task.py:277-280,475-479` | **indirect**（经 `LOOP_LANGUAGE`→`kb_root_name`） | 需新 KB 树存在，否则 exit |
| 9 | KERNEL_FILE | `taskcard.py:37` | no | 加一项；漏则 `KeyError` |
| 10 | DSL 源码解析 | `dsl_source.py:37,250-254` | **yes** | 新增一个 `_triton_source()` 分支；**漏则静默空**（§2.2b） |
| 11 | `dsl_source` fallback 常量 | `dsl_source.py:49-50` | **yes**（`TILELANG_FALLBACK_SOURCE` 硬编码绝对路径） | 需并行的 triton 路径常量 |
| 12 | agent policy notice | `agent_policy.py:426-430,454-458` | **indirect** | 三分支已够（`else "DSL package sources"`），文案不精确但不阻断 |
| 13 | agent policy 源码校验 | `agent_policy.py:686-701` | **yes** | 需要新分支；否则走 `cutedsl` 分支检查错键 |
| 14 | policy 构造 | `agent_policy.py:819-820` | no（只特判 cuda） | 无需改 |
| 15 | **NPU devicetime 计时基** | `fk_dev.py:253-259,278-290` | **YES —— 唯一真正的运行时依赖** | 见 §5。TLA 无 `bench_with_msprof` 等价物 |
| 16 | msprof 缓存刷新内核 | `tilelang/profiler/msprof.py:15-36` | **YES**（用 `TA.SimtVF`+`l2_cache_ctrl="WTS_FV"` 写成） | 须用 TLA 重写一个 flush 内核，或放弃 msprof 基 |
| 17 | timing 窗口 NPU 分支 | `timing.py:212-219` | **indirect**（只调 `fk_dev`） | 无需改 |
| 18 | timing 基标签 | `fk_dev.py:293-301` | no（`msprof`/`npu_events` 两值） | 无需改 |
| 19 | **judge.py** | `judge.py` 全文 280 行 | **no（0 次 tilelang 引用）** | **无需改** |
| 20 | **device.py** | `device.py:1-83` 全文 | **no（0 次）** | **无需改**；`kind_of_arch` 按 `ascend` 前缀判定（`:41-49`） |
| 21 | **timing_audit.py** | `timing_audit.py` 346 行 | **no（0 次 tilelang / 0 次 triton / 0 次 npu 字符串）** | **无需改** |
| 22 | **window_contract.py** | `window_contract.py:1-1231` | **no（0 次 tilelang）** | **无需改**；74 处 npu 是设备无关的 kind 分支 |
| 23 | precision.py | `precision.py` 421 行 | **no（0 次）** | **无需改** |
| 24 | closed_ref.py | `closed_ref.py` 全文 | **no**（3 次 `triton` 全是"名字不得碰撞"的注释，非依赖） | **无需改** |
| 25 | ncu_probe.py | `ncu_probe.py` 13 行 | no | 无需改（CUDA-only 工具，NPU 不用） |
| 26 | eval/leaderboard.py | `eval/leaderboard.py:550` | no（`triton` 仅出现在禁止泄漏的 regex） | 无需改 |
| 27 | leaderboard 导出 | `leaderboard.py:1363-1368,1371-1386` | **no，但闭集** | 加一项 + 加 `infer` 分支；**漏则静默错标**（§2.3） |
| 28 | task 卡 | `tasks/rmsnorm*/task.toml:5-6` | no（只是字符串） | 改 `language`；KB 树须存在 |
| 29 | `--language` choices | `cli.py:1983-1984,2207` | no（自动从三集合取并） | 注册后自动包含 |
| 30 | 启动门 | `cli.py:2415-2426` | no | 无需改 |
| 31 | KB 内容 | `hw_kb/tilelang-ascend/ascend950pr/` 5 个 md | **yes（内容层面）** | 知识本身是 Ascend 的，**大部分可复用**，但 API 名要换 |
| 32 | 提示词 overlay | `meta/tilelang-ascend/` **不存在** | no | 今天走共享提示词（`lang_mode.py:26-28`）；新语言同理可用共享 |

### 3.1 这张表最重要的一行

**第 19–23 行：判定机制共 1860 行代码，零 TileLang 引用。**

```
judge.py            280    tilelang:0  triton:0
device.py            83    tilelang:0  triton:0
timing_audit.py     346    tilelang:0  triton:0
window_contract.py 1231    tilelang:0  triton:0
```
[verified: `grep -ci 'tilelang' <each>` 全部为 0；`timing_audit.py` 连 `npu` 都是 0]

它们操作的对象是"一次测量的一组数与一个契约"——`Check`、`device kind`、
`audit`、`window contract`——不是任何 DSL 的语法树。
**这是本项目最宝贵的资产，它语言无关，迁移可以原样带走。**
四轮争用偏差修正（R48/R54/R58/R74）沉淀在这四个文件里，迁移不会损失它们。

### 3.2 但它的**计时基**是 TileLang 的——这个区分是决定性的

第 15/16 行。`fk_dev.device_ms` 调 `tilelang.profiler.msprof.bench_with_msprof`，
而该函数的 NPU 窗口之所以是"host-free per-call device time"，
是因为它用一个**用 TileLang 写的缓存刷新内核**在每次重复前刷 L2：
```python
@tilelang.jit(out_idx=[], target="ascend")
def get_msprof_cache_flush_kernel(...):
    ... TA.copy(ub, Cache[...], l2_cache_ctrl="WTS_FV")
```
[verified: `tilelang/profiler/msprof.py:15-36`]

**这不是一个可选的便利层。** 在它不可用时，`timing.py:220-231` 回落到 `torch.npu.Event`，
而代码自己的注释说那是 **"a host-inclusive window, marked host-bound in the report"**
[verified: `timing.py:220-222`]。

即：**换 DSL 会降级测量质量，而不是仅仅换掉一个 import。**
判定机制本身不受影响（它只看比值），但比值的**噪声底**会变宽——
这正是 R84 那五次门控尝试里最痛的一个变量（§"配对比值才是稳定量"）。

```
可携带（0 改）:  judge.py, device.py, timing_audit.py, window_contract.py,
                precision.py, closed_ref.py, eval 脚手架
不可携带（须重写）: fk_dev.py 的 NPU 计时基 + 一个 TLA 版 cache-flush 内核
```

---

## 4. 注册一个第二 Ascend 语言：7 个门，逐个实测

本文在**不修改任何源文件**的前提下，直接调用各模块函数，确认一个新语言名
（以 `"tla-ascend"` 为例）在每个门上的行为：

| # | 门 | 实测结果 |
|---|---|---|
| 1 | `lang_mode.mode_of("tla-ascend")` | `UnknownLanguage` |
| 2 | `lang_mode.kb_root_name("tla-ascend")` | `UnknownLanguage` |
| 3 | `taskcard.KERNEL_FILE["tla-ascend"]` | `KeyError` |
| 4 | `task.LOOP_LANGUAGE["tla-ascend"]` | `KeyError` |
| 5 | `"tla-ascend" in task.LANGUAGES` | `False` |
| 6 | `task.ARCH_RE` 匹配 `ascend950pr` | `False`（回落到 `^sm\d+[a-z]?$`） |
| 7 | `dsl_source.resolve(language="tla-ascend")` | **不抛**，返回 `"no DSL source to default"` |
| 8 | `leaderboard.infer_language(<TLA 源码>, "tla-ascend")` | **不抛**，返回 `"triton"` |

[verified: 本文 `/tmp/probe_gates.py` 实跑，输出见上]

**门 1–6 是"响的"。门 7–8 是"哑的"**——这是本次测绘最有操作价值的一条：
前六个漏了会立刻崩，后两个漏了会**静默产出一个看起来成功的、标错的 run**。

---

## 5. 工作量估算（保守）

分 MUST / SHOULD。**MUST = 不做则该语言无法启动或判定错误；SHOULD = 能跑但知识/来源不完整。**

### 5.1 MUST-change（6 个文件）

| # | 文件 | 改什么 | 行数（估） | 漏了会怎样 |
|---|---|---|---|---|
| 1 | `loop/lang_mode.py` | 新 mode 常量 + `*_LANGUAGES` 集合 + `KB_ROOT`/`PROVENANCE_SUBDIR`/`PROMPT_OVERLAY_DIR` 各一项 + `mode_of` 一个分支 + `language_problem` 一个分支 | **18–26** | `UnknownLanguage`，无法启动 |
| 2 | `task.py` | `LANGUAGES`/`LOOP_LANGUAGE`/`ARCH_RE`/`ARCH_EXAMPLES` 各一项 | **6–10** | `KeyError` 或 arch 被拒 |
| 3 | `taskcard.py` | `KERNEL_FILE` 一项 | **1** | `KeyError` |
| 4 | `loop/dsl_source.py` | 一个 `_triton_source()`（仿 `:193-228`）+ `resolve` 分支 + 一个 fallback 常量 | **30–45** | **静默空**（门 7） |
| 5 | `loop/agent_policy.py` | `_dsl_source_warnings` 新分支（仿 `:686-701`） | **12–18** | 检查了错键，学生看不到警告 |
| 6 | `leaderboard.py` | `_EXPORT_LANG` 一项 + `infer_language` 分支 | **4–8** | **静默错标成 NVIDIA triton**（门 8） |

**MUST 小计：约 71–108 行，6 个文件。**

### 5.2 SHOULD-change

| # | 项目 | 估 | 说明 |
|---|---|---|---|
| 7 | `hw_kb/triton-ascend/ascend950pr/` 新 KB 树 | **内容工作，非行数** | 现有 5 个 md 的知识多为 Ascend 平台事实（56 核、L2 128 MB、UB 上限、争用），**可迁移**；但 API 名须全换 |
| 8 | 提示词 overlay `meta/<mode>/` | 0 或一套 | 今天 TileLang 也没有（`lang_mode.py:26-28`），共享提示词够用即可 |
| 9 | `tests/test_language_registry.py` 扩展 | **40–60** | 现有 18 个测试里多数按语言参数化，加语言需补断言 |
| 10 | `fk_dev.py` NPU 计时基（§3.2） | **60–120 + 一个 TLA flush 内核** | **这是最大单项**，且不是"注册"范畴 |
| 11 | 候选内核本身 | **数千行** | v33 是 284 行，但那是**已调优终态**；TLA 版需重做 §6 的等价性工作 |

### 5.3 估多少是可辩护的

- **注册面（MUST，1–6 项）**：**约 71–108 行，6 个文件。区间而非单点**：下界是"照抄现有 cuda/tilelang 分支"的机械量；
  上界含 `dsl_source` 的探针逻辑（`:62-190` 约 130 行是探针模板，新写一份可复用大部分）。
- **保守起见把 MUST 说成"6 文件 / 100 行以内（±30%）"**，而不是"大约 90 行"。
- **[unknown]** 5.2 第 10、11 项无法估：**我们的内核**还没有 TLA 版，没测过一次编译/运行，
  也就无法知道 TLA 版的真实工作量。**任何把"迁移成本"当作注册面的说法都低估了它。**

---

## 6. L2 读路径：一个已测得的 1.103× 损失（本文关键一节）

**这一节不重复论证，引用 lead 的 `docs/ascend950pr/tla/l2-policy-field-width.md`（153 行，逐层 file:line）。
我独立复核了其中三处最关键的源码，结论一致。**

### 6.1 复核结果

| 断言 | 我的复核 |
|---|---|
| 硬件读枚举 6 值，3 号位空 | [verified: `/data/pri/cann-9.2.0/cann/asc/include/c_api/utils/enum.h:17-25`] 实读，`NOTALLOC_KEEP=4, NOTALLOC_CLEAN=5, NOTALLOC_DROP=6`，无 3 |
| TLA 向量路径只有 `eviction_policy`，无 `l2_cache_ctrl` | [verified: `tla/dma.py:134-141`] `LoadGM2UBParams` 字段为 `pad_mode/pad_value/left_padding_num/right_padding_num/eviction_policy` |
| 写路径更窄，只有 `atomic_kind` | [verified: `tla/dma.py:143-147`] `StoreUB2GMParams` **确实只有 `atomic_kind`**，连 `eviction_policy` 都没有 |
| MLIR 属性 1/2/3 | [verified: `HIVMAttrs.td:1223-1226`] `EvictNormal=1, EvictFirst=2, EvictLast=3` |
| C++ 模板枚举收窄到 0/1 | [verified: `Template/include/RegBase/DMAUtils.h:39`] `enum class EvictionPolicy : uint32_t { EvictFirst = 0, EvictLast = 1 };` |
| 落到 intrinsic 是 `uint8_t` | [verified: `Template/lib/RegBase/Vector/Copy3D.cpp:144`] `uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);` |
| Cube 路径有宽字段 | [verified: `tla/dma.py:149-155,266-275`] `LoadGM2L1Params.l2_cache_ctrl: int` → `l2_cache_mode` |
| 厂商文档自证 | [verified: `HIVMDMAOps.td:381-387`] 原文含 "the field is wider than that enum … **no eviction *priority* can express that**" |

**结论：在向量 GM→UB 路径上，送给硬件的 cache 控制字节被压到 `{0,1}`，`5` 无处安放。**
且这是**编译器结构性**的，不是 TLA 漏接一个参数——厂商文档自己说"字段比枚举宽"。

### 6.2 这对迁移面的含义：必须写进估算

| 项 | 值 | 证据 |
|---|---|---|
| 读路径 L2 收益 | **1.103×（prefill）** | `core-count-and-l2-breakthrough.md` §2.4；20 配对 quiet 窗口 1.0185→1.1185 |
| 迁移后能否表达 | **当前源码树上不能** | §6.1 |
| 几何改动单独水平 | **~1.01**（FAIL→parity） | 同上，两臂均 `NORMAL_FV` 的对照 |
| **迁移后的 prefill 预期** | **约 1.02，而非 1.19** | [inferred] 1.1883 ÷ 1.103 ≈ 1.077，与"几何 1.01 + 写路径 1.02×"量级一致 |

**因此：任何不含"−1.103×"的迁移成本估算都不可用。** 这不是一个实现细节，
它是 v33 相对 v23 的两处修复之一，且是较便宜的那一处。

### 6.3 唯一能把 [inferred] 变成证据的实验 —— 未做

`EvictFirst` 是否可作为 `NOTALLOC_CLEAN` 的功能替代，只能实测：
128 MB 工作集撞 128 MB L2 的 prefill 窗口，`EvictFirst` vs 硬件默认，看能否复现 ~1.10×。

**该实验：NOT DONE。** 理由不是时间，也不是"TLA 不可用"（它可用，见文首更正），而是：**该窄策略在源码结构上就不存在可测的替代量**，加上我们的内核没有 TLA 版可跑
（§0 已证）。没有 `tla` 就无法编译该内核，无法在设备上跑。**本文不模拟、不外推。**

厂商文档 §3 那句 "no eviction *priority* can express that" 直接否定这个等价
[verified: `HIVMDMAOps.td:381-387`]，所以**默认假设是不可替代**，除非实验翻案。
[inferred：`EvictFirst` 与 `NOTALLOC_CLEAN` 不等价；实验是唯一上诉渠道]

---

## 7. 本文没有证明的

- **没有跑过任何 TLA 代码。** 本机无 `tla` 包（§0）。表中 "Triton-based 语言需要什么" 一列
  是**读 TLA 源码得出的**，不是编译验证过的。
- **没有验证 `EvictFirst` 的带宽效果**（§6.3）。这是整个调研最大的单一未知。
- **没有写代码**，所以 §5 的行数是**读代码后的估计**，不是实现后的实测。区间已尽量放宽。
- **没有评估"借用 `LoadGM2L1Params.l2_cache_ctrl` 到向量路径"** 的可行性——那是设计问题。
- **没有核 `StoreUB2GMParams` 之外是否另有写策略通路**；就源码所见没有
  [verified: `tla/dma.py:143-147`]，但 [unknown：后端是否存在未在 Python 层暴露的属性]。
- **没有测 `torch.npu.Event` 回落对配对比值的实际噪声增量**（§3.2 的影响是定性的：
  基从 host-free 变成 host-inclusive，代码自己标注 `host-bound`）。
- **没有查 `triton_ascend` 之外的发行渠道**是否有含 `tla` 的 wheel；6 个 wheel 全查过，均为 0 文件。

---

## 8. 三个事实，供决策使用（不作结论）

1. **这是 DSL 家族更换，不是版本升级。** TileLang → Triton + `tla.*`。
   TLA 自述"does not reuse the existing `al`/`bl` general interfaces (different programming paradigm)"
   [verified: `tla/__init__.py:23-25`]。
2. **TLA 已在本机可用**（专用 conda env + 容器 root，门已 PASS）[更正：初版此处说"没有可安装的 TLA"，是错的]。
   但**我们的内核没有 TLA 版**，所以"能不能迁"仍然未知。
3. **迁移会确定性地丢掉 1.103×**，除非 §6.3 的实验翻案；而厂商文档已经用同义措辞说明
   "任何逐出优先级都表达不了不分配"。**注册面很小（6 文件 / 约 100 行），
   但注册面不是迁移成本的主要部分**——`fk_dev` 的 NPU 计时基和内核重写才是。

---

## 9. 复现命令

```bash
cd /home/s00653124/workspace/ForgeKernel

# 语言注册门（7 个，逐个实测）
/home/s00653124/.conda/envs/s00653124_tilelang/bin/python /tmp/probe_gates.py

# 判定机制与 TileLang 无关（四个文件应全为 0）
for f in judge.py device.py timing_audit.py window_contract.py; do
  echo -n "src/forgekernel/$f tilelang:"; grep -ci tilelang src/forgekernel/$f; done

# 注册表面
grep -n 'LANGUAGES\|LOOP_LANGUAGE\|ARCH_RE' src/forgekernel/task.py
grep -n 'KERNEL_FILE' src/forgekernel/taskcard.py
grep -n 'CUTEDSL_LANGUAGES\|TILELANG_LANGUAGES\|MODE_' src/forgekernel/loop/lang_mode.py

# "triton" 今天的语义
grep -n '"triton"' src/forgekernel/loop/lang_mode.py src/forgekernel/leaderboard.py

# NPU 计时基的 TileLang 依赖
grep -n 'bench_with_msprof\|msprof\|npu_events' \
  src/forgekernel/card_template/eval/fk_dev.py src/forgekernel/card_template/eval/timing.py

# TLA 是否存在可安装产物（应无输出）
for w in $(find /home/s00653124 /data/pri -name "triton_ascend*.whl" 2>/dev/null); do
  echo -n "$w "; unzip -l "$w" | grep -c "extra/tla/"; done
```
