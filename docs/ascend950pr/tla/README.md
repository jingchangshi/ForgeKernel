# ForgeKernel × TLA：迁移决策文档

**作者**：implementer-v33（task-11）。**日期**：2026-11-03。
**读者**：提出"考虑把我们在 Ascend950PR 上跑通的路径从 tilelang 迁到 TLA；TLA adapter 已经在 AgentGates 里了"的操作者。
**性质**：决策文档。本文给建议，不给验收结论。

> **`adapter-audit.md` 的作者是 lead**，不是 task-8 原调研 agent（后者运行约 40 分钟后中断、
> 未产出文件，由 lead 从一手源码重写）。**因此它是"brief 的源码硬化版"，不是对 lead 先前简报的
> 独立佐证**——引用时不应把它当作第二个信源。其"原 agent 未交付的两处更正"不在该文件内。
> 第 5、6 节取自该文件，其中 `GATE_TEST`、`--candidate`、`parse_argv`、`TOOLCHAIN_KEYS`、
> `README.md:23-31`、`RECORDED, NOT GATED`、`evaluator.py:6-13` 及行数我已独立复核。

标签沿用全项目约定：**[verified: 路径:行]** / **[measured]** / **[source-only]** / **[inferred]** / **[unknown]**。
判定词表只用 **EXCEED / MATCH / FAIL / RETEST / NOT DONE**，不使用"通过"类措辞。

**四份同级文档**（本文引用，不重述）：

| 文档 | 回答什么 |
|---|---|
| [`adapter-audit.md`](adapter-audit.md) | tla-npu adapter 到底是什么、证明了什么、做不到什么 |
| [`migration-surface.md`](migration-surface.md) | TileLang 在 ForgeKernel 里嵌得多深，MUST/SHOULD 改动清单 |
| [`tla-capability.md`](tla-capability.md) | 逐构造映射，探测结果与 NOT DONE 项 |
| [`l2-policy-field-width.md`](l2-policy-field-width.md) | 1.103× 损失风险的逐层证据链 |

---

## 0. 一句话结论

**建议选 D（STAGED）。** 理由不是 TLA 不好，而是：**C（直接迁移 rmsnorm）会拿我们唯一一个已认证的
性能结论去换一个尚未测过的实现**。

**先读这一句，它约束全文：本机没有可安装的 TLA，所以本文所有关于 TLA 能力的陈述都是
[source-only]，不是实测——没有人在本卡上跑过 TLA。**

C 的代价要分两层写，混在一起会把它写轻：

| | 值 | 性质 |
|---|---|---|
| **下界**（最好情况） | prefill 1.1883 → **1.0170 [1.0144, 1.0179]** | 这是同一 v33 内核在同 body、56 块、`NORMAL_FV` 读策略下的**实测值**——即"迁移后只剩 L2 这一处损失、其余完全复现"时的水平 |
| **期望值** | **不高于上表，且很可能更低** | §7 的 staging 重搜是**未量化的额外成本**，内核重写本身还有成本 |

**1.0170 是损失的地板，不是迁移的期望值。** 把它当作 C 的代价会让 **C 显得比实际更好**——
因为它假设了一次"除 L2 外完美移植"，而这正是我们**无法证明**的部分（§2、§7）。
另外它是 [inferred] 映射：1.0170 实测于 TileLang，不是 TLA；TLA 在本机跑不起来，没人测过 TLA 版。

**D 也不是免费的，这一点必须说清楚。** D 的第一步（让一个含 `tla` 的 wheel 存在）**是 B/C/D 三个
选项共同的前置条件**，这是它的结构优势；但按本文 §2，这一步需要**厂商构建，或在本机自行编译
整个 `triton-ascend`**，两者都不在当前范围内。**所以 D 是一个"以该工作获得授权为前提"的建议**——
本文件不能自己解除那个前提。谁来做这件事需要在 D 被采纳时一并指定。

**TLA 不是坏东西，本机今天也不是能用的东西。** 这两句都要说，且都不等同于对方。

---

## 1. 先纠正问题本身：TLA 不是 TileLang 的扩展

这是全文最早要说的一件事，因为操作者的措辞（"迁移到 TLA"）暗示了一个可能不成立的前提：
**把 TLA 当作 TileLang 的一个新版本或一个新扩展。它不是。**

| | 事实 |
|---|---|
| TLA 是什么 | 厂商 `triton_ascend` wheel 给 **TRITON** 加的一个命名空间 `triton.language.extra.tla` |
| 它扩展的不是 | TileLang。与 `triton.language.extra.al` / `.bl` 是**并列的独立命名空间** |
| TLA 自述 | "Independent `tla.*` programming namespace for Ascend NPU kernels, migrated from the TLA repo's SIMD VF additions. It does **not** reuse the existing `al`/`bl` general interfaces (**different programming paradigm**)." [verified: `third_party/ascend/language/tla/__init__.py:23-25`，与 [`adapter-audit.md`](adapter-audit.md) §1 引用同一行同一段，措辞一致] |

导出面 [verified: 同上 `:27-32`]：`alloc_buffer` / `alloc`（片上 buffer + versions）、`copy`（GM↔buffer）、
`view`、`scope`（`tla.scope.simd()` / `tla.scope.cube()`）、`simd`、`annotate_buffer_versions`。

**所以这是 DSL 家族的更换**：TileLang（`T.*` 跑在 TVM/TIR 上）→ Triton + `tla.*`。
不是同族版本升级，不是换一个 import，也不是把 `T.copy` 改名成 `tla.copy`。

为什么这句话重要：同族升级的迁移风险主要是"改 API 名"，而 DSL 家族更换的风险主要在
**"两个 DSL 到同一硬件的 codegen 路径不同"**。本文第 3 节的全部内容，就是那个差异的实例——
同一个硬件字段，一个 DSL reach 得到、另一个 reach 不到。

---

## 2. 本机没有可安装的 TLA

这不是"配置错了"，是**缺一个构建产物**。

| 检查 | 结果 |
|---|---|
| 已装 `triton` | `triton_ascend 3.6.0+dev20260816011301`（2026-08-16 构建） |
| `from triton.language.extra import tla` | **ImportError** [measured] |
| 已装 `language/extra/` 内容 | `cann, cuda, hip, kernels, libdevice.py`——无 `tla` |
| 本机 **6 个** `triton_ascend` wheel 的 RECORD | `language/extra/tla/` **全部 0 文件** [measured，6/6] |
| `tla` 在哪里存在 | 仅源码：`triton-ascend/third_party/ascend/language/tla/`（19 个 .py，3977 行） |
| TLA 首次落地 | 2026-10-24（`9e6fdfcfc`）——**晚于本机所有 wheel** |
| adapter README 钉的 wheel | `dev20261030+git0c9d99ca`，**本机不存在** |

我把"把源码树挂进去"这条没人试过的路也走了一遍 [measured，`probes/probe_aliased_import.py`]：
Python 层可以导入、`@tla.jit` 可以装饰，然后编译在
`compiler.py:232  ascend_ir.tla_builder(context, arch)` 处死掉——
`AttributeError: module 'triton._C.libtriton.ascend.ir' has no attribute 'tla_builder'`。
直接枚举确认：源码引用的 24 个 `create_tla_*` 与 26 个 `create_simd_*` 原生 builder，
在已装的 `libtriton` 里**一个都没有**。

**含义**：手工把包拷进去也编译不了任何东西。任何迁移计划的第 0 步都是**让一个含 `tla` 的
可安装产物存在**——而这需要厂商构建或本机自行编译整个 `triton-ascend`，两者都不在当前范围内。

---

## 3. L2 账本：迁移确定性地丢掉 1.103×

本节只放数字与结论，逐层证据见 [`l2-policy-field-width.md`](l2-policy-field-width.md) 与
[`tla-capability.md`](tla-capability.md) §3。

### 3.1 我们赢在哪（两个独立量）

| 改动 | 性质 | 独立收益 |
|---|---|---|
| `N_CORES` 64 → 56 | host 侧整数 | 主项（几何） |
| 读路径 `NORMAL_FV` → `NOTALLOC_CLEAN` | 一个 `l2_cache_ctrl` 参数 | **1.103×** [measured，20 配对 quiet] |

### 3.2 迁移后能否表达

**当前源码树上不能** [source-only，high]。收窄链：TLA 向量读只有
`eviction_policy ∈ {EvictNormal, EvictFirst, EvictLast}`（`dma.py:134-141`, `:21-24`）→
MLIR 属性 1/2/3（`HIVMAttrs.td:1223-1231`）→ C++ 模板枚举
`{EvictFirst=0, EvictLast=1}`（`Template/include/RegBase/DMAUtils.h:39`）→
`uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy)`（`Copy3D.cpp:144`）。
**字节 5 无处安放。**

### 3.3 账本

| 项 | 值 |
|---|---|
| 读路径 L2 收益 | **1.103×**（prefill，CLEAN 1.1185 vs FV 1.0170） |
| 迁移后能否表达 | **当前源码树上不能** |
| 几何改动单独水平 | ~1.01 |
| **迁移后 prefill 下界**（最好情况） | **1.0170**，即"只丢 L2、其余完美复现"时 |
| **迁移后 prefill 期望值** | **不高于 1.0170，很可能更低**（§7 staging 重搜未量化）[inferred] |

关于那一行 [inferred] 的准确含义，不要让它比证据更硬：

- **1.0170 是实测的**，但它是 **TileLang + NORMAL_FV** 的实测——不是 TLA 的实测。
- **"TLA 版会落在 1.02" 是推理**：推理链是"TLA 的向量读路径拿不到 CLEAN（源码五层，§3.2）→
  因此等价于 NORMAL_FV → 因此落在 1.0170 附近"。**这条链的设备端从未被验证**，因为本机没有 TLA。
- 若 TLA 在我们不认识的别处有补偿（例如某种我们没识别出的写路径或调度优势），实际数字可能不同。
  **本轮没有找到这样的补偿**，但"没找到"不是"不存在"。

**因此：任何不含"−1.103×"的迁移收益估算都不可用；同样，任何把"1.02"当作 TLA 实测值的引用都是错的。**

### 3.4 这件事的准确形态（不要让它变成"TLA 不行"）

准确的说法是一条关于**字段在哪里被暴露**的事实：

- **TileLang 根本不过 HIVM。** `EmitGmToUbufCopy_`（`codegen_ascend.cc:1417-1433`）直接发射 CANN C-API
  并把参数强转到**宽硬件枚举** `asc_load_l2_cache_mode`，所以 `5` 原样通过。
  发射出的 v33 设备源码实测三个 DMA 站点：W 读 `=0`、**X 读 `=5`（我们的 1.103×）**、Y 写 `=0`。
- **TLA 走 HIVM**，而 HIVM 那一层的属性类型就是窄的（`DMAUtils.h:39` 的 C++ 枚举 +
  `HIVMToStandard.cpp:1233-1235` 那句 "Only support EvictFirst & EvictLast Policy for now"）。

所以这是**两条 codegen 路径到达同一硬件字段的方式不同**，不是"TLA 团队漏接了一个参数"。
厂商自己的文档用几乎相同的措辞说明了原因：

> "…keeping a streamed-once operand out of L2 entirely is what stops it evicting the operand
> that is re-read, and **no eviction *priority* can express that.**"
> [verified: `HIVMDMAOps.td:381-387`]

**什么会让本节的结论翻案**（这是 D 选项存在的理由）：

1. 上游把 `LoadOp` 的 cache 属性从窄枚举改成宽字段，**或**让向量拷贝绕开 HIVM 的窄枚举
   （即 `DMAUtils.h:39` / `Copy1D/2D/3D.cpp` 那条通路改宽）。这是**编译器侧**改动，不是 wheel 侧。
2. 实测证明某个窄策略能在我们的窗口下复现 ~1.10×。**注意这条目前无从做**，见 §3.5。

> **我们说清楚，读者才不会误会**：本机**没有**可安装的 TLA，所以这个 1.103× 损失是
> **[source-only] 结论，不是设备实测**。它在源码上是硬的（五层都核过），但它没有被设备验证过。

### 3.5 那个实验：NOT DONE，且是无可做，不是没做

lead 原先提议"`EvictFirst` vs 硬件默认，看能否复现 ~1.10×"。**该实验 NOT DONE，且我没有模拟它。**
两条已核实的理由：

1. 本机没有 TLA 产物（§2），内核编译不出来，没有东西可计时。
2. **`EvictFirst` 就是默认值。** `ConvertToHIVMOp.cpp:312-313` 的原文是
   `// Default set to evict_first (Ascend950 / Triton cache hint).`，每个 load 无条件写 EvictFirst；
   `HIVMToStandard.cpp:68-71` 的映射表只有 `{EvictFirst->0, EvictLast->1}`，
   `:1228-1236` 对其它取值告警并强制回落。
   所以"EvictFirst vs 默认"是**默认对默认**，测到的是噪声。
   真正有意义的对照（默认 vs 字节 5）恰恰是 TLA 表达不了的。

**结论：这一行从 [inferred] 变成结构性不可测。** 没有上诉渠道，因此默认假设必须是"不可替代"。

---

## 4. 什么是免费带走的（这是支持迁移便宜的主要论据）

我们真正的资产不是 TileLang，是**配对同窗口判定机制**。逐组件测绘见
[`migration-surface.md`](migration-surface.md) §3。

| asset | 规模 | TileLang 引用 |
|---|---|---|
| `judge.py` | 280 行 | **0** |
| `device.py` | 83 行 | **0** |
| `timing_audit.py` | 346 行 | **0** |
| `window_contract.py` | 1231 行 | **0** |
| `precision.py` | 421 行 | **0** |
| `closed_ref.py` | 797 行 | **0** |

[verified：我本轮逐个复跑 `grep -ic`，六个文件对 `tilelang` 均**为 0**。
`closed_ref.py` 里那 3 处是 `triton`（`closed_ref.py:47-51` 讲的是候选自己的
`kernel` 与参照文件 `kernel.py` 名字不碰撞，与语言无关），不是 TileLang 提及——
上一版把 `triton` 的计数误记成了 TileLang 提及，已更正为 0。]

**合计约 3160 行零 TileLang 引用的判定代码。** 它们操作的对象是"一次测量的一组数与一个契约"，
不是任何 DSL 的语法树。R48/R54/R58/R74 四轮争用偏差修正沉淀在这几个文件里，
**迁移不会损失它们**。

同样免费带走的（`migration-surface.md` 第 3 节）：

- 门控协议（≥9 且 ≥70% quiet 窗口、判据事先声明、单形状分开跑）
- 配对同窗口的接受纪律（R74：两侧绝对时长同比漂移 ±20%，**配对比值才是稳定量**）
- ledger / manifest 实践
- KB 里大部分 Ascend **平台**事实（56 核、L2 128 MB、UB 上限、争用形态）——知识可迁移，API 名要换

**这是"迁移成本比听起来低"的真实依据**，应当由我们先说，而不是等读者替我们算。

### 4.1 但计时基不是免费的（这个区分是决定性的）

`fk_dev.device_ms` 调 `tilelang.profiler.msprof.bench_with_msprof`，而它之所以给出
"host-free per-call device time"，是因为**用一个 TileLang 写的缓存刷新内核**在每次重复前刷 L2
[verified: `tilelang/profiler/msprof.py:15-36`，该内核用 `TA.SimtVF` + `l2_cache_ctrl="WTS_FV"` 写成]。

该内核不可用时，`timing.py:220-231` 回落到 `torch.npu.Event`，而代码自己标注那是
**"a host-inclusive window, marked host-bound in the report"**。

**即：换 DSL 会降级测量质量，而不只是换掉一个 import。** 判定机制本身不受影响（它只看比值），
但比值的**噪声底**会变宽——而这正是 R84 五次门控尝试里最痛的变量。

---

## 5. 门需要什么：tla-npu 今天门的是一个**固定文件**

这条最容易误判："TLA adapter 已经在 AgentGates 里了"是真的，**但它不门我们的内核**。
逐条落点与"抓住/抓不住"的完整映射见 [`adapter-audit.md`](adapter-audit.md) §2–§3，此处只取决策相关的部分。

### 5.1 它今天不能接受候选内核，而且不是配置问题

| 证据 | 内容 |
|---|---|
| 测试文件名 | `GATE_TEST = 'test_tla.py'` 是**模块级常量** [verified: `tla_gate.py:124`]，注释说明它 "fixed in this digest-pinned file, and never read from the environment" [verified: `:117-124`] |
| 测试**目录** | 来自 `TLA_NPU_TEST_DIR`，必填**绝对路径**，且必须是 `test_tla.py` 所在目录 [verified: `:349-366`] |
| CLI 参数 | `parse_argv()` **只**接受可选 `--timeout SECONDS`，其它任何参数一律 `BridgeError` [verified: `:565-573`]；`main()` 另只特判 `--check-config` [verified: `:1046`] |
| 配置键 | `TOOLCHAIN_KEYS` 是**封闭元组**，未知键**报错** [verified: `:140-166`]，其中没有任何 CANDIDATE 类键 |

**所以最小改造不是"加一个 `--candidate` flag，是改这道门的契约。**
理由：门要求被测文件输出两个 marker，而 `CAST_BACK_MARK` / `TRITON_MARK`
是 cast_back 语义专属的；换成我们的 rmsnorm 后，`castBackOk` / `tritonCastBackOk`
这两个检查**就没有意义了**，必须连同 `REQUIRED` 一起参数化 [verified: `:991-995`]。
**改造范围是契约重写，不是开一个开关。** [inferred：未实施，不给行数点估计]

### 5.2 一个具体机制：它拒绝了一次"免费通过"

这是审计里最值得记的一条正面证据，且是**机制**而非评价。

`CAST_BACK_MARK` 是 `TRITON_MARK` 的**字面子串**，所以朴素的 `in` 判断会让
**纯 Triton 参照路径**的输出同时为 TLA 路径记分——即一个根本没走 TLA 的运行会被判两项都过。
代码用**负向后查**绕开 [verified: `:921-922`]：

```python
cast_back = isinstance(stdout, str) and bool(re.search(r'(?<!TRITON_)' + CAST_BACK_MARK, stdout))
```

作者在注释里写明了原因 [verified: `:916-919`]："a plain `in` test would credit the TLA route
for output that only the plain-Triton reference produced."

**这条对迁移评估的意义**：它证明这道门在**主动堵**一个会让我们高估 TLA 覆盖面的漏洞。
把它描述成粗糙，是我方错误。

### 5.3 它的克制程度与我们的纪律相当，且是独立到达的

| 事实 | 证据 |
|---|---|
| README 主动声明通过结论**不意味着什么**："does **not** mean the whole TLA surface is covered, and **it is not a compiler test suite**" | [verified: `README.md:23-31`] |
| **缺失数据一律为假，从不假定**："Missing data is false, never assumed." | [verified: `observations()` docstring, `:906`] |
| **拒绝断言 by-construction 的值**：`visibleDeviceEcho` 是 runner 自己 export 后读回的；`deviceUuid` 因本机两张卡报同一 UUID 而**记录不门控** | [verified: `RECORDED, NOT GATED` 整表, `:949-975`] |
| **拒绝不可满足的要求**：`.o` 被计数但不要求，因为后端不产出它，"requiring it would be unsatisfiable rather than strict" | [verified: `:935-937`] |
| 设备占用**报告不强制**，理由是 "on this class of machine a card is shared"，卡被占给 `BLOCKED_ENV` | [verified: `README.md:65-70`] |

**它甚至独立收敛到了我们的同一条结论。** `PLAN-perf-workflow.md` §3 **主动拒绝**把性能阈值做成门
[verified: `:37-78`]，而不是做不到：`manifest.py` 不对 `observationChecks` 做类型校验，
而 `evaluator.py:6` 的 `equal()` 先做 `type(actual) is not type(expected)` 再 `==`
[verified: `:6-13`]，所以 `{"speedupAtLeast": 1.2}` **能加载**然后**永远失败**——float 永不等于 JSON 的 `true`。
一句话就能加一个比较算子，**作者明确拒绝加**，理由是那会把中位数变成确定性判定，而本卡跨进程漂移 25.8%。

**这与我们 R84 用五轮 RETEST 换来的"记录而不门控"是同一条纪律，两个项目独立到达。**
这一点记在 adapter 账上。

### 5.4 plan 与 code 的差距（`PLAN-perf-workflow.md` 首行自述）

> "Status: proposal, reviewed. **Nothing in this document is implemented yet.**"
> [verified: `:3`]

| 项 | 状态 |
|---|---|
| 一个固定的端到端设备回归测试 | **已实现** |
| 接收 agent 自己写的内核 | **未实现**（plan §2 自述这是 "the current adapter does not fit" 之处） |
| 逐 kernel 报告结果（而非单次 pass/fail） | **未实现** |
| 性能记录进 receipt | **部分**：TLA 实现身份/版本已记录；**时长未记录** |
| 性能阈值门 | **主动拒绝**（§5.3） |
| 观察经 MCP 到达 agent | **未实现**（plan 说 in scope） |

---

## 6. 两个 adapter 家族是两件不同的事

`agent_gates/adapters/` 下现存 5 个 adapter，其中与本决策相关的是两个：

| adapter | 规模 | 干什么 |
|---|---|---|
| `tla-npu` | `tla_gate.py` 1125 行、11 项 check | 对**固定** `test_tla.py` 的 **capability 设备回归门** |
| `forgekernel-ascend-npu` | `ACCEPTANCE.md` 598 行、`accept_rmsnorm.py` 2044 行、`fk_gate.py` 1078 行 | **门我们今天这条已认证路径的那一个** |

[verified：`wc -l` 实测；与 `final-status.md` §6 的复现命令一致]

**所以"TLA adapter 已经在 AgentGates 里了"这句话：真，但今天在干那个活的是另一个 adapter。**
现有集成不是"把已有能力打开"，是**把验收职责从 `forgekernel-ascend-npu` 迁移或复制到 `tla-npu`**，
而后者今天连候选内核都收不了（§5.1）。换掉它等于把 §4 里那些免费带走的东西**在 adapter 层重做一遍**。

### 6.1 这修正了对"迁移成本"的上界估计

[`migration-surface.md`](migration-surface.md) 给的注册面是 **6 文件 / 约 100 行（±30%）**。
那是**语言注册**的成本，不是迁移成本。迁移成本还要加：

| 项 | 量级 | 可估？ |
|---|---|---|
| 语言注册（6 文件） | 71–108 行 | 可估 |
| `fk_dev` NPU 计时基 + TLA 版 cache-flush 内核 | 60–120 行 + 一个新内核 | 可估但要设备 |
| **tla-npu 门改为接受候选** | **契约重写**（§5.1），非加 flag | 可估但要读完全文 |
| **内核本身重写** | v33 的 284 行是**已调优终态**，不是起点 | **不可估** |
| **staging 重搜**（见 §7） | 未知 | **不可估** |

**任何把"迁移成本"当成注册面的说法都严重低估了它。**

---

## 7. 第二根承重柱的不确定性：staging 需要重搜

L2 之后最要命的一行。**这一行是"未量化的成本"，不是"已解决的对应项"。**

| 项 | 结论 | 标签 |
|---|---|---|
| 缓冲版本标注 `annotate_buffer_versions` / `alloc_buffer(versions=)` | **存在且是一等公民**（`aux_ops.py:57-74` 发同一个 `hivm.multi_buffer` 注解；`alloc.py:41`；门内 `test_tla.py:43-45` 三缓冲全用 `versions=2`） | [source-only, high] |
| `T.Pipelined(n, num_stages=k)` | **无对应物**——19 个 tla 文件里 `Pipelined` / `num_stages` **零出现** | [source-only, medium] |
| `tl.range(..., num_stages=)` | 语言里有，但 TLA/HIVM 是否把它当作软件流水指令**未知** | [unknown] |
| 我们的 `(4,2)` 能否直接搬 | **不能**。它是 `stages,yver = (4,2) if rows>1 else (2,1)`，是**实测出来的缓冲版本数** | [inferred, high] |

**含义：如果 TLA 的 staging 只能用 `versions=` 表达，那我们是在手工重建 `T.Pipelined` 现在
替我们推导的那套调度，而实测最优值必须在 DSL 上重搜。** 这是一个**未量化的成本**。

为什么它比 L2 更麻烦：L2 那一行的损失**有数字**（1.103×，可写进账本）；
staging 这一行的损失**没有数字**，因为没有任何 TLA 内核在本机跑过，
连一次编译都没成功（§2）。**一个没有数字的成本项，不能用"应该差不多"来处置。**

---

## 8. 一个与 TLA 无关的风险（任何迁移都必须付）

**我们的 1.1883 / 1.7144 只对 8192×4096 与 64×4096 两个形状认证。**
任何迁移都必须**重新认证**，而且：

- **换语言本身不改善任何一个已测数字。** 全新写法在同等条件下最好的结果是 MATCH，
  不是 EXCEED——TLA 不会让 HBM 带宽变大，也不会让 128 MB L2 变大。
- 换语言**唯一可能**的性能收益来自"TLA 是否暴露了 TileLang 没暴露的杠杆"。
  就本轮调研所及，**没有找到这样的杠杆**；反过来找到了一个 TileLang 有、TLA 没有的（§3）。
- 认证成本是确定的（两个形状分别跑、≥60 配对、quiet ≥70%、child-timeout 900–2700 s），
  而迁移收益是**未识别**的。

**结论：把一个已认证的 EXCEED 换成一个未认证的同构实现，在账面上是纯风险。**
这正是 A 与 D 的立足点，也是 C 必须自证的部分。

---

## 9. 四个选项

### A. STAY：只用 tilelang，不碰 TLA

| | |
|---|---|
| 得到什么 | 零成本。保住已认证的 1.1883 / 1.7144。R84 的 L2 杠杆完好。 |
| 付什么 | 放弃 TLA 的表面能力（`cube`/`sync`/`compensated_rstd` 等），以及厂商主推方向的早期经验。 |
| 什么会证伪它 | 出现一个**只有** TLA 能表达、且对我们有实质收益的杠杆；或 TileLang-Ascend 停止维护。 |

A 是**诚实的最低成本选项**，但它有一个真缺陷：它把"TLA 到底能不能用"这个不确定性**永久留着**。
下一轮有人再问同一个问题时，答案仍然是"不知道，本机装不上"。

### B. PARALLEL：注册第二个 Ascend 语言，两条路都留，同任务对比

| | |
|---|---|
| 得到什么 | 可以在**同一个 task** 上并排比两个 DSL，不必先断言谁赢。这正是 `lang_mode.py` 的设计所预告的形态（它的 docstring 明确担心"注册错语言会静默读到错 KB"，说明多语言是预期用法）[verified: `lang_mode.py:32-35`]。 |
| 付什么 | 注册面 6 文件 / 约 100 行；**两个静默失败点**（§9.1）必须一起修；KB 树、`fk_dev` 计时基、TLA flush 内核；以及**长期维护两条路径**的持续成本。 |
| 什么会证伪它 | 如果 TLA 侧长期没有可用产物（§2），B 会退化成"养一个跑不起来的语言"。 |

**B 的前置条件与 D 相同**（必须先有含 `tla` 的 wheel），但它比 D 多做了一件现在还不该做的事：
在**还没测过任何 TLA 内核**的情况下，先把两套 plumbing 建起来。

#### 9.1 B 的两个静默失败点（不是 B 的问题，是必须先修的）

`migration-surface.md` §4 实测 8 个门，**6 个会响，2 个是哑的**：

| 门 | 行为 |
|---|---|
| `dsl_source.resolve(language="tla-ascend")` | **不抛**，返回 `notes=("no DSL source to default",)`——学生拿不到任何 API 源码，run 不报错 |
| `leaderboard.infer_language(<TLA 源码>, "tla-ascend")` | **不抛**，返回 `"triton"`——一个 Ascend 内核会被**导出成 NVIDIA triton 解**（`dependencies=["triton >= 2.3"]`） |

[verified: `migration-surface.md` §2.2b / §2.3，我复核了 `lang_mode.py` 与 `leaderboard.py:1363-1386`]

**这两处是本决策里最容易被"已经能跑了"掩盖的风险**：前六个门漏了会立刻崩，后两个漏了会
**静默产出一个看起来成功、但标错的 run**。且 `"triton"` 这个字符串已被 NVIDIA 语义占用，
`mode_of("triton") == "cutedsl"`、KB 指向 `hw_kb/cutedsl/`——**放行、不报错、指向错的知识库**。

### C. MIGRATE：把 rmsnorm 迁到 TLA，退役 tilelang

| | |
|---|---|
| 得到什么 | 单一 DSL，统一到厂商主推方向。**未识别出任何性能收益。** |
| 付什么 | §3 的 **−1.103×**（prefill 最好情况 1.1883 → 1.0170，期望值更低）；§7 的 staging 重搜（**未量化**）；§6.1 的全部工程项；**并且要重新认证两个形状**。 |
| 什么会证伪它 | 它需要先自证收益，而目前**找不到收益**。反过来说：如果 §3.4 的两条翻案任一成立，C 才可辩护。 |

**C 是唯一一个在账面上确定亏损的选项。** 本文不建议 C，理由不是"TLA 差"，是
"这个交换在账本上通不过"——我们拿一个 EXCEED 换一个未测实现，且已知会丢掉 1.103×。

### D. STAGED：先用一个便宜内核证明 TLA，再决定（**推荐**）

| | |
|---|---|
| 得到什么 | 先解除**共同前置条件**（有一个含 `tla` 的 wheel），再用最小代价把最大的不确定性变成数字。 |
| 付什么 | 一次构建/获取产物的努力；一个便宜内核的编写与门控；时间。 |
| 什么会证伪它 | 若便宜内核上 TLA 也拿不到 MATCH，则 C 被否决、回到 A，而**这次是有证据的 A**。 |

**D 的第一步不是"试点 rmsnorm"，是让产物存在。** 这一条值得强调：A/B/C/D 里 **B、C、D 三个**
都绕不过它，所以做它**不会因为后续选错而白费**。这是 D 相对 A 的结构性优势——
A 什么都不做，D 花同样的钱解除一个共同前置条件。

#### 9.2 那个"便宜内核"该是什么

**不该是 rmsnorm**——它是我们唯一的已认证结果，拿它当试验品等于把认证押上去。

建议的候选（按偏好排序）：

1. **一个纯带宽的 streaming copy / cast 内核**（fp16↔bf16 或 fp8→bf16，`rows×4096`，单阶段，无归约）。
   为什么合适：
   - 它**只测 MTE2/MTE3 与 L2 行为**，正是 §3 争议所在的那条通路；
   - 它**不含归约**，所以绕开了 `vcadd`(6 级) vs `vreduce_sum` 的等价性未知；
   - 它**不含 staging 深度选择**，所以绕开了 §7 的未量化项；
   - 它可以在 **128 MB 工作集**下跑，直接对上我们的 L2 窗口条件；
   - 它足够简单，TLA 若连它都写不顺，C 就不必谈了。
2. 备选：**`agent_tla/test_tla.py` 那个 cast_back 内核**本身。它已经存在、已经被 tla-npu 门过、
   并且用了 `versions=2` 与 `scope.simd()`，是一个现成的真实负载。缺点是它带 per-group scale，
   比纯 copy 复杂一点。

**这个内核要回答的问题，按优先级**：

| # | 问题 | 若答案为否 |
|---|---|---|
| 1 | 能否在**本机**编译并跑起来？ | C 立即否决，回 A（有证据） |
| 2 | 它的**计时基**能否达到 host-free（`fk_dev` 的 msprof + flush 内核可重建）？ | 噪声底变宽，比值的可信度下降 |
| 3 | 在 128 MB 窗口下，它的**带宽/时长**能否 MATCH 同几何的 TileLang 版？ | 出现无法解释的性能缺口 → C 风险陡增 |
| 4 | 用 `versions=` 手工调 staging 与 TileLang 的 `T.Pipelined` 差距多大？ | **这正是 §7 的未量化项，用便宜内核把它量化** |

**问题 4 是 D 的核心价值**：它把本文唯一一个"没有数字的成本项"变成有数字的。
这比在 rmsnorm 上重做一遍便宜得多。

---

## 10. 建议与翻案条件

### 10.1 建议：选 D（STAGED）

一句话：**先把"含 `tla` 的产物存在"这个共同前置条件解除，再用一个不含归约、不含 staging 选择的
纯带宽内核把最要命的不确定性变成数字；在这两件事完成之前，不动 rmsnorm。**

三条支撑：

1. **C 在账本上确定亏。** 最好情况是 prefill 1.1883 → 1.0170（§3.3 的下界，实测于
   TileLang + `NORMAL_FV`），**期望值不高于它**，因为 §7 的 staging 重搜是未量化成本。
   我们唯一的已认证 EXCEED 会被换成一个未认证的同构实现，而**收益至今未被识别**。
2. **A 太省，省掉的是信息。** A 完全站得住（保住认证结果），但它不解除任何不确定性，
   下一轮这个问题会被重新问一遍。D 花的是同一笔前置成本，却带回数字。
3. **D 的第一步对 B/C/D 三者都是必要的。** 它不是"为迁移铺路"，
   是"先让这件事具备可评估性"。（§2：本机没有任何含 `tla` 的 wheel。）

**但 D 的前置工作本身是有代价的，本文必须点明，否则 D 会显得比实际更可行：**

| D 的第一步 | 现实 |
|---|---|
| 让一个含 `tla` 的 wheel 存在 | 需要**厂商构建**，或**在本机自行编译整个 `triton-ascend`** |
| 两种方式是否在当前范围内 | **都不是**（本文 §2 已说明） |

所以准确的表述是：**D 是一个"以该前置工作获得授权与人力为前提"的建议。**
本文只能指出它是必要的、且对 B/C/D 三者通用，**本文不能自己解除那个前提**。
采纳 D 时需要同时指定由谁承担这项构建工作；若不授权，则实际可得的选择是 **A**，
而不是 D ——这一点不应被 §0 的"三个选项都绕不过它"掩盖。

**D 的交付物不是"TLA 版 rmsnorm"，是一份带数字的前置调研**：那个便宜内核的
编译/计时/带宽/staging 四项读数。有了它，A/B/C 里选哪个都变成有证据的选择。

### 10.2 什么证据会翻转这个建议

**转向 C（直接迁移）**——需要**同时**满足：

| # | 条件 | 现状 |
|---|---|---|
| 1 | 出现一个含 `tla` 的**可安装 wheel**（或本机能构建） | 今天没有（§2） |
| 2 | §3 的 L2 损失被**翻案**：上游把 `LoadOp` 的 cache 属性改宽，或向量拷贝绕开窄枚举 | 今天收窄在 C++ 模板层（§3.4） |
| 2b | 或：实测证明某个窄策略能在我们的窗口复现 ~1.10× | **无从测**（§3.5，EvictFirst 即默认） |
| 3 | §7 的 staging 差距被量化且可接受 | 未量化 |
| 4 | TLA 暴露出一个 TileLang 没有的、对我们有实质收益的杠杆 | **未找到** |

**注意条件 2b 已经死了**，所以翻案实际上必须走 2（编译器侧改动）。这是本文最硬的一处：
**它不是"等实验"，是"等上游改属性类型"。**

**转向 A（彻底不碰）**——需要：便宜内核上 TLA **连编译或连 MATCH 都做不到**。
那时 A 不是"保守"，是"有证据的结论"。

**转向 B（并行保留）**——需要：便宜内核证明 TLA **可用且有独立价值**，
但 rmsnorm 的迁移收益仍不足以退役 tilelang。B 在那个时点才合理，
因为两套 plumbing 的维护成本只有在两条路都**活着**时才值得付。

### 10.3 本文不建议的事

- **不建议**因为 TLA 更新、厂商主推、或是"看起来是方向"而迁移。本文找不到收益。
- **不建议**在拿到 wheel 之前做 B 或 C 的 plumbing（§9.1 那两个静默失败点要先修，
  但在没有产物的前提下修完也无法验证）。
- **不建议**用 `EvictFirst` 去"试一试"（§3.5：它就是默认值，测到的是噪声）。

---

## 11. 本文没有证明的

- **没有在设备上跑过任何 TLA 代码——一句写在最前面的总纲：本文关于 TLA 的全部陈述都是
  [source-only]，不存在任何 TLA 的设备测量值，因为没有人在本卡上跑过 TLA。**
  本机无 `tla` 包（§2），连一次编译都未成功过。
  这包括：能力映射（[tla-capability.md](tla-capability.md) 全表）、L2 损失（§3，源码五层但未上机）、
  以及对 TLA 版 rmsnorm 的任何时长/带宽预期。**没有任何一个 TLA 数字是测出来的。**
- **没有测过 L2 损失的设备值**（§3.5）。1.103× 的损失是源码推理，不是实测。
- **没有量化 staging 差距**（§7）。这是本文最大的未量化项。
- **没有评估 `LoadGM2L1Params.l2_cache_ctrl` 借到向量路径的可行性**——那是设计问题，超出范围。
- **没有核 `StoreUB2GMParams` 之外是否另有写策略通路**；就源码所见没有，
  但 v33 的写策略是硬件默认 0，所以这一条**不需要通路**（§3，已关掉）。
- **没有估内核重写的真实工作量**。v33 的 284 行是已调优终态，不是起点。
- **第 5、6 节**取自 [`adapter-audit.md`](adapter-audit.md)（作者为 lead），
  其中 `GATE_TEST`、`parse_argv`、`TOOLCHAIN_KEYS`、marker 负向后查、`README.md:23-31`、
  `RECORDED, NOT GATED` 整表、`evaluator.py:6-13`、`PLAN` `:3` 及两个 adapter 的行数
  我已逐条独立复核。**该文件不是来源独立的第二信源**——它是 lead 自己简报的源码硬化版（见文首说明）。

---

## 12. 一句话给读者

**TLA 不是 TileLang 的新版本，是另一个 DSL 家族；本机今天没有它的可安装产物；
迁过去会按我们自己的数据丢掉 1.103×，还会额外背上一笔未量化的 staging 重搜成本。**
所以现在不该迁——但也**不该把它忘掉**，因为让产物存在这件事，B/C/D 三条路都要先做，
而它就是 D 的第一步。**先把可评估性买回来，再决定要不要买迁移。**