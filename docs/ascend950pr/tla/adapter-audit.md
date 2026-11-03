# tla-npu adapter 审计：到底是什么、证明了什么、做不到什么

**作者**：lead（接手；原 task-8 由调研 agent 起草未完成，本文件由我从一手源码重写）。
**日期**：2026-11-03。**性质**：源码审计，**未运行该 adapter，未在设备上跑任何 TLA 代码**。
**范围**：/home/s00653124/workspace/AgentGates/agent_gates/adapters/tla-npu/。

标签：**[verified: 路径:行]** / **[plan-only]** / **[inferred]** / **[unknown]**。

> **说明**：原调研 agent 在同一任务上运行约 40 分钟后中断，未产出文件。本文件不是它的续写，
> 是我按同一清单重新从源码核出的结果；它中断前曾表示"另有两处对简报的更正"但未能交付，
> 那两处更正因此**不在本文件内**，读者不应假定它们已被吸收。

---

## 0. 一句话

**tla-npu 是一个诚实、设计克制的"设备回归门"——它证明"这台机器能编译并执行一个 TLA kernel 且数值正确"，
不证明 TLA 的 API 覆盖面，也不能为任意候选内核打分。**
它今天门的是一个**固定文件名**的测试，因此**不能**直接用来验收我们的 rmsnorm 内核。
而本机**没有**含 tla 的可安装 wheel（§6），所以即使改造完这道门，也没有可编译的 TLA 交给它门。

---

## 1. TLA 是什么——先纠正问题本身

**TLA 不是 TileLang 的扩展，也不是 TileLang 的新版本。** 它是厂商 triton_ascend wheel
加到 **Triton** 上的一个命名空间。

TLA 自己的 __init__.py 原文 [verified: third_party/ascend/language/tla/__init__.py:23-25]：

> Independent tla.* programming namespace for Ascend NPU kernels, migrated from the TLA repo's
> SIMD VF additions. It does **not** reuse the existing al/bl general interfaces
> (**different programming paradigm**).

导出面 [verified: 同上 :27-32]：alloc_buffer / alloc（片上 buffer + versions）、copy（GM↔buffer）、
view、scope（tla.scope.simd() / tla.scope.cube()）、simd、annotate_buffer_versions。

adapter README 对同一事实的表述 [verified: README.md:11-13]：它运行 test_tla.py，
该测试 "builds a cast_back kernel with @tla.jit, using tla.alloc_buffer, tla.view, tla.copy,
tla.scope.simd() and tla.simd.*"，并 "runs the same semantics as a plain @triton.jit kernel"。

**结论**："从 tilelang 迁到 TLA"是**更换 DSL 家族**（TileLang → Triton + tla.*），
不是同族升级。操作者的措辞可能隐含了后一种理解，值得尽早澄清。

---

## 2. 这道门证明什么：11 个检查逐条落点

manifest.json 的 tla-npu-e2e 门声明了 11 个 observationChecks:true。
它们的实际计算在 scripts/tla_gate.py 的 observations() [verified: :904-975]。

| # | check | 落点 | 抓住什么 | 抓不住什么 |
|---|---|---|---|---|
| 1 | exitCodeZero | :920 前置 evaluate() | 进程退出码非 0 | — |
| 2 | scriptExitZero | :925 | 测试脚本自身异常 | 崩溃前的部分成功 |
| 3 | castBackOk | :927 | stdout 含 CAST_BACK OK | 必须不被 TRITON_ 前缀误吃（见下） |
| 4 | tritonCastBackOk | :928 | stdout 含 TRITON_CAST_BACK OK | 同上 |
| 5 | tensorMovedToDevice | :931 | 进程内计数 device='npu' 工厂 + Tensor.npu() | 张量是否被**正确**使用 |
| 6 | numericalAssertionsObserved | manifest | 数值断言确实跑过 | 断言强度是否足够 |
| 7 | npuDeviceUsed | :921 | npuAvailable | 用的是**哪一张**卡（见 §4） |
| 8 | compiledArtifactsPresent | :937 | .npubin 计数 > 0 | 产物是否对应测试的语义 |
| 9 | synchronizeOk | :941 | torch.npu.synchronize() 成功 | — |
| 10 | npuAvailable | :921 | 设备可见 | — |
| 11 | deviceCountOne | :947 | npuDeviceCount == 1 | 掩码是本门自己 export 的（见 §4） |

**一个值得记录的细节（诚实性的正面证据）**：CAST_BACK_MARK 是 TRITON_MARK 的字面子串，
所以朴素的 in 判断会让纯 Triton 参照路径的输出**同时**为 TLA 路径记分。
代码用负向前查绕开 [verified: :921-922]：

    cast_back = isinstance(stdout, str) and bool(re.search(r'(?<!TRITON_)' + CAST_BACK_MARK, stdout))

作者在注释里写明了原因 [verified: :916-919]："a plain in test would credit the TLA route
for output that only the plain-Triton reference produced."

---

## 3. 它今天能接受候选内核吗：**不能**

**答案：不能，且不是配置问题。**

| 证据 | 内容 |
|---|---|
| 测试文件名 | GATE_TEST = 'test_tla.py' 是**模块级常量** [verified: tla_gate.py:124]，注释说明它 "fixed in this digest-pinned file, and never read from the environment" [verified: :117-124] |
| 测试**目录** | 来自 TLA_NPU_TEST_DIR，一个必填绝对路径 [verified: :349-366]，且必须是 test_tla.py 所在目录 [verified: :360-361] |
| CLI 参数 | parse_argv() **只**接受可选 --timeout SECONDS，其它任何参数一律 BridgeError [verified: :565-573]；main() 另只特判 --check-config [verified: :1046] |
| 可配置键白名单 | TOOLCHAIN_KEYS 是封闭元组，未知键**报错** [verified: :140-166]，其中没有任何 CANDIDATE 类键 |

**最小改造**（诚实估计）：允许 --test-file（或 TLA_NPU_TEST_FILE）覆盖 GATE_TEST，
并把"给定文件必须存在且输出两个 marker"的契约参数化。**但**这两个 marker 是 cast_back
语义专属的，换成我们的内核后 castBackOk / tritonCastBackOk 就没有意义了，需一并参数化。
**所以这不是加一个 flag，是改门的契约。** [inferred：未实施，行数不给点估计]

---

## 4. 记录 vs 门控：哪些字段只是"给读者看"

这道门明确区分两件事，并把区分写进注释 [verified: :949-975]。

**RECORDED, NOT GATED**（下表的理由均为门自身注释的转述）：

| 字段 | 为什么不门 |
|---|---|
| visibleDeviceEcho | 是 runner **自己** export 后读回的值，为真 by construction |
| deviceUuid / deviceName | 本机**两张卡报同一个 UUID**，门它等于门一个常量；且 get_device_properties 报的是型号不是板卡 |
| tlaImplementationSha256 / FileCount / tritonVersion / tlaHasDma / tlaImplementationError | 只让读者把一次 PASS 绑到产生它的那份 TLA 代码上 |

**这条纪律与我们 R84 独立收敛**：agent_gates/evaluator.py:6 的 equal() 先做
type(actual) is not type(expected) 再 == [verified]，而 manifest.py **不**对
observationChecks 做类型校验。所以一个浮点阈值（{"speedupAtLeast": 1.2}）**能加载**，
然后**永远失败**，因为 float 永远不等于 JSON 的 true。

PLAN-perf-workflow.md §3 因此**主动拒绝**把性能阈值做成门 [verified: :37-78]，
理由是把中位数变成确定性判定、而本卡跨进程漂移 25.8%。**这是设计选择，不是代码限制**——
一句话就能加一个比较算子，作者明确拒绝加。**这与我们 R84 用五轮 RETEST 换来的
"记录而不门控"纪律是同一条结论，两个项目独立到达。** 这一点应记在 adapter 账上，
而不是当作它的缺陷。

---

## 5. 现有状态：plan 与 code 的差距

PLAN-perf-workflow.md 首行自述 [verified: :3]："Status: proposal, reviewed.
**Nothing in this document is implemented yet.**"

| 项 | 状态 | 证据 |
|---|---|---|
| 一个固定的端到端设备回归测试 | **已实现** | §2、§3 |
| 接收 agent 自己写的内核 | **未实现**（plan §2 明说这是 "the current adapter does not fit" 之处） | [plan-only: :47-56] |
| 逐 kernel 报告结果（而非单次 pass/fail） | **未实现** | [plan-only: :55] |
| 性能记录进 receipt | **部分**：TLA 实现身份/版本已记录；**时长未记录** | §4 表 |
| 性能阈值门 | **主动拒绝** | §4、[plan-only: :60-78] |
| 观察经 MCP 到达 agent | **未实现**（plan 说 in scope） | [plan-only: :9-11] |

---

## 6. wheel 钉版：为什么"取最新"在这里是错的

README 给了一个具体的两轮对照 [verified: README.md:139-158]：

| wheel | tla 面 |
|---|---|
| triton_ascend-3.6.0.post20261030+gita414a5cb | **不可用**：无 tla.jit（@tla.jit 立即 AttributeError）；native library 只导出 34 个 create_tla_* / create_simd_* 中的 **9 个**，把 tla.jit 别名到 triton.jit 只会把失败推进 native 层（no attribute 'create_tla_alloc'）。**不可通过 shim 或换 wheel 配对救活** |
| triton_ascend-3.6.0.dev20261030+git0c9d99ca | **可用**：跑到 CAST_BACK OK / TRITON_CAST_BACK OK，零 mismatch |

> **⚠️ 更正（2026-11-03，task-12）：本节初版结论"本机一个都没有"是错的。**
> 真的部分是：磁盘上那 6 个 triton_ascend **wheel 文件**（/home/s00653124/Packages/ 与 /data/pri 下）
> 的 `language/extra/tla/` **均为 0 文件**；默认环境 s00653124_tilelang 的 `language/extra/` 只有
> `cann, cuda, hip, kernels, libdevice.py` [verified，我复核]。
> **但那答的是"wheel 文件里有没有"，不是"本机有没有能用的"。**
>
> **本机真正可用的是 `/home/s00653124/.conda/envs/s00653124_tla`**，装的是
> `triton_ascend-3.6.0.dev20261101+git2a00c605`（既非上表两个 wheel 中的任何一个），
> 含 **21 个 tla .py**、RECORD **43** 条；`tla.jit` / `tla.dma` / `tla.annotate_buffer_versions` 均存在；
> **门实测 PASS**（run `9e8c59f1c356badc32a0e0d396af4659`，11/11，`Ascend950PR_9579`，12.45 s）。
> 运行路径见 [`how-to-run.md`](how-to-run.md)。
>
> 所以"PASS 不可复现"这句要撤回：**它可以复现，而且已经复现过。**
> 仍成立的是：**它证明的是"这台机器能跑 TLA"，不是"我们的内核已被验收"**（§3）。

---

## 7. 两个 adapter 家族：不要混为一谈

agent_gates/adapters/ 下有 5 个：

| adapter | 是什么 |
|---|---|
| **tla-npu** | 本文件审计对象。11 检查，capabilities: ["device-regression"]，1125 行门，**固定测试文件** |
| **forgekernel-ascend-npu** | **今天门我们当前认证路径的那一个**。ACCEPTANCE.md 598 行、accept_rmsnorm.py **2044** 行、fk_gate.py 1078 行 |
| ascendnpu-ir / tilelang-hivm / tilelang-hivm-npu | 编译器与后端层的门 |

所以"TLA adapter 已经在 AgentGates 里了"**是真的**，但**不意味着集成已经完成**：
执行验收职责的是**另一个** adapter。迁移要动的不是"启用一个已有开关"，
而是把验收职责从 forgekernel-ascend-npu 迁移或复制到 tla-npu，
而后者今天连候选内核都收不了（§3）。

---

## 8. 这道门的诚实性（应当记录，不应被审计口径误伤）

以下都是正面事实：

- README **主动声明 PASS 不意味着什么** [verified: README.md:23-31]："It does **not** mean
  the whole TLA surface is covered, and it is not a compiler test suite."
- **缺失数据一律为假**，从不假定 [verified: observations() docstring, :906]：
  "Missing data is false, never assumed. A run that produces no output, an unavailable device,
  an all-skipped suite and a failed synchronization cannot pass."
- **拒绝断言自己证明不了的事**：.o 被计数但不要求，因为后端不产出它，
  "requiring it would be unsatisfiable rather than strict" [verified: :935-937]。
- **拒绝把 by-construction 的值为真当作证据**（§4 的 RECORDED NOT GATED 整表）。
- 设备占用**报告而不强制**，理由是 "on this class of machine a card is shared" [verified: README.md:65-70]，
  卡被占时给 BLOCKED_ENV——这与我们"争用导致 RETEST 而非 FAIL"的区分同源。

**结论**：如果我们的迁移评估把这道门描述成粗糙，那是我方错误。它的克制程度与 R84 的
测量纪律相当，且独立到达。

---

## 9. 对迁移决策的三条净结论

1. **它不能当验收门用**，今天门的是一个固定文件（§3）。改造是改契约，不是加开关。
2. **它证明的那件事本机已经做到了**（§6 更正）：**PASS 可复现，且有回执。**
   但它证明的是"TLA 能在本机编译并正确执行"，**不是**"我们的内核已被验收"。
3. **它的纪律可以照搬**（§4、§8）："记录而不门控"、"缺失即假"、"不断言 by-construction 的值"
   ——这三条在新路径上应原样继承，而不是重新发明。

**本文不回答"该不该迁"**（见 README.md 与 migration-surface.md）；
它只回答"这道门能给你什么"，答案是：**一个诚实的回归信号，和一套值得照抄的纪律，不是一个验收器。**

---

## 10. 复现命令

    cd /home/s00653124/workspace/AgentGates/agent_gates/adapters/tla-npu

    # 固定测试文件 + 只接受 --timeout
    grep -n "GATE_TEST\|parse_argv" scripts/tla_gate.py
    sed -n '565,575p' scripts/tla_gate.py

    # 11 个检查的真实计算
    sed -n '904,975p' scripts/tla_gate.py

    # 封闭配置键
    sed -n '/^TOOLCHAIN_KEYS = (/,/^)/p' scripts/tla_gate.py

    # 精确比较（浮点阈值必然失败）
    sed -n '6,13p' ../../evaluator.py

    # 本机无含 tla 的 wheel
    for w in $(find /home/s00653124 /data/pri -name "triton_ascend*.whl"); do
      echo -n "$w "; unzip -l "$w" | grep -c "extra/tla/"; done

    # plan 自述未实现
    sed -n '1,12p' PLAN-perf-workflow.md
