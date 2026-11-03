# Ascend950PR RMSNorm：R1–R84 优化轨迹（跨会话复盘）

> **本文取代 `perf-report.md` 成为 R1–R84 的轨迹叙事。** `perf-report.md` 保留不删、不改——它记录的是
> 2026-11-02 14:2x 那个时刻的真实认知（prefill 0.3224 FAIL），本身是轨迹的一部分。若要读"当时怎么想的"，
> 读它；若要读"整个 84 轮到底发生了什么"，读本文。
>
> **范围：** Ascend950PR、bf16 RMSNorm、TileLang-Ascend SimdVF。两个会话：原始会话 R1–R80，当前会话 R81–R84。
> **事实骨架：** `docs/ascend950pr/evidence/optimization-ledger-2026-11-03.json`（39 条逐轮记录，其每一条所引
> artifact 均已核对存在）。本文的数字取自该 ledger 或直接取自 ledger 所引的原始 artifact。
>
> **标签约定**（沿用 `core-count-and-l2-breakthrough.md`）：
> **[measured]** = 有具名 artifact 的实测值；**[derived]** = 对实测值做算术；
> **[quoted]** = 引自其他文档的转述；**[spec]** = 本仓库任何测量都未确立的机制或推理。
>
> **判定词表：** 本文只用门的词表 **MATCH / EXCEED / FAIL / RETEST**，从不为整体写"通过"或"达标"。
> 门是逐 shape 判定的，因此文中严格区分 **shape 级判定**与 **run 级判定**——这两者在本项目里多次不一致，
> 而这种不一致本身是结论（§8.5）。

---

## 1. 执行摘要

### 1.1 最终数字

| 形状 | 判定 | 点估计 | 95% 区间 | 配对 | quiet | 绝对时长（候选 / 厂商） |
|---|---|---|---|---|---|---|
| **prefill 8192×4096** | **EXCEED** | **1.1883** | **[1.1833, 1.1933]** | 60 | **60/60（100%）** | 0.051932 / 0.061693 ms |
| **decode 64×4096** | **EXCEED** | **1.7144** | **[1.6699, 1.7448]** | 80 | 76/80（95.0%） | 0.002187 / 0.003745 ms |
| 精度 | 合格 | 6/6 | — | — | — | — |

两形状均以**下界**越过 1.0（prefill 1.1833、decode 1.6699），不是只靠点估计。
两次均为单形状 run、device 1、`verdict.status = "EXCEED"`、`verdict.reasons = []`、`stage = "performance"`。
离散度：prefill P90/P10 = 1.0226，decode 1.1540，均在门的 1.25 上限内。
[measured: §10.1 所列五份 `accept-rmsnorm-v33-*`]

### 1.2 一句话结论

**84 轮里真正的修复只有两处，且都不在 kernel body 里：把 `T.Kernel(64)` 改成 56（这颗芯片的 AIV 核数），
和把读路径的 L2 策略从 `NORMAL_FV` 改成 `NOTALLOC_CLEAN`。**
v33 的 SimdVF 主体与 v23 逐字节相同。前 80 轮否决了 R80 自己列出的 9 条杠杆、十几个内核变体，
**却从未动过这两个常量里的任何一个——而它们一直是可测的。**

### 1.3 诊断改了三次

| 时代 | 轮次 | 信念 | 死因 |
|---|---|---|---|
| **A** | R1–R58 | DMA 流水喂不饱向量单元 | 杠杆（rows）在 rows=8 处撞 UB 硬墙，无后续机制 |
| **B** | R59–R80 | L2 策略是决定性旋钮，然后其余皆死路 | 死路判断成立，但"穷尽"是错的：网格尺寸从未变过 |
| **C** | R81–R84 | 答案在启动几何，不在内核 | 成立 |

每个时代写**信念 → 证据 → 什么活下来了 → 什么死了**。活下来的部分通常比死掉的部分小，
但那不是失败——它是最后这 4 轮能一次命中的原因。

---

## 2. 逐轮时间线

来源：`optimization-ledger-2026-11-03.json` 的 `ledger` 数组，39 条，逐条对应。
轮次标签按语料原样保留（`"3-5"`、`"9-45"` 等是语料本身的分组，不是本文的合并）。
"证据"列给的是该轮 ledger 记录所引 artifact 的**件数**；**"—"表示该轮无原始 artifact**
（只有 R9–R45 一行属此类，其 ledger 记录自带 `note: log-only summary`）。

| 轮 | 时代 | 类型 | 判定强度 | 决定性数字 | 证据 |
|---|---|---|---|---|---|
| R1 | A | diagnosis | 配对实测 | `vcadd` 六级链到标量 == torch；bf16 `UNPK_B16` 载入精确 | 2 件 |
| R2 | A | diagnosis | 单次实测 | bf16 写回 `PK_B32`、`vdupv` 广播、rstd 链误差 2.4e-8 | 1 件 |
| R3–5 | A | kernel-change | 单次实测 | 首个端到端内核 0.539× →（w 预转换）0.581× | 1 件 |
| R6 | A | diagnosis | 单次实测 | vec 73.1% / mte2 64.9% / mte3 22.6%（厂商 52.5 / 91.7 / 78.1） | 1 件 |
| R7 | A | kernel-change | 单次实测 | decode 1.29；prefill 0.622；cube 路径设备编译失败 | 1 件 |
| R8 | A | diagnosis | 派生 | 每 64 元素约 10 条向量指令，无可用融合原语 | 1 件 |
| R9–45 | A | kernel-change | **转述** | v2–v3 迭代、结构调优、流水深度、多累加器；**逐轮未单独归档** | **—** |
| R47 | A | negative-result | 单次实测 | 候选无法认证：自家 ForgeKernel run 占卡 | 1 件 |
| R48 | A | verification | 配对实测 | 干净窗口：decode **1.2687 EXCEED**；prefill **0.7566 FAIL** | 1 件 |
| R49 | A | diagnosis | 单次实测 | 归因：vec 60 vs 36.8 µs；DMA-in 68.9 vs 63.1 µs | 1 件 |
| R50 | A | negative-result | 配对实测 | 加深流水 + 输出双缓冲：0.7526，比值不动 | 1 件 |
| R51 | A | verification | 配对实测 | **decode EXCEED 1.2809 [1.2470, 1.3087]**；prefill 0.7574 | 1 件 |
| R52 | A | negative-result | 配对实测 | `annotate_unlimit_memory` 不放宽 launch 检查（286720 B 必拒） | 1 件 |
| R53 | A | correction | 配对实测 | 门 bug：候选无测量被判"委托厂商" → 改 RETEST | 1 件 |
| R54 | A | correction | 配对实测 | 声称连续分区 **+8.9%** —— **R58 撤回** | 1 件 |
| R55 | A | measurement-methodology | 单次实测 | 三名外部用户占卡；三个变体 0 quiet，全部作废 | 1 件 |
| R56 | A | verification | 配对实测 | v5 全输入族精度通过 | 1 件 |
| R58 | A | correction | 配对实测 | 候选 0.0801→0.0794（0.8%）；厂商 0.0591→0.0645（+9%） | 2 件 |
| R59 | B | negative-result | 配对实测 | **真实竞争**：rows=2 全尺寸错，batch=128 不确定（rel=1.0） | 1 件 |
| R60–61 | B | kernel-change | 配对实测 | v15 源码级展开 + 每行寄存器：**0.0794 → 0.0755 ms** | 1 件 |
| R62 | B | diagnosis | 配对实测 | vec **41.0 vs 37.2** µs；mte2 76.2 vs 69.9；瓶颈转 DMA | 1 件 |
| R63 | B | kernel-change | 配对实测 | v16 **decode 1.3160 [1.3013, 1.3324]**；prefill 0.0745 ms | 1 件 |
| R64 | B | diagnosis | 单次实测 | 拷贝地板 1.7 TB/s、128 MB 需 0.078 ms —— **R65 撤回** | 1 件 |
| R65 | B | breakthrough | 配对实测 | 写路径未设策略 1.138 → `NORMAL_FV` **3.079 TB/s（2.7×）** | 2 件 |
| R66 | B | negative-result | 配对实测 | 微基准冠军 `NORMAL_LV` 让内核**变慢**（0.0648→0.0699） | 2 件 |
| R67 | B | kernel-change | 配对实测 | **v23**：decode 1.3726；prefill 0.8074（0.065149 ms） | 1 件 |
| R68–70 | B | negative-result | 配对实测 | 七类变体无一胜 v23；v28 八链 0.072535（慢约 12%） | 3 件 |
| R71 | B | verification | 配对实测 | v23 全输入族精度 pass；decode 1.3726 / 1.4056 | 1 件 |
| R72 | B | verification | 配对实测 | 冷 cache 复现 1.3550；块大小失效（32 KB 1.757 / 64 KB 1.725） | 1 件 |
| R73 | B | diagnosis | 单次实测 | 混合流量：我们 1.97 vs 厂商 2.42 TB/s（+23%） | 1 件 |
| R74 | B | correction | 配对实测 | 绝对时长漂移 **±20%**；只有比值稳定（0.771–0.813） | 1 件 |
| R76 | B | verification | 配对实测 | 5 项文档数字对原始证据逐项审计，全部一致 | 1 件 |
| R77 | B | verification | 配对实测 | 第四窗口 **0.8502 [0.8294, 0.8830]**，四窗口中最好，仍 FAIL | 1 件 |
| R79 | B | verification | 配对实测 | 交付前独立复验：精度、确定性通过；目标未达成 | 1 件 |
| R80 | B | archival | 配对实测 | prefill 四窗口 0.7711–0.8502，最好下界 **0.8294**；**宣布杠杆穷尽** → **R84 撤回** | 1 件 |
| R81 | C | diagnosis | 配对实测 | 模拟器可行：341 s 墙钟，8.1 vs 5.826 µs（约 39%） | 1 件 |
| R82 | C | correction | 配对实测 | `SIMT_*` 是模型命名约定，不是 SIMT 回退（13925/15360 = 91%） | 1 件 |
| R83 | C | negative-result | 配对实测 | 模拟器对 v23/v28 **逐位相同**（338639 cycles、13925 executed），硬件差 12% | 1 件 |
| R84 | C | breakthrough | 配对实测 | **64→56 块 + 读路径 `NOTALLOC_CLEAN`**：prefill 1.1883、decode 1.7144 | 5 件 |

**关于 R9–R45：** 这一行是语料的**有意**分组（`claim_strength = "quoted"`，`artifacts = []`，
`note = "log-only summary; per-round detail not separately archived"`）。
本文不为它虚构逐轮细节；缺口照原样呈现，用"—"而不是留白。

---

## 3. Era A（R1–R58）："DMA 流水喂不饱向量单元"

### 3.1 信念及其看起来有多硬

msprof 的管道占比给出了一幅极其清晰的图，来自仓库示例内核与厂商同窗口对照（4096×4096 bf16）：

| 管道 | 候选 | 厂商 | 当时的读法 |
|---|---|---|---|
| `aiv_vec` | **84.9%** | 80.7% | 候选向量更满——已经是向量受限 |
| `aiv_mte2` | **32.0%** | **64.2%** | 搬入只有厂商一半 |
| `aiv_mte3` | **12.6%** | **44.9%** | 搬出只有厂商 28% |

[measured: `perf-report.md` §3.1，源自 `evidence/probe-bf16-2026-11-02.json`]

读法：向量单元 85% 忙、数据供给 32%，所以它在**空转等数据**。这不是算力问题，是流水问题。
配套的量级证据是 rows 扫描（8192×4096）：rows=1 **289 µs** → rows=2 **189.6 µs** → rows=4 **164.2 µs**，
即 **1.76×**；rows=8 需要 **271360 B**，超过 256 KiB 的 UB 上限，launch 阶段被拒。
[measured: `evidence/simdvf-r6-pipes-2026-11-02.json`]

一个时代能有的最好开局：一个机制故事、一个方向正确的 1.76×、一个清晰的下一步。

### 3.2 这个时代真正做对的事

**内核确实不是 compute-bound。** 这个判断从未被推翻，而且它是后续一切的前提。如果当时认定
"向量指令太多"，研究方向会走向指令融合与删除——而真实答案在数据搬运与启动几何上。
R73 后来把这条判断量化成 1.97 vs 2.42 TB/s 的混合吞吐 [measured]，但**方向在 R6 就对了**。

第二条做对的事：**把残差记成了否定的机制结论，而没有含糊过去**。rows=8 撞 UB 硬墙（271360 B > 256 KiB）、
`annotate_unlimit_memory` 不放宽 launch 检查（R52，286720 B 必拒）、`vsqz`/`vusqz` 不是平方和原语、
bf16 `vmula` 无混合精度、`T.gemm` 设备编译失败——这些是 Era A 用真实探测换来的，且**至今没有一条被推翻**。

### 3.3 这个时代错在哪

**它把"搬运不足"归因到流水结构与缓冲深度，而真实的主导项是一个没人设置过的缓存策略，
以及更后来才被发现的错误网格尺寸。**

致命处不在推理，在**杠杆耗尽之后的动作**。R50 的"加深流水 + 输出双缓冲"在门上给出 0.7526——比值不动
[measured: `evidence/accept-v3-prefill-r50.json`]；R58 的五变体表进一步确认候选时长锁在 0.079–0.083 ms。
一个正确的诊断如果其对应杠杆已经用完，下一步应当质疑**前提**（是不是有别的层没被看？），
而不是宣布"还差 20–30%、无可见机制"。

当时的原话是：prefill 还需 20–30% 削减，"给定每 64 元素已约 10 条向量指令，且 vec 忙碌约 60 µs、
DMA-in 约 65–69 µs，单纯调度/分区无法弥补，**需要硬件级融合或更强的重叠手段**"
[quoted: `plan-40-rounds.md` §8, R46–R58]。

**"需要硬件级融合"是一个不可执行的结论。** 它把问题移交给了不存在的硬件，
而不是移交给了下一个可测假设。这是 Era A 真正的方法论错误——不是结论错，是**出口错**。

### 3.4 什么活下来了

- **decode 认证**：R51 **EXCEED 1.2809 [1.2470, 1.3087]**，候选 0.002777 / 厂商 0.003596 ms [measured]。
  从未被推翻。
- "端到端由数据搬运决定"的定性结论。
- 一批硬约束（UB 上限、无融合原语、无混合精度 FMA、cube 不可用）。

### 3.5 什么死了

- **"流水/分区/缓冲深度是杠杆"**：R58 五变体表关掉了这条路——v4 0.0801、v5 0.0794、v7 0.0829、
  v8 0.0810、v9 0.0862 ms，除两条明显更差的，其余全在噪声内 [measured: `evidence/r58-variant-table.json`]。
- **R6/R7 的"结构上限约 0.8×"**：这条被自己的 artifact 明确撤回——"profiler utilization 不是硬上限的证明"，
  该测法未被充分配对 [quoted: `simdvf-r6-pipes` / `simdvf-r7-decode` 的 `review_correction` 字段]。
  事后看它错得很有教益：R77 实测 **0.8502** 已超这个"上限"，R84 的 **1.1883** 超它 48%。
- **R54 的 +8.9%**：见 §7。

---

## 4. Era B（R59–R80）："L2 是决定旋钮，然后其余皆死路"

这一节分两半读。**前半（R59–R67）是 84 轮里最有效的连续推进；后半（R68–R80）是一次系统性、
高质量、但在结论上错了的封口。**

### 4.1 开场不是信念，是一次否定

R59 的两阶段内核（先全部归约、再全部缩放，用共享缓冲传每行 rstd）**有一个真实竞争**：
rows=2 形式在所有测试尺寸上都错，batch=128 时同一输入 5 次结果不同（rel=1.0）。
给它显式声明流水版本（v14）也没修好（rel 23.4 与 NaN）。
v10/v11/v13/v14/v14b 全部 REJECTED，其 **0.0733 ms 不得作为性能结论**。
[measured: `evidence/r59-two-phase-race.json`]

这条否定的价值在于它排除的是**"用共享状态跨相位"这一整类构造**，而不是两阶段本身。
R60 随即给出正解：**源码级显式展开 4 行 + 每行独立寄存器**，不引入任何跨相位共享状态。
R61 门控确认 prefill 候选 **0.0794 → 0.0755 ms** [measured: `evidence/r61-v15-gated.json`]。

然后 R62 的归因给出时代信号：**vec 41.0 µs vs 厂商 37.2 µs**（曾多 63%，现在只多 10%），
瓶颈正式移到 DMA [measured: `evidence/r62-dma-bound.json`]。**算力追平了，问题变纯了。**

### 4.2 R64 → R65：这个时代最重要的一次自我推翻

R64 用纯拷贝微基准得出"拷贝路径已触底"：读峰值 2.13、写 1.121–1.143、分离缓冲拷贝最优 1.725 TB/s，
128 MB 需 0.0778 ms [measured: `evidence/r64-copy-floor.json`]。结论写得很斩钉截铁：
"我们的完整内核 0.0745 ms 已经持平甚至略优于纯拷贝同样字节的 0.0778 ms"，
因此残差在搬运路径、且**在当前工具链内无法闭合**。

R65 推翻了它，原因是一个没人检查过的默认值：

| 写路径策略 | UB→GM 吞吐 |
|---|---|
| 未设置（`None`） | **1.138 TB/s** |
| **`NORMAL_FV`** | **3.079 TB/s** |
| `NORMAL_RED` | 3.089 TB/s |
| `NOTALLOC_CI` | 1.134 TB/s |

[measured: `evidence/r65-l2-breakthrough.json`]

**写路径不是饱和在 1.13 TB/s，它是被一个"未设置"的默认策略限制在 1.13 TB/s。**
显式设置后 **2.7×**，v18 候选落到 **0.0648 ms** [measured]。这一发现从未被推翻，并直接通向最终的答案。

R66 紧接着给出本项目的**方法论主定理**：读路径微基准冠军 `NORMAL_LV`（2.351 TB/s）
在真内核里让 prefill 从 0.064779 **退化**到 0.069899 ms [measured: `evidence/r66-v20-policy-fit.json`]。
**微基准排名不能搬到内核。** 这条定理在 R84 被量化成安全结论——`NORMAL_LV` 不只是更慢，
它在 decode 上 **30/30 全错**（最差 max_abs 12.3）。而它**至今仍是微基准冠军**。

R67 产出 v23：decode **1.3726 [1.3561, 1.3829]**、prefill **0.8074**（0.065149 ms），
stages=4 + 全 `NORMAL_FV` [measured: `evidence/r67-v23-best.json`]。
这个 artifact 后来被证明**就是最终答案的载体**。

### 4.3 R68–R70：系统性否定，且这些否定是对的

七类变体逐一挑战 v23，全部落败：

| 变体 | 改动 | 候选时长 | 结果 |
|---|---|---|---|
| v26 | rows=2, stages=4 | 0.079967 ms | 更差 |
| v27 | rows=2, stages=6 | 0.084178 ms | 更差 |
| v28 | rows=4, **每条 8 累加链** | 0.072535 ms | 更差（慢约 12%） |
| v25 | stages=4 + **y 三缓冲** | 0.073510 ms | 更差 |
| v29 | **x 五缓冲** | 0.080879 ms | 更差 |
| v22 | 读 `NORMAL_PREF` + 写 `NORMAL_RED` | 0.069120 ms | 更差 |
| v30/v31/v32 | 手动 `order`/`stage` 流水调度 | 0.0689–0.0743 ms | 无一胜自动调度 |

[measured: `evidence/r68-no-better-variant.json`、`evidence/r70-manual-pipeline.json`、`evidence/r69-decode-repeat.json`]

这些否定**大部分是对的且至今有效**。它们是真正的资产：说明"更大的 tile / 更深的流水 /
更多的缓冲 / 更多累加链 / 手动调度"这条路是死的，而且死得很确定。

### 4.4 R80：那次过早的归纳

收口记录在预算用尽（80/80）时写下 `objective_met = false`，并把 `levers_exhausted` 列成一张长表：
分区方式、rows 1/2/4/8、stages 2/3/4/5/6、输出缓冲 1/2/3、累加链 2/4/8、
**每方向 L2 缓存策略（FV/LV/PREF/RED/NOTALLOC/IDS/EXCLUSIV/WBH/WTS）**、
两阶段归约、手动流水调度、DMA 块大小。
[measured: `evidence/r80-closing-record.json`]

这张表里**有两处是错的**：

**(1) "L2 缓存策略（…NOTALLOC…）"被列为已穷尽，但 `NOTALLOC_CLEAN` 与 `NOTALLOC_KEEP` 从未在内核门控过。**
内核对读路径只试过 {unset, `NORMAL_FV`, `NORMAL_LV`, `NORMAL_PREF`}，写路径只试过
{unset, `NORMAL_FV`, `NORMAL_RED`}。`NOTALLOC_*`、`WTS_*`、`WBH_*`、`IDS_*`、`EXCLUSIV_*`
全部只被微基准排过名——**而 R66 刚刚证明微基准排名不能搬**。
[measured: `devkit-lessons-memory.md` §2.4，对全部 36 个 `candidates/*/kernel.py` 的 grep]

**(2) 网格尺寸从未被列为杠杆，因此也从未被质疑。** 从 v4 到 v23，每一个候选都是 `T.Kernel(64)`。

R80 的 `residual_cause` 写的是"混合读写吞吐 1.97 vs 2.42 TB/s，TileLang 未暴露 DRAM 页局部性与流数量"。
这句话**对 DRAM 成立，对 L2 策略不成立**——`devkit-lessons-memory.md` §6 在同期就指出了这一点：
"该说法对 DRAM 成立，但对 L2 策略不成立；`NOTALLOC_KEEP`、`NOTALLOC_CLEAN`、`NOTALLOC_DROP`
与 `WT_SHARE_FIRST_VICTIM` 全都可以用一行 `l2_cache_ctrl` 改动 reach，且从未门控。"

### 4.5 Era B 关闭时，具体地、可 reach 地未被测试的项目

| # | 项目 | 关闭时的状态 | 后来的结果 |
|---|---|---|---|
| 1 | **网格块数（`N_CORES`）** | 从未变过，恒为 64 | **R84 的第一处修复** |
| 2 | **读路径 `NOTALLOC_CLEAN`**（load enum 5） | 仅微基准，从未门控 | **R84 的第二处修复** |
| 3 | 读路径 `NOTALLOC_KEEP`（load enum 4） | 仅微基准 0.906 TB/s | R84 测了：**0.6600 vs 0.6626，无收益** |
| 4 | 读路径 `NOTALLOC_DROP`（load enum 6） | 从未门控 | R84 测了：**更快但数值错误** |
| 5 | 写路径 `WT_SHARE_FIRST_VICTIM`（store enum 12） | 从未门控 | **至今未测** |
| 6 | `NORMAL_PERS` 用于 W 拷贝（仅 8 KB，唯一被 56 核重读的张量） | 从未门控 | **至今未测** |
| 7 | TileLang 编译器缺口：VF 归纳变量 `int32_t` 阻断 Hardware Loop；VF 体内 `#pragma unroll N` 可达但 0/553 内核行使 | 已验证可达，未测 | **至今未测，且不需要设备** |

前两项的代价是约 74 轮；后五项至今仍然开着。

### 4.6 什么活下来了

- **L2 发现本身**（R65）：写路径显式 `NORMAL_FV` 的 2.7×，是最终答案的一半。
- **测量纪律**：R58（用候选时长而非比值判断）、R74（比值才是稳定量）、R66（微基准不能排队）——
  这三条把后续所有结论的可信度定了下来（§8）。
- **七类变体的否定**（§6.1）。

---

## 5. Era C（R81–R84）："答案在启动几何，不在内核"

Era C 只有 4 轮，前三轮**全部用于证伪一个工具**，最后一轮才是修复。这个配比值得单独说：
**用 3 轮确认"没有离线神谕"，是那 4 轮能一次命中的前提。**

### 5.1 R81–R83：模拟器研究，以决定性否定收尾

R81 发现 msprof 有 `op simulator` 子命令，且能对 v23 的最小等价模型（1 核、rows=4、6 迭代）端到端跑完，
输出逐指令 CSV（instr / pipe / call_count / cycles），并抽出 621 条依赖关系。
这严格强于设备侧 msprof——后者只有管道百分比。[measured: `evidence/r81-simulator-feasibility.json`]

但同一份 artifact 已经埋了两个致命观察：

- **MTE2 只占 0.1% 的周期**，而设备在同模型上报 DMA 27%、全内核上报 86–93%。
- **1 核模型是向量受限的（vec 87.7%），而真实的 64 核内核是访存受限的（mte2 86–93%）。**
  减核数恰好删掉了真正的那份 DRAM 争用。

R82 修正了一处误读：模拟流里的 `SIMT_LDK` / `SIMT_STG` / `SIMT_FFMA` 曾被认为是
"内核被降级到 SIMT 路径"，但同一模型的 `device_kernel.asc` 含 **93 处 `simd_inst` 调用**
（`vlds_unpack`、`vcvt`、`vmula`、`vadd`、`vcadd`、`vdupv`、`vdiv`、`vsqrt`、`vadds`、`vmuls`、`vsts_norm`），
执行 **13925/15360 = 91%** 的预期向量操作。**`SIMT_*` 是 camodel 的命名约定，不是回退。**
[measured: `evidence/r82-sim-correction.json`]

**R83 给出决定性否定。** 取一对在硬件上相差 12% 的内核——v23（四累加链）与 v28（八累加链）：

| | v23 | v28 |
|---|---|---|
| **设备实测** | **0.065149 ms** | **0.072535 ms**（慢约 12%） |
| 模拟 `total_tick` | 43686 | 43804 |
| 模拟 `total_cycles` | **338639** | **338639** |
| 模拟 `executed` | **13925** | **13925** |
| 模拟 `core_us` | 8.1 | 8.1 |

[measured: `evidence/r83-simulator-not-discriminating.json`]

两内核指令数确实相同（每行 64 loads + 64 fma，无论如何），所以 `executed` 相同是预期的；
但**它们的硬件时长差 12%，因为依赖链深度与寄存器压力不同，而模拟器报出完全相同的周期数**。
一个对"设备上是 12% 的差异"不动的模型，无法给优化排序，也就无法定位瓶颈。

**根因也查清了**：`npu-smi` 报的芯片是 Ascend950PR_**9579**，而模拟器**没有 9579 的模型目录**，
于是回退到通用的 `dav_3510` v100 模型；已填充的 9571/9572 模型在本机 `aclInit` 报错 507000
[measured: 同上，`why` 字段]。

R82 还留了一个**未解决的开问题**，本文原样保留：向量载入平均 **514 cycles/条**，看起来像 stall 归因
而非真实单指令成本；隔离的拷贝微基准也呈同一签名（**913 cycles/load**），所以该指标可能是 artifact，
"必须在任何结论之前用受控 A/B 验证" [quoted: `evidence/r82-sim-correction.json`]。

### 5.2 R84：修正不在内核里

**结论：没有离线神谕可用；只有设备能裁决一个 12% 的问题。**
于是转向"**测量这台机器**"而不是"测量这个内核"。

#### 5.2.1 第一处：64 块 → 56 块

三份独立的声明一致指向 56：

| 来源 | 数值 |
|---|---|
| `platform_config/Ascend950PR_9579.ini`（CANN 安装树内，非本仓库） | `ai_core_cnt=28`、`vector_core_cnt=56`（第 20、22 行） |
| `torch_npu` 设备属性 | `vector_core_num=56`、`multi_processor_count=56`、`L2_cache_size=128MB` |
| 阶跃探针 | grid 1→56 **平坦（~10.0 µs）**；57 **+82%（~18.5 µs）**；58→84 再平坦；112 第二阶 |

[measured: `core-count-and-l2-breakthrough.md` §1.1–1.2]

56 = 28 × 2（每 AIC 两个 AIV）。探针在 device 1 重复 6 次、device 0 一次，**阶跃位置一致**
（幅度不同：dev1 +82%、dev0 +110%），说明这是**部件的属性**，不是某张卡的状态。

固定总量扫描（13440 tiles × 4096 bf16 = 110 MB 进 + 110 MB 出 = **220 MB 聚合**，
**总流量不变、只改如何铺开**）：

| Grid | 时长 | 聚合带宽 |
|---|---|---|
| **56** | **187.2–188.3 µs** | **1169–1176 GB/s** |
| 64 | 260–277 µs | 794–846 GB/s |
| 28/32/40/48/60/70/80/120/140/280/420/560 | 持平或更差 | — |
| 224, 672（= 4×56, 12×56） | 187.9 µs，与 56 打平 | — |

[measured]。64 块比 56 块**慢 39–48%** [derived: 260/188.3 = 1.38，277/187.2 = 1.48]。
224/672 与 56 打平正是**波量化**的签名——它们是 56 的整数倍，跑整数个完整波、无 ragged tail。

**几何改动单独的贡献**（同 body、两臂均 `NORMAL_FV`）：8192×4096 上 64 块 0.0760–0.0766 ms →
56 块 0.0549–0.0556 ms，厂商 0.0555 ms，即 **~0.726 FAIL → ~1.01** [measured]。
两处改动因此可以干净地分解：**网格修复买下 FAIL→parity 这一步，L2 修复买下其余。**

2048 tiles 不能被 56 整除，所以切分改为 ragged：

```
tiles = batch // rows
nbase = tiles // N_CORES          # 每核至少这么多
rem   = tiles % N_CORES           # 前 rem 个核各多一块
start = core * nbase + min(core, rem)
cnt   = nbase + (1 if core < rem else 0)
```

`start` 的写法保证每段仍连续（无 per-core striding），余数无缝隙无重叠。
8192×4096/rows=4 即 36 base + 32 核各多 1；decode 64 tiles / 56 核即 8 核做两倍工——
**在波代价最高的形状上有 2× 不均衡，但实测仍 1.71×，说明不均衡被流水吸收**
[measured: `core-count-and-l2-breakthrough.md` §1.5]。

#### 5.2.2 第二处：读路径 `NOTALLOC_CLEAN`

机制：prefill 工作集 = 64 MB X + 64 MB Y = **128 MB，恰好等于 128 MB 的 L2**，每行只碰一次。
两向都用 `NORMAL_FV`（分配 + 首个牺牲）时，每次读 X 分配一条 L2 行、驱逐一条**脏的** Y 行，
于是在读流中插入 GM 写回。`NOTALLOC_CLEAN` 保留行但标记为 **Clean**，驱逐时无需写回。
写路径保持 `NORMAL_FV` 是有意的：Y 仍需要 L2 吸收，需要停止分配的是读侧。

20 个配对 quiet 窗口、device 1、56 块（比值 = 厂商/候选，越高越好）：

| 读策略 | prefill 8192×4096 | decode 64×4096 |
|---|---|---|
| `NORMAL_FV` | 1.0170 [1.0144, 1.0179] | 1.6500 |
| **`NOTALLOC_CLEAN`** | **1.1185 [1.1179, 1.1193]** | **1.6699** |

[measured]。即 CLEAN 比 FV 快 **1.103×**（prefill）/ 1.009×（decode）[derived]。
区间窄于 1%，所以 20 个窗口足以分辨这一臂。

**这里必须写清楚 devkit 的真实贡献。** devkit 提供的是**机制**，不是答案：

- **机制层面有效**：`l2_cache_mode_config.md:33,35` 明确写了"工作集超 L2 时旧数据须先写回 GM，
  占用 GM 带宽"，以及"只访问一次的数据应设为不进 L2"。这句话直接预言了 §5.2.2 的现象，
  且它使 R84 知道**该往哪个方向调**——这是研究真实的价值。
- **答案层面无效**：devkit 自己的 Case-5 等价物是 `NOTALLOC_KEEP`（load enum 4），在研究里排名第一。
  **在内核层面实测 0.6600 vs `NORMAL_FV` 0.6626——没有复现 devkit 的收益**
  [measured: `core-count-and-l2-breakthrough.md` §2.4]。
- 真正取胜的 `NOTALLOC_CLEAN` **在同一族里，但不是 devkit 点名的推荐值**。它之所以胜出，
  是因为它做的是"让脏行不存在"（消除写回），而不是笼统的"绕过 L2"——这比"照抄 devkit"更精确，
  也比文档单独能给出的结论更锐利。

#### 5.2.3 安全红线：两个更快的策略是错的

**`NOTALLOC_DROP` 与 `NORMAL_LV`（读路径）比安全策略快约 13%，并且会静默产生错误结果。
它们的性能数字不是收益，是"内核没在做正确的事"的测量值。**

每策略每形状 **64 次运行**（8 seeds × 8 runs），判据 `max_abs < 0.06`，参考值用 CPU float64：

| 读策略 | prefill 8192×4096 | decode 64×4096 |
|---|---|---|
| `NORMAL_FV` | 0/64 错 | 0/64 错 |
| `NOTALLOC_CLEAN` | 0/64 错 | 0/64 错 |
| `NOTALLOC_DROP` | 0/64 错 | **56/64 错，最差 16.6** |
| `NORMAL_LV` | （未报） | **30/30 错，最差 12.3** |

[measured: `core-count-and-l2-breakthrough.md` §3]

- **DROP 的形状依赖性正是它危险的原因**：prefill 上 0/64 干净，decode 上 56/64 错。
  **只在 prefill 上验证过的团队会把它发出去。**
- **`NORMAL_LV` 在 decode 上 30/30 全错**，而它恰恰是读路径微基准冠军（2.351 TB/s）。
  这是 R66 定理的量化版：**只有门控能排出策略优劣。**
- max_abs 12–17 对 0.06 的界，不是舍入差异，是另一个计算。

**规则：本内核可信的读策略只有 `NORMAL_FV` 与 `NOTALLOC_CLEAN`。**
任何新策略必须先在两形状上通过 64 次正确性表，否则其计时数字**一个都不许引用**。

### 5.3 什么活下来了

- 两处修复，及其**可分解的归因**（网格买 FAIL→parity，L2 买其余）。
- `NOTALLOC_KEEP` 的否定结果——它把 devkit 的机制从"绕过 L2"精确化为"**消除脏行写回**"。
- 安全表：DROP / `NORMAL_LV` 永久排除，且带 64 次运行的证据。

### 5.4 这个时代也没一次成功

修复不是一次到位的。第一次门控尝试因配置被拒（`--pairs` 低于 9、`--cache` 缺失或非空），
第二次 115 s 的 `--child-timeout` 中途过期——两次都是配置检查拒绝，不是内核问题。
随后**前三次正式门控全部 RETEST**，原因是 co-tenant 争用（§8.3）。
`NOTALLOC_KEEP` 作为"devkit 的答案"被认真测了，**没赢**。
**Era C 的正确性不在于它没错，而在于它每一步都把错误测出来了。**

---

## 6. 死路的代价与收益

84 轮里累积了 **40 个候选目录**（`candidates/`，其中 `rmsnorm_simdvf_v1…v33` 及其变体）、
R80 收口时列出的 **9 条**杠杆、十余个被完整门控否决的内核变体，以及 7 类被排除的架构假设。
逐个记下它们**排除了什么**，是为了让下一个人不必重跑。

### 6.1 被门控否决的内核变体

| 变体 | 改动 | 决定性数字 | **它排除了什么** |
|---|---|---|---|
| v9 | rows=8 + 输出单缓冲 | 0.086162 ms | 最大 tile 配单缓冲：输出 DMA 与向量工作串行化 |
| v8 | rows=4, stages=2 | 0.080988 ms | 流水过浅 |
| v7 | rows=4, stages=4 | 0.082930 ms | 反向确认 v5 的 stages=3 不是运气 |
| v17 / v25 | y 双→三缓冲 | 0.080580 / 0.073510 ms | **输出缓冲深度 2 是上限**，再深有害 |
| v22 | 读 `NORMAL_PREF` + 写 `NORMAL_RED` | 0.069120 ms | 异族策略组合不优于全 FV |
| v19 / v20 | 读 `NORMAL_LV` | 0.069899 ms | **微基准排名不可搬**（R66 定理） |
| v26 / v27 | rows=2, stages=4 / 6 | 0.079967 / 0.084178 ms | **细 tile 即便在快 DMA 下仍然更差** |
| v28 | 每行 8 累加链 | 0.072535 ms | **4 条是最优**；多链的代价高于短依赖链的收益 |
| v29 | x 五缓冲 | 0.080879 ms | **x 深度 4 是上限** |
| v30 / v31 / v32 | 手动 `order`/`stage` | 0.0689–0.0743 ms | **自动流水调度优于手动**，无需再试 |
| v10/v11/v13/v14/v14b | 两阶段 + 共享缓冲 | 0.0733 ms（**无效**） | **跨相位共享状态这一整类构造**（真实竞争） |

### 6.2 被排除的架构假设

| 假设 | 结论 | 代价 |
|---|---|---|
| cube/matmul（`T.gemm` 求 `X·Xᵀ` 对角线） | 设备编译失败；官方 `example_gemm.py`（CANN 示例，非本仓库）同样失败 → 工具链不兼容，非用法问题 | 一次深挖 |
| bf16 混合精度 FMA（`vmula`） | 不存在 | 探针即否 |
| `vsqz`/`vusqz` 作为平方和原语 | 不是 | 探针即否 |
| `annotate_unlimit_memory` 放宽 UB | 不放宽 launch 检查（286720 B 必拒） | 一次门控 |
| rows=8 | UB 硬上限 271360 B > 256 KiB | 一次门控 |
| 更大的 DMA 块 | **一旦 L2 策略设置就失效**（32 KB 1.757 vs 64 KB 1.725 TB/s） | 一次微基准，**避免了一次大重写** |
| 模拟器作为离线神谕 | **对硬件 12% 的差异报出完全相同的周期数** | **3 轮** |

### 6.3 这笔账怎么算

三项"贵"的否定，三项都买到了东西：

- **R59 的竞争（1 轮）** 排除了整类共享缓冲构造，并直接催生 R60 的正解。**最便宜且最高价值。**
- **R64 的"地板"（1 轮，后被 R65 推翻）** 看似浪费，但它把注意力**精确地**逼到了"路径还能不能更快"，
  而 R65 的答案是"能，2.7×"。**没有 R64 的量化，R65 的发现不会有那么强的对比度。先错后对，净收益为正。**
- **R81–R83 的模拟器（3 轮，纯否定）** 代价最大，但它是 R84 能一次命中的**前提**：
  如果能离线测，团队会继续在模拟器里找；正因为证明了"没有神谕"，才转向测量机器。

**唯一纯亏的是 R68–R80 的封口方式。** 否定本身是对的，但把它写成"杠杆穷尽"，
使一个**从未被质疑的前提**（网格 = 64）获得了"已经检查过"的地位。
这 12 轮的代价不是浪费的测量，而是**一次过早的归纳**——
它让一个 20 秒的算术（128 MB 工作集 vs 128 MB L2）和一个数小时的探针（核数 = 56）
在 74 轮里都没有被提上日程。

---

## 7. 撤回的结论

| 轮 | 当时的声称 | 被哪一轮杀掉 | 为什么 |
|---|---|---|---|
| **R6 / R7** | "两阶段向量算法有约 0.8× 的结构上限" | 该 artifact 自带的 `review_correction` | profiler 利用率不是硬上限的证明；测法未充分配对 [quoted]。**R77 实测 0.8502、R84 实测 1.1883 均已超过它** |
| **R44–R48 期间** | prefill 0.53–0.62× 是真实水平 | **R48** | 那些读数是在**自家 ForgeKernel run 占同一张卡**时测的，已作废 [measured: `evidence/accept-clean-r48.json` 的 `correction_of_earlier_numbers`] |
| **R49–R51** | "调度/分区是杠杆" | R58 / R67 | 五变体全在噪声内；候选时长只在 0.079–0.083 ms 内动 [measured: `evidence/r58-variant-table.json`] |
| **R54** | 连续分区把 prefill 从 0.757 提到 0.825（**+8.9%**） | **R58** | 候选只从 0.080056 → 0.079435 ms（**0.8%，噪声**）；差异来自**厂商窗口波动**（0.059085 → 0.064479，+9%）。**动的是分母** [measured: `evidence/r58-correction-partition-claim.json`] |
| **R59** | 两阶段共享缓冲内核 0.0733 ms 代表约 8% 收益 | **R59（同轮）** | 该内核有**真实竞争**（rows=2 全尺寸错、batch=128 不确定）。0.0733 ms 不得作为性能结论 [measured: `evidence/r59-two-phase-race.json`] |
| **R64** | 拷贝路径**触底**于 1.7 TB/s 聚合，128 MB 需 0.078 ms | **R65** | 它测的是**未设策略**的路径。显式写策略下 3.08 TB/s（2.7×），v18 候选 0.0648 ms [measured: `evidence/r65-l2-breakthrough.json`] |
| **R66 之前** | 微基准可以给 L2 策略排序 | **R66**，并在 **R84** 被量化 | 读路径微基准冠军 `NORMAL_LV`（2.351 TB/s）在真内核里更慢；R84 进一步发现它在 decode 上 **30/30 数值错误** [measured] |
| **R61 / R63 / R67 措辞** | "候选绝对时长是稳定量" | **R74** | 第三个干净窗口给出 0.078896 ms 对 0.065149 / 0.069417，**21% 跨度**；厂商同向漂移。**比值才是稳定量**（0.771–0.813，5.5% 跨度）[measured: `evidence/r74-variance-correction.json`] |
| **R80** | 已识别优化杠杆**全部系统测试并穷尽** | **R84** | 网格尺寸**从未变过**（v4–v23 恒为 64 块，芯片是 56 核）；读路径 `NOTALLOC_CLEAN`/`NOTALLOC_KEEP` **从未在内核门控**。两者一直可 reach [measured: `evidence/r80-closing-record.json` 的 `withdrawn` 字段] |
| **R81** | 模拟流里的 `SIMT_*` 暗示内核被降级到 SIMT 路径 | **R82** | `device_kernel.asc` 含 93 处 `simd_inst` 调用，执行 13925/15360 = 91% 预期向量操作。`SIMT_*` 是 camodel 命名约定 [measured: `evidence/r82-sim-correction.json`] |

**读这张表的方式。** 10 条撤回里：**5 条是测量口径问题**（R48、R54、R58 类、R74、R66 之前），
**2 条是归因层级问题**（R6/R7 的"上限"、R64 的"地板"），**1 条是范围归纳问题**（R80），
**1 条是工具误读**（R81），**1 条是同轮自查**（R59）。

**没有一条是"算错了"。** 这个分布本身就是结论：**这个项目失败的方式不是算术错误，
而是把某一层的结论外推到了另一层。** R54 是这条规律的教科书案例——
分子（候选时长）动 0.8%，分母（厂商窗口）动 9%，而人读的是比值。它之所以值得单独记住，
是因为它**当时看起来像一次真正的胜利**。

---

## 8. 测量纪律：为什么最后这个结论可信

### 8.1 门的判据（事先声明）

```
MATCH   = 点估计 >= 1.0  且  配对 bootstrap 95% 下界 >= 0.98
EXCEED  = 下界 > 1.0
FAIL    = 95% 上界 < 1.0        <- 板上钉钉地在噪声之外更慢
RETEST  = 其余                    <- 证据不足以判定，不是失败
```

[quoted: `acceptance-protocol.md` §3.3]

三条附属规则决定了这个项目的形状：

- **可用窗口 < 9 → RETEST，永不 FAIL。** 证据不足既不能判过也不能判失败。
- **quiet 窗口必须 ≥ 9 个且 ≥ 70%。** 理由是门自己的原文：被剔除的争用窗口**不是随机剔除的**，
  所以它们不构成无偏样本。
- **P90/P10 > 1.25 → RETEST。** 窗口之间自相矛盾时不能当结论。

0.98 是**事先声明的 2% 工程等价容忍**，不是看到结果后调的。

### 8.2 配对是唯一有效的比较

这台机器的绝对设备时间跨 run 漂移达 **2.5×**（同一厂商内核实测 0.0206–0.0514 ms）
[quoted: `acceptance-protocol.md` §3.1]。任何"这次 0.021、上次 0.051"的对比都是伪结论。

R74 把它收紧到 prefill：**两侧绝对时长同向漂移约 ±20%，因此配对比值才是稳定量。**
这条规则直接决定了本文怎么写 prefill：

**prefill 报告为 1.19–1.25 的区间，不是单点。** 采用 60 配对那次（**1.1883**）作 headline，
因为 **100% quiet 是证据质量最高的**，不是因为它数字最大——**它反而是两者中较低的一个**。
把 1.2425 当作"结果"属于挑选（cherry-picking）；把两者平均则是在编造一个没人测过的数。
**两者都不做。**
[measured: §10.1 五份 artifact；判断依据 `core-count-and-l2-breakthrough.md` §4.3]

### 8.3 轨迹本身就是证据：五次门控尝试

同一份候选**字节**，五次门控尝试。点估计几乎不动，**变的只是证据质量**：

| 运行 | 卡 | 配对 | prefill 比值 | prefill quiet | decode 比值 | decode quiet | **run 级** |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 12 | 1.2088§（4/12） | 33.3% | **1.6396** [1.5561, 1.7328] | 12/12（100%） | **RETEST** |
| 2 | 1 | 30 | 1.2459§（17/30） | 56.7% | 1.6913§（20/30） | 66.7% | **RETEST** |
| 3 | 1 | 45 | **1.2425** [1.2188, 1.2642] | 42/45（93.3%） | 1.6288§（30/45） | 66.7% | **RETEST** |
| 4 | 1 | 80（仅 decode） | —（未跑） | — | **1.7144** [1.6699, 1.7448] | 76/80（95.0%） | **EXCEED** |
| 5 | 1 | 60（仅 prefill） | **1.1883** [1.1833, 1.1933] | **60/60（100%）** | —（未跑） | — | **EXCEED** |

**§** = **本文从 per-window `speedup` 数组自行计算的中位数**。该 shape 当时判 RETEST，
门的 `pointSpeedup` 字段被**有意留空**，所以这三个数**不是门的输出，是本文的算术**。
这是全文唯一一类不是门直接给出的数字，在此明确标注（另见 §11.2）。

**这张表是论证本身。** prefill 读 1.2088 → 1.2459 → 1.2425 → 1.1883，而它的 quiet 比例走
33% → 57% → 93% → 100%。decode 读 1.6396 → 1.6913 → 1.6288 → 1.7144，quiet 比例走
100% → 67% → 67% → 95%。

**如果任何一次 RETEST 是真实失败，比值会保持不变而 quiet 比例上升时结论不变；
但事实是两个正式 EXCEED 恰好出现在 quiet 比例越过 70% 门槛的时刻。**
这是"**门卡在证据上、不是卡在内核上**"的签名。

门对 30 配对那次给出的 `why` 原文（保留，因为它记录了判定规则）：

```
8192x4096: 17 quiet window(s) of 30 (56.7%) is not an unbiased sample: 9 and 70% are both required,
because the windows excluded for contention are excluded systematically, not at random
```

**它是关于采样的话，不是关于内核的话。**

### 8.4 争用会双向翻转判定

争用不是保守误差。受控正对照实测（同一**未改动**候选，干净卡 vs 同卡 99% AICore 负载）：

| 形状 | 干净 | 争用 | 变化 |
|---|---|---|---|
| 8192×4096 点估计 | 0.3224 | **0.6615** | **×2.052** |
| 64×4096 点估计 | 0.9939 | 0.8273 | **−17%** |

[measured: `acceptance-protocol.md` §3.8]

**偏差方向随 shape 反转**，所以 2.052× 的膨胀足以让真实 0.478 的实现被报成 0.98、越过 MATCH 门槛。
这就是"contended 窗口一律剔除、quiet 不足 70% 判 RETEST"这条规则的来源。

### 8.5 单形状运行是结论的一部分

**五次尝试中，达到 run 级 EXCEED 的只有两次，而那两次都是单形状运行。**

原因在门的结构里：门要求**每个 shape 各自** ≥ 70% quiet，任一 shape 不达标则整个 run 判 RETEST。
所以 **run 级判定是一个合取**。把一个便宜但易被打断的形状（decode，4 µs）和一个昂贵的形状
（prefill，50–70 µs）放进同一个 run，**脏的那个会系统性地把干净的那个的判定拖下来**：

- **12 配对那次**：prefill 4/12 quiet → 拖掉了一个 12/12 quiet、点估计 **1.6396** 的 decode。
- **45 配对那次**：decode 30/45 quiet → 拖掉了 **shape 级 EXCEED 1.2425** 的 prefill。

两次都是"**一个形状赢、一个形状被拖累**"。这不是候选的问题，是 run 设计的问题。
**结论：验收按形状分开跑**——这正是两次 headline 的取得方式，也是本文严格区分
shape 级与 run 级判定的原因。

### 8.6 筛查级数字不得当作结果

这是本项目最昂贵的一课，因此单列。

ledger 用 `claim_strength` 区分证据强度。本文严格保留这一区分：

| 强度 | 含义 | 本文如何处理 |
|---|---|---|
| **measured-paired** | 同窗口配对、过门 | 可直接引用为结论 |
| **measured-single** | 单进程交替或单次探针 | **只作排序/诊断**，不写作"结果" |
| **derived** | 对实测值做算术 | 标注算式 |
| **quoted** | 转述，无原始 artifact | 标注来源，不补充细节 |

**9 条 `measured-single` 记录是筛查级，从未被门认证**（R2、R3–5、R6、R7、R47、R49、R55、R64、R73）。
本文引用它们时一律保持这一区分。理由不是形式主义——**上一会话最糟的时刻全都是这一类**：

- **R54** 把一次由厂商窗口漂移造成的比值上升（+8.9%）读成了候选改进，而候选只动了 0.8%。
- **R49/R50** 的单进程筛查曾预测 **+12%**，门控给出 **0**。
- **R66 之前** 的 L2 策略排序完全基于微基准，而微基准冠军在真内核里更慢、
  且在 decode 上 30/30 数值错误。

**规则：任何未被门认证的筛查数字，不得以"改进"的形式出现在结论里。**
一个筛查数字最大的用途是**决定下一个门控跑什么**，不是**代替那一次门控**。

### 8.7 这套纪律防不住什么

诚实列在 `acceptance-protocol.md` §10，本文摘要：

- 真实内核里藏了未声明的近似（只有精度表能挡，而精度表只覆盖两形状 × 三族输入）。
- cache 命中使时长偏乐观（`--cache` 必须为空目录正是为此）。
- 窗口有 witness 但仍被同卡大流量任务干扰到超 ±8%。
- 候选只在当前 shape 快（所以逐形状判定、禁止几何平均）。
- 反作弊依赖"归档自包含"这一契约。

---

## 9. 有多少是关于硬件的，有多少是关于内核的

### 9.1 代码事实

v23 → v33 的 diff 只有四处，**本文直接从两份源码核对过**（不是转述文档）：

| # | 改动 | 性质 |
|---|---|---|
| 1 | `N_CORES` **64 → 56** | **一个 host 侧整数常量** |
| 2 | 均匀切分 → **ragged 切分**（`start = core*nbase + min(core, rem)`） | 由 #1 派生的算术 |
| 3 | 新增 `LOOP_MODE` 常量，默认 `"dyn"` | 一个开关，两模式均过正确性 |
| 4 | `L2_READ_CTRL = "NOTALLOC_CLEAN"`，写保持 `NORMAL_FV` | **一个参数（load enum 5）** |

`md5` 不同（v23 `0a1e6e2c…` / v33 `b07a9530…`），行数也不同（209 vs 284），
但**差异全在 docstring、注释与上述四处**。SimdVF 主体、6 级 `vcadd` 归约树、每行寄存器累加、
4 行展开、staging 深度——**是 v23 的，逐字节相同**。
[measured: 源码 diff，`candidates/rmsnorm_simdvf_v23/kernel.py` vs `candidates/rmsnorm_simdvf_v33/kernel.py`]

### 9.2 这个事实的含义

**80 轮内核工作 vs 4 轮质疑一个常量。**

前 80 轮改了 40 个候选目录、测了 R80 列出的 9 条杠杆、否定了 7 类架构假设，
把候选时长从 0.0801 压到 0.0651 ms（**−19%**），而 prefill 判定**始终是 FAIL**。
后 4 轮没有改内核主体，把同一个 artifact 从 **0.771–0.850** 推到 **1.19–1.25**。

**这不是"优化白做了"。** 前 80 轮产出的是**必要条件**：

- 没有 v23 的 body，56 块 + CLEAN 也不会有 1.19（几何改动单独只到 ~1.01）。
- 没有 R65 的 L2 发现，没人会想到去试 CLEAN。
- 没有 R66 的"微基准不可搬"，CLEAN 会在微基准阶段被 `NORMAL_LV` 挡掉。
- 没有 R74 的"比值才是稳定量"，0.771→0.850 的窗口波动会被误读成进展。

**但前 80 轮也产出了一次昂贵的错误归纳。** 它以"杠杆穷尽"结束，
而穷尽的清单里**不含网格尺寸**——因为它从来没被当成一个杠杆，它被当成了**环境常量**。

### 9.3 下次先看哪里

把这个教训一般化：**被当成环境常量、实际上从未验证过的量，其优先级应高于任何内核内改动。**

本项目里这类量有三个，一个都没在前 80 轮内被查：

| 量 | 当时的状态 | 验证成本 | 实际影响 |
|---|---|---|---|
| **AIV 核数 / 网格块数** | 假定 64，从未测 | **一次阶跃探针（数小时）** | **−39～48% 时间** |
| **L2 容量 vs 工作集** | 128 MB，已知但未与策略关联 | **一次算术（20 秒）** | **1.103×** |
| VF 归纳变量位宽 / Hardware Loop | 未查 | 读生成代码，**不需要设备** | 未知，**仍未测** |

第一条的验证成本是数小时的探针，而它被跳过的时间是 74 轮。

**具体的行动规则：** 在优化一个内核之前，先用探针确认这台机器的**并发单元数、缓存容量、
存储层级带宽**三个常量，并把它们写成**被测过的事实**而不是平台文档的抄写。
这不是性能工作的一部分——**这是它的前置条件。**

### 9.4 也要说清的另一面

`1.19–1.25` 不是"到顶了"。同一次改动在规模泛化扫描里给出 **1.24×–1.68×** 的提升幅度，
且 **8192×2048 在 56 块下仍只有 0.879**（低于 0.98）——所以这个修复普遍有益但**不是万能**
[measured: `core-count-and-l2-breakthrough.md` §5，每格单次配对，方向强、单点近似]。
**认证形状仍然只有 8192×4096 与 64×4096。**

---

## 10. 证据索引

### 10.1 逐轮证据（ledger 所引，全部存在于磁盘）

**Era A：** `evidence/simdvf-r1-reduce-2026-11-02.json`、`evidence/simdvf-r2-api-2026-11-02.json`、
`evidence/simdvf-r3-r5-kernel-2026-11-02.json`、`evidence/simdvf-r6-pipes-2026-11-02.json`、
`evidence/simdvf-r7-decode-2026-11-02.json`、`evidence/prefill-ceiling-2026-11-02.json`、
`evidence/accept-r47-structural-r47.json`、`evidence/accept-clean-r48.json`、`evidence/prefill-gap-r49.json`、
`evidence/accept-v3-prefill-r50.json`、`evidence/accept-r51.json`、`evidence/r52-ublimit-and-contention.json`、
`evidence/r53-authenticity-falsepositive.json`、`evidence/r54-partition-and-gate-fix.json`、
`evidence/r55-environment-and-variants.json`、`evidence/r56-v5-precision-coverage.json`、
`evidence/r58-correction-partition-claim.json`、`evidence/r58-variant-table.json`、
`evidence/simdvf-feasibility-2026-11-02.json`

**Era B：** `evidence/r59-two-phase-race.json`、`evidence/r60-v15-race-free.md`、`evidence/r61-v15-gated.json`、
`evidence/r62-dma-bound.json`、`evidence/r63-v16-dual.json`、`evidence/r64-copy-floor.json`、
`evidence/r65-l2-breakthrough.json`、`evidence/r65-write-path-and-ceiling.json`、
`evidence/r66-v20-policy-fit.json`、`evidence/r66-v21-best.json`、`evidence/r67-v23-best.json`、
`evidence/r68-no-better-variant.json`、`evidence/r69-decode-repeat.json`、`evidence/r70-manual-pipeline.json`、
`evidence/r71-v23-input-families.json`、`evidence/r72-repro-and-chunk-limit.json`、
`evidence/r73-mixed-traffic.json`、`evidence/r74-variance-correction.json`、`evidence/r76-audit.json`、
`evidence/r77-fourth-window.json`、`evidence/r79-final-verification.json`、`evidence/r80-closing-record.json`

**Era C：** `evidence/r81-simulator-feasibility.json`、`evidence/r82-sim-correction.json`、
`evidence/r83-simulator-not-discriminating.json`

**R84 的五份门控报告（§1 的正式 EXCEED 所依赖的全部证据）：**

| 文件 | 内容 | run 级 | shape 级 |
|---|---|---|---|
| `evidence/accept-rmsnorm-v33-12pair-2026-11-03.json` | 12 配对，device 0 | RETEST | decode EXCEED 1.6396；prefill RETEST 4/12 |
| `evidence/accept-rmsnorm-v33-30pair-2026-11-03.json` | 30 配对，device 1 | RETEST | 两者皆 RETEST（prefill 17/30、decode 20/30） |
| `evidence/accept-rmsnorm-v33-45pair-2026-11-03.json` | 45 配对，device 1 | RETEST | prefill **EXCEED 1.2425**；decode RETEST 30/45 |
| `evidence/accept-rmsnorm-v33-prefill-60pair-2026-11-03.json` | **60 配对，仅 prefill** | **EXCEED** | **prefill EXCEED 1.1883** |
| `evidence/accept-rmsnorm-v33-decode-80pair-2026-11-03.json` | **80 配对，仅 decode** | **EXCEED** | **decode EXCEED 1.7144** |

### 10.2 事实骨架

- `evidence/optimization-ledger-2026-11-03.json` — 39 条逐轮记录；本文时间线与撤回表的事实来源。

### 10.3 叙述与分析文档

- `core-count-and-l2-breakthrough.md` — R84 证据文档（核数、L2 机制、安全表、泛化扫描）。
- `final-status.md` — 最终状态；**含 v23 历史记录，完整保留不删改**。
- `perf-report.md` — **本文取代其轨迹叙事地位，但保留不删不改**；它记录了 2026-11-02 14:2x 的真实认知。
- `plan-40-rounds.md` — 计划与 §8 逐轮日志（含 R46–R58 的叙述与更正）。
- `d3-assessment.md` — R63 的"当前手段可否达 parity"评估，含被排除手段表。
- `acceptance-protocol.md` — 验收协议 12 节，含协议自身的失效模式。
- `pitfalls.md` — 实施风险与验收登记表。
- `devkit-lessons-reg-compute.md` — `02_reg_compute` 研究；§0 结论为"对内核级改动已穷尽"。
- `devkit-lessons-memory.md` — `04_memory_access` 研究；§2.4 记录"只有 4/12 load、3/9 store 策略被门控过"。
- `archive-manifest.json` — 31 件 artifact 的 sha256 清单（第 84 轮，候选 v33）。

### 10.4 候选

- `candidates/rmsnorm_simdvf_v33/kernel.py` — **最终候选**（md5 `b07a95300bf895a383060f7226a8ed1f`）。
- `candidates/rmsnorm_simdvf_v33/NOTES.md` — 候选说明。
- `candidates/rmsnorm_simdvf_v23/kernel.py` — 前一最终候选（md5 `0a1e6e2cab9a26ffabb5b04ddb1ee667`），
  其 body 即 v33 的 body。

---

## 11. 出处与限制

### 11.1 什么算实测

- **[measured]** — 五份 `accept-rmsnorm-v33-*` 的 `verdict` / `timing` / `precision` 字段
  （本文写作时重新解析了这些 artifact，不是转述）；阶跃探针与固定总量扫描；64 次/策略的正确性表；
  20 配对的 L2 两臂比较；`NOTALLOC_KEEP` 0.6600 vs 0.6626；每个被否定变体的候选时长；
  争用正对照的 2.052×；跨 run 绝对时长 2.5× 漂移。
- **[derived]** — 128 MB = 恰好 L2 容量；64 vs 56 块的 39–48% 惩罚；36/32 ragged 切分算术；
  CLEAN vs FV 的 1.103× / 1.009×；1.97 vs 2.42 TB/s 与 20% 时间差的对应。
- **[quoted]** — 门判据原文、`why` 字符串、`plan-40-rounds.md` §8 的叙述、devkit 文档要点
  （经两份研究文档转述；本文未直接读 asc-devkit 源码）。
- **[spec]** — *为什么* 8 个掉队块要花整整一波而不是 12.5% 的工作；*为什么* DROP 是形状依赖的
  （失败本身是实测的，行状态机制不是）；"脏行写回插入读流"是 devkit 的机制套用于我们的数字，
  不是对写回的直接观测。

### 11.2 本文自身引入的数字

**三处，全部是 quiet 窗口 `speedup` 数组的中位数，且都不是门的输出**
（门在该 shape 判 RETEST 时有意留空 `pointSpeedup`）：
prefill **1.2088**（12 配对，4 quiet）、**1.2459**（30 配对，17 quiet）、
decode **1.6288**（45 配对，30 quiet）。

它们与门对同 run 内 shape 级 EXCEED 的报数方法一致，但**不是门签发的数**。
除此之外本文没有自行推导任何数字。

同理，ledger 中 `claim_strength = "measured-single"` 的 9 条记录是**筛查级**，不是门认证级；
`claim_strength = "quoted"` 的 R9–R45 是 log-only 分组、**无原始 artifact**，本文未为其补充细节。
详见 §8.6。

### 11.3 机器是共享的

所有测量都在有其他租户的卡上完成。已用配对 + 剔除 + 双向正对照缓解，但**无法消除**——
这正是 §8.3 那些 RETEST 的成因。**卡 0 健康状态为 `Alarm`**，正式测量用卡 1
[quoted: `perf-report.md` §7；`final-status.md` §9]。

### 11.4 prefill 是一个区间

**prefill ≈ 1.19–1.25，始终 EXCEED，方向从未动摇。**
headline 用 1.1883，因为它是 60/60 全 quiet 的单形状 run，是证据质量最高的一次——
**它也恰好是两次里较低的那个数**。这一点必须和结论一起引用，
否则会出现"数字越大越可信"的误读。

### 11.5 未确立的东西

- 读路径上 {`NORMAL_FV`, `NOTALLOC_CLEAN`, `NOTALLOC_DROP`, `NORMAL_LV`} 之外任何策略的行为。
- **写路径 `WT_SHARE_FIRST_VICTIM`（store enum 12）——从未门控**，且它是另一条"让脏行不存在"的
  文档化途径。
- **`NORMAL_PERS` 用于 W 拷贝（8 KB）——从未门控。**
- **TileLang 编译器缺口**：VF 归纳变量 `int32_t` 阻断 Hardware Loop；`#pragma unroll N` 在 VF 体内
  可达但从未被 553 个缓存内核中的任何一个行使。**两者都不需要设备即可澄清，至今未测。**
- 泛化扫描之外的形状。
- **任何绝对 HBM 带宽结论**——本仓库所有微基准速率都是含 L2 的。
- 模拟器的 per-pipe 数字（514 cycles/load 疑为 stall 归因而非单指令成本，未做受控 A/B）。

### 11.6 一句话

**这次翻案不是"最后找到了巧妙的优化"，而是"最后去测了三个被当成环境常量的数"。**
前 80 轮的否定大部分是对的，它们把可选空间压到只剩两个未经检验的前提；
错的是把那个缩小后的空间当成了全部空间。
