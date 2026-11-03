# 优化点目录：每个优化点 -> 实现它的 kernel 版本

**作者**：lead。**日期**：2026-11-03。**依据**：`evidence/optimization-ledger-2026-11-03.json`（39 条，R1–R84）、`core-count-and-l2-breakthrough.md` §7、各候选目录、原会话事件。

标签：**[measured]** / **[source]** / **[inferred]**。

> **一条免责**：40+ 个候选目录里只有 v33 带 `NOTES.md`，其余版本的逐版说明散在 ledger 与 `forgekernel-ascend950pr-work/experiments/` 里。下表凡标注 **[inferred]** 的版本归属，是从 ledger 的 artifacts 字段反推的，不是从该版本自己的文档读出。

---

## A. 先分两类（这是这份目录最重要的一栏）

操作者问的是**优化点**。但 R1–R84 里有相当一部分改动是**正确性修复**，不是提速。混在一起会误导。所以每条都标注 `kind`：

- **PERF** = 让它更快
- **CORRECT** = 让它算对（这类即使不提速也必须留）
- **METHOD** = 改的是测量/判定方法，不是 kernel

---

## B. Era A（R1–R58）：从 SIMT 模拟换到 SIMD，再把 DMA 流水喂饱

| # | 优化点 | kind | 版本 | 代码落点 | 效果 | 存留 |
|---|---|---|---|---|---|---|
| A1 | **SimtVF → SimdVF**：放弃 `T.SimtVF`(SIMT 线程模拟)，改用 `with T.SimdVF()` + `T.simd.*` 微指令 | PERF | cw4 → v1 | `with T.SimdVF()` 替换 `with T.SimtVF(threads=)`；`vld/vcvt/vmula/vsts` | prefill 0.388× → 0.58× → 0.62× [measured R3-5,R7] | **存活**，全后续版本 |
| A2 | **bf16 载入语义**：`vld(dist="UNPK_B16") + vcvt(part=0)` 把 64 个 bf16 正确解成 f32 | CORRECT | v1 | 每次 `vld` 后接 `vcvt(...,"float32",part=0)` | 逐元素 == torch [measured R1] | **存活** |
| A3 | **w 预转一次**：权重 bf16→f32 在 UB 里预转成 `wf`，而不是每行重复转 | PERF | v3 | `wf=T.alloc_shared((d,),"float32")`，进循环前一次性 `vsts` | 0.539× → 0.581× [measured R3-5] | **存活** |
| A4 | **rows 分组（每核多行）**：一次 DMA 搬多行，摊薄启动与 W 复用 | PERF | v2/v3 | `_rows_per_core()` 选 8/4/2/1；`rows*d` 的 `x_ub` | rows 1→4 得 1.76× [measured R6] | **存活**（v33 用 rows=4 于 prefill） |
| A5 | **DMA 与 VF 分离**：`T.copy` 必须在 `with T.SimdVF()` 之外 | CORRECT | v1 | DMA 与 VF 块顺序组合 | 见 `simdvf-dma-plan.md` [source] | **存活**（硬约束） |
| A6 | **`T.Pipelined` + `annotate_buffer_versions`**：多版本 UB 缓冲做软流水 | PERF | v2/v3 | `T.annotate_buffer_versions({x_ub:stages,y_ub:yver})`；`num_stages=stages` | 阶段深度扫描定 (4,2)/(2,1) [measured R6,R9-45] | **存活** |
| A7 | **多累加链**：4 条独立 `acc` 链并行累加平方和，掩盖 vmula 延迟 | PERF | multiacc/v2x | `acc0_0..acc0_3` 各 `vdup(0)`，循环内四条 `vmula`，末尾 `vadd` 归并 | 见 R9-45；R47 因卡被自占无法认证 | **存活**（v33 的 rows==4 分支即 4 链） |

## C. Era B（R59–R80）：两相结构 + L2 策略，以及一大片否定

| # | 优化点 | kind | 版本 | 代码落点 | 效果 | 存留 |
|---|---|---|---|---|---|---|
| B1 | **两相共享缓冲是竞态**（必须先修） | CORRECT | v10/v11/v13/v14 | 两相共享 buffer | rows=2 全尺寸错、batch=128 5 次结果不一 rel=1.0 [measured R59] | **否决**（这是负结果，见下） |
| B2 | **源码级 unroll + 每行独立寄存器消除竞态** | CORRECT | v15 | 每行独立 `alloc_var` 寄存器，不用共享缓冲 | v15 全尺寸确定正确；prefill 0.0794→0.0755 ms [measured R60-61] | **存活** |
| B3 | **写路径 `NORMAL_FV`**：UB→GM 显式 cache 策略 | PERF | v18 | `T.copy(y_ub, Y, ...)` 带 `NORMAL_FV` | 1.13 → 3.08 TB/s（2.7×）；v18 0.0648 ms [measured R65] | **存活** |
| B4 | **全 DMA 路径 `NORMAL_FV`** | PERF | v21/v23 | 读、写、W 载入统一策略 | v23 两形状最佳 [measured R66-67] | **存活**（v33 写路径仍是它） |
| B5 | **stages=4 + 输出缓冲** | PERF | v23 | `stages,yver=(4,2)` | decode 1.3726 [measured R67] | **存活** |
| B6 | **微基准排序不可迁移**：读微基准里 `NORMAL_LV` 赢，进内核反而更慢 | METHOD/负 | — | — | NORMAL_LV 让 kernel 更慢 [measured R66] | **教训** |
| B7 | **系统否定一批杠杆**：rows=2/8、stages=5/6、8 链、y 三缓冲、读写异策略、手动 stage 列表 | 负结果 | v28/v29/v32 | — | **没有一个**改善 gated prefill（~0.75）[measured R68-70] | **全部否决** |

## D. Era C（R81–R84）：答案在启动几何与读路径，不在内核体

| # | 优化点 | kind | 版本 | 代码落点 | 效果 | 存留 |
|---|---|---|---|---|---|---|
| C1 | **64 块 → 56 块**：芯片只有 56 个 AIV 核，64 块要跑近空的第二波 | PERF | v33 | `N_CORES = 56` | step-detection 探针 grid 1..56 平坦、57 跳 +82%；prefill 首个 EXCEED 的必要条件 [measured R84] | **存活**（v33/v34） |
| C2 | **ragged split**：2048 tile 不被 56 整除，改成 36 基 + 前 32 核多 1 | CORRECT | v33 | `nbase/rem/start/cnt` 四行 | 无此则 56 核不可用（8192%224=128≠0）[measured] | **存活** |
| C3 | **读路径 `NORMAL_FV` → `NOTALLOC_CLEAN`**：X tile 只读一次，不该占 L2 名额 | PERF | v33 | `L2_READ_CTRL = NOTALLOC_CLEAN`（写路径不动） | prefill FAIL→EXCEED；1.103× 的来源 [measured R84] | **存活**（v33/v34） |
| C4 | **`NOTALLOC_KEEP` 未复现 devkit 的赢**：两个最快的 L2 策略会**破坏正确性**，明令禁再试 | 负结果 | — | — | `core-count-and-l2-breakthrough.md` §3 | **红线** |
| C5 | **模拟器无法区分 12% 差异**：v23 vs v28 在仿真里 bit-identical | METHOD/负 | — | — | 无离线 oracle，只能上真机 [measured R83] | **教训** |

## E. 本会话新增

| # | 优化点 | kind | 版本 | 代码落点 | 效果 | 存留 |
|---|---|---|---|---|---|---|
| E1 | **vcadd 链坍缩**：`T.simd.vcadd` 是全归约且幂等，6 次链式＝5 条死指令；压成 1 次 | PERF | **v34** | 5 处（kernel.py:141,161,181,201,261 各两行→两行），30 次 `vcadd`→5 次 | prefill +0.66% [1.0053,1.0093] 24/24；decode +0.60%；**两形状 bit-identical** [measured] | **新增候选** |

---

## F. 汇总：真正贡献了性能的只有 6 条

把 39 条里所有 PERF 且被判为有效的挑出来，按贡献排序：

1. **A1 SimtVF→SimdVF** —— 换范式，0.39×→0.62×（最大的一步）
2. **C1 56 核** —— 消除第二波，prefill 跨过 1.0 的关键之一
3. **C3 NOTALLOC_CLEAN 读路径** —— 1.103× 的直接来源，prefill 跨过 1.0 的另一半
4. **B3/B4 NORMAL_FV 写路径** —— 2.7× 写带宽
5. **A3 w 预转 + A4 rows 分组 + A6 流水** —— 0.58→0.72 的渐进项
6. **E1 vcadd 坍缩** —— +0.6%，零风险

**其余大多是负结果或方法修正**，而且这是对的：R68–R70 系统性否定 8 个杠杆、R64 的 floor 误判被 R65 推翻、R80 过早宣告穷尽——这些负结果的密度本身就是这条轨迹的特征。**一个只列成功的目录会让人以为调参很顺，事实相反：39 条里有效的只有 6 条。**