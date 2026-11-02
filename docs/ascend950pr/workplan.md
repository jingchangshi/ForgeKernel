# 后续工作计划：优化 Ascend950PR 上的 RMSNorm 算子

状态：初稿　更新：2026-11-02　适用仓库：ForgeKernel（分支 main）

本文件是 [性能报告](perf-report.md) 的后续行动篇。报告说明现状与瓶颈，本文件说明接下来怎么把它推到达标。

---

## 0. 起点（实测，不含乐观假设）

| 形态 | 候选 | 厂商 | 比值 | 判定 |
| --- | --- | --- | --- | --- |
| 8192x4096 预填充 | 163 us | 51 us | **0.32x** | FAIL，确定慢约 3.1 倍 |
| 64x4096 解码 | 3.74 us | 3.71 us | 0.99x | RETEST，已到固定开销地板 |

**瓶颈已定量**：向量管道 84.9%（厂商 80.7%）已满；差距全在数据搬运——mte2 32.0% vs 64.2%、mte3 12.6% vs 44.9%；有效带宽 0.8 vs 2.6 TB/s。**解码形态基本无空间，全部精力放预填充。**

### 0.1 一处事实订正：UB 上限

先前文档写“UB 预算约 216 KiB”是**未经测量的猜测**，现撤回。实测方法：在设备上编译并运行一个只含单个共享缓冲的内核，对缓冲字节数做二分（编译本身不校验，运行期 aclrtLaunchKernelWithHostArgs 才校验动态+静态共享内存）。结果：262144 B（256 KiB）通过，263168 B（257 KiB）被拒。CANN include/pto/common/buffer_limits.hpp 对 A5（950PR 代）亦定义 PTO_UBUF_SIZE_BYTES = 256*1024。**结论：本卡每核 UB 可用上限为 256 KiB = 262144 B。**

关于 248 KiB（253952 B）：它低于实测 256 KiB 上限，可能是真实内核在扣除编译器管理的静态共享内存后的**有效动态预算**，但硬件/可用上限我复现到的是 256 KiB。计划按 256 KiB 为硬上限、以 248 KiB 为保守可用值来设计分块。详见 evidence/probe-bf16-2026-11-02.json 的 ub_capacity_measurement 段。

---

## 1. 阶段 0：先让优化闭环真正能跑（硬前提）

当前只验证了“设备 job 能执行”，judge -> calibrate -> curriculum -> student -> KEEP/REVERT **从未跑完一轮**。在 eval 设备化之前做内核优化都是手工、不可复现的。必须补齐：

1. **P5 eval 设备化（最关键）**：fk_dev 设备层（tensor/stream/event/sync/RNG）；float64 距离一律拷回 CPU；problem.json 增加 device 字段。没有它，性能反馈回路无法自动给分。
2. **P8 计时与基线**：逐 shape 厂商基线、漂移重测规则、数学 judge 与 vendor baseline 分离。
3. **P4 gpu_entry 设备变体**：完成 marker 与错误码契约，与 sealed reference 树共用环境/设备/超时。
4. **验收判据**：跑一个平凡变异（只改一个 tile 常量），闭环能自动产出 KEEP/REVERT 并落库。

**出口条件**：一次真实的 student 迭代被完整执行、评分、并与基线比较后给出 KEEP 或 REVERT。

## 2. 阶段 1：让闭环有先验

5. **P7 hw_kb/tilelang-ascend**：把已测得事实写进知识库——管道上限、UB 256 KiB、rows=8 溢出阈值（dyn_ubuf 271360 B）、row-tile 与 mte2 的关系。这是每轮迭代的先验来源。
6. **P9 候选准入**：可达代码 + 运行期/产物（.aibin）证据，拒绝融合算子代算（允许 torch_npu 注册/事件）。
7. **P6 profiling 降级**：driver 真正消费 ncu_available，无 NCU 不提交不重试；NPU profiler 另行开发。

## 3. 阶段 2：核心优化迭代（按证据强度排序）

| 实验 | 做法 | 目标指标 | 依据 |
| --- | --- | --- | --- |
| E1 copy pipelining | T.Pipelined 多级、双缓冲、异步 copy，喂饱 mte2 | mte2 32% -> 55%+ | 差距主因 |
| E2 UB 分块 | 消除同时驻留 bf16+fp32 副本（rows*d*2 + rows*d*4*2），使更大行批可容纳 | 在 256 KiB 内提高行批 | rows=8 溢出 271360 B |
| E3 store 路径 | 合并/向量化 store，减少 mte3 额外搬运 | mte3 12.6% -> 35%+ | mte3 仅为厂商 28% |
| E4 解码固定开销 | 降低启动/同步开销 | 保持 >=0.99x | 已在 3.7 us 地板 |
| E5 并行宽度 | T.Parallel 的 coalesced_width / VF 切分调优 | vec 保持 85% 前提下提带宽 | 需配合 E1 |

每轮迭代判据（沿用已验证协议）：同窗口配对、msprof 设备时长、逐 shape 判定、剔除 contended/sparse 窗口、bootstrap 区间；KEEP 需在同 shape 上稳定优于上一版，并把结论写入 hw_kb。

## 4. 阶段 3：验收

8. 对最佳候选跑独立验收门（accept_rmsnorm.py）——冷 cache、>=15 对、witness、逐 shape。预填充 MATCH 需点估计 >=1.0 且 95% 下界 >=0.98。

---

## 5. 里程碑与风险

- **M1（阶段 0 完成）**：闭环能对平凡变异产出 KEEP/REVERT。
- **M2（阶段 2 首轮）**：预填充从 0.32x 提到 >=0.5x（E1 单独达成即可）。
- **M3（目标）**：预填充 >=1.0x。

**风险 1（结构性）**：厂商 2.6 TB/s 可能来自我们尚未使用的高效数据通路；若 E1/E2 后仍停在 ~0.6x，说明需要换算法（两趟 vs 一趟、split-K），而非继续调参。这是最可能触发路线变更的点。

**风险 2（测量）**：共享机器噪声大——必须严格配对，禁绝对时间对比。

**风险 3（UB 硬顶）**：UB 上限 256 KiB（实测）。若行批无法在预算内继续增大且 pipelining 收益饱和，预填充可能长期停在 0.6-0.7x。

---

## 6. 建议时间盒

阶段 0（约 1 周）-> 阶段 1（约 3 天）-> 阶段 2 迭代（每轮 1-2 天，先做 E1/E2）-> 阶段 3 验收。阶段 0 是硬前提。# 后续工作计划：优化 Ascend950PR 上的 RMSNorm 算子

状态：初稿　更新：2026-11-02　适用仓库：ForgeKernel（分支 main）

本文件是 [性能报告](perf-report.md) 的后续行动篇。报告说明现状与瓶颈，本文件说明接下来怎么把它推到达标。

---

## 0. 起点（实测，不含乐观假设）

| 形态 | 候选 | 厂商 | 比值 | 判定 |
| --- | --- | --- | --- | --- |
| 8192x4096 预填充 | 163 us | 51 us | **0.32x** | FAIL，确定慢约 3.1 倍 |
| 64x4096 解码 | 3.74 us | 3.71 us | 0.99x | RETEST，已到固定开销地板 |

**瓶颈已定量**：向量管道 84.9%（厂商 80.7%）已满；差距全在数据搬运——mte2 32.0% vs 64.2%、mte3 12.6% vs 44.9%；有效带宽 0.8 vs 2.6 TB/s。**解码形态基本无空间，全部精力放预填充。**

### 0.1 一处事实订正：UB 上限

先前文档写“UB 预算约 216 KiB”是**未经测量的猜测**，现撤回。实测方法：在设备上编译并运行一个只含单个共享缓冲的内核，对缓冲字节数做二分（编译本身不校验，运行期 aclrtLaunchKernelWithHostArgs 才校验动态+静态共享内存）。结果：262144 B（256 KiB）通过，263168 B（257 KiB）被拒。CANN include/pto/common/buffer_limits.hpp 对 A5（950PR 代）亦定义 PTO_UBUF_SIZE_BYTES = 256*1024。**结论：本卡每核 UB 可用上限为 256 KiB = 262144 B。**

关于 248 KiB（253952 B）：它低于实测 256 KiB 上限，可能是真实内核在扣除编译器管理的静态共享内存后的**有效动态预算**，但硬件/可用上限我复现到的是 256 KiB。计划按 256 KiB 为硬上限、以 248 KiB 为保守可用值来设计分块。详见 evidence/probe-bf16-2026-11-02.json 的 ub_capacity_measurement 段。

---

## 1. 阶段 0：先让优化闭环真正能跑（硬前提）

当前只验证了“设备 job 能执行”，judge -> calibrate -> curriculum -> student -> KEEP/REVERT **从未跑完一轮**。在 eval 设备化之前做内核优化都是手工、不可复现的。必须补齐：

1. **P5 eval 设备化（最关键）**：fk_dev 设备层（tensor/stream/event/sync/RNG）；float64 距离一律拷回 CPU；problem.json 增加 device 字段。没有它，性能反馈回路无法自动给分。
2. **P8 计时与基线**：逐 shape 厂商基线、漂移重测规则、数学 judge 与 vendor baseline 分离。
3. **P4 gpu_entry 设备变体**：完成 marker 与错误码契约，与 sealed reference 树共用环境/设备/超时。
4. **验收判据**：跑一个平凡变异（只改一个 tile 常量），闭环能自动产出 KEEP/REVERT 并落库。

**出口条件**：一次真实的 student 迭代被完整执行、评分、并与基线比较后给出 KEEP 或 REVERT。

## 2. 阶段 1：让闭环有先验

5. **P7 hw_kb/tilelang-ascend**：把已测得事实写进知识库——管道上限、UB 256 KiB、rows=8 溢出阈值（dyn_ubuf 271360 B）、row-tile 与 mte2 的关系。这是每轮迭代的先验来源。
6. **P9 候选准入**：可达代码 + 运行期/产物（.aibin）证据，拒绝融合算子代算（允许 torch_npu 注册/事件）。
7. **P6 profiling 降级**：driver 真正消费 ncu_available，无 NCU 不提交不重试；NPU profiler 另行开发。

## 3. 阶段 2：核心优化迭代（按证据强度排序）

| 实验 | 做法 | 目标指标 | 依据 |
| --- | --- | --- | --- |
| E1 copy pipelining | T.Pipelined 多级、双缓冲、异步 copy，喂饱 mte2 | mte2 32% -> 55%+ | 差距主因 |
| E2 UB 分块 | 消除同时驻留 bf16+fp32 副本（rows*d*2 + rows*d*4*2），使更大行批可容纳 | 在 256 KiB 内提高行批 | rows=8 溢出 271360 B |
| E3 store 路径 | 合并/向量化 store，减少 mte3 额外搬运 | mte3 12.6% -> 35%+ | mte3 仅为厂商 28% |
| E4 解码固定开销 | 降低启动/同步开销 | 保持 >=0.99x | 已在 3.7 us 地板 |
| E5 并行宽度 | T.Parallel 的 coalesced_width / VF 切分调优 | vec 保持 85% 前提下提带宽 | 需配合 E1 |

每轮迭代判据（沿用已验证协议）：同窗口配对、msprof 设备时长、逐 shape 判定、剔除 contended/sparse 窗口、bootstrap 区间；KEEP 需在同 shape 上稳定优于上一版，并把结论写入 hw_kb。

## 4. 阶段 3：验收

8. 对最佳候选跑独立验收门（accept_rmsnorm.py）——冷 cache、>=15 对、witness、逐 shape。预填充 MATCH 需点估计 >=1.0 且 95% 下界 >=0.98。

---

## 5. 里程碑与风险

- **M1（阶段 0 完成）**：闭环能对平凡变异产出 KEEP/REVERT。
- **M2（阶段 2 首轮）**：预填充从 0.32x 提到 >=0.5x（E1 单独达成即可）。
- **M3（目标）**：预填充 >=1.0x。

**风险 1（结构性）**：厂商 2.6 TB/s 可能来自我们尚未使用的高效数据通路；若 E1/E2 后仍停在 ~0.6x，说明需要换算法（两趟 vs 一趟、split-K），而非继续调参。这是最可能触发路线变更的点。

**风险 2（测量）**：共享机器噪声大——必须严格配对，禁绝对时间对比。

**风险 3（UB 硬顶）**：UB 上限 256 KiB（实测）。若行批无法在预算内继续增大且 pipelining 收益饱和，预填充可能长期停在 0.6-0.7x。

---

## 6. 建议时间盒

阶段 0（约 1 周）-> 阶段 1（约 3 天）-> 阶段 2 迭代（每轮 1-2 天，先做 E1/E2）-> 阶段 3 验收。阶段 0 是硬前提。
---

## 现状与结论（2026-11-02 更新）

本轮攻关的实际结论（证据见 evidence/ ）：

- 解码 64x4096：已达标。coalesced_width=4 的内核经独立验收门认证为 EXCEED 1.095x（bootstrap95 [1.074, 1.122]），优于厂商约 9.5%。
- 预填充 8192x4096：未达持平。从起点 0.32x 改善到 0.388x（1.21x），距厂商仍差约 2.5x。
- 差距性质：TileLang 与厂商的能力边界，不是本内核缺陷。TileLang 官方 RMSNorm 示例（fp32）也只有厂商 0.69x，且不支持 bf16；厂商 bf16 路径比其 fp32 快约 3x；本任务 bf16 精度要求 fp32 统计，需上转，天然付出 fp32 级成本（见 evidence/tilelang-vs-vendor-rmsnorm-2026-11-02.json）。
- 调参已穷尽（E1-E11，仅 coalesced_width=4 有效）。真正达标需 bf16 原生、仍满足 fp32 统计精度门的 SIMD 微指令内核（T.SimdVF / T.simd.*，可行性已验证），属厂商级研究工作量。

因此：解码形状达成持平或超越；预填充持平列为需 bf16 原生 SIMD 实现的后续专项，非调参可达。
