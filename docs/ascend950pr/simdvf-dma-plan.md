# SimdVF + DMA RMSNorm 优化方案

状态：审查草案。标注 **[已实测]** / **[源码证据]** / **[未验证]**。作者：Lead（原定的两名小组成员连续回合失败未产出，故由 Lead 完成）。日期：2026-11-02。

## 0. 结论先行

现有候选 `docs/ascend950pr/candidates/rmsnorm_cw4/kernel.py` 使用 `T.SimtVF`（SIMT 模拟）承载全部算术。用户判断正确：Ascend 向量单元是 SIMD，**SimdVF + DMA** 是理论上更贴近硬件的路径。审查结果：

- **[源码证据]** DMA（`T.copy`）必须位于 `with T.SimdVF()` **之外**：官方 `examples/ascend/flash_attention/core.py` 与 `testing/ascend/language/simdvf/` 全部如此组织。
- **[源码证据]** SimdVF 内已有逐 lane 累加模式：`alloc_var` + `vdup(0)` + 循环 `vld → vmula` + 末尾 `vsts`（`test_simdvf_arithmetic.py::serial_vreg_accum_kernel`）。这正是平方和所需。
- **[已实测]** bf16 载入语义已解决：`vld(x_ub[i], dist="UNPK_B16")` + `vcvt(v, "float32", part=0)` 对 64 个 bf16 元素给出**完全正确**的 f32（逐元素与 torch 相等）。**这推翻了此前"part 语义不可知"的担忧。**
- **[未验证]** 跨 lane 归约到**单个标量**（vcadd 链）、bf16 回写（`vsts` 的 dist）、rstd 标量广播、以及最终性能。这些必须在动工前用探针逐个钉死。

## 1. 目标形态映射

每行 D=4096，按 64 bf16 = 一个向量分块，共 64 个分块。

阶段 A（DMA）：`T.copy` 把 `rows×D` 的 bf16 tile 搬入 shared（`T.Pipelined` + `annotate_buffer_versions` 与现内核一致）。

阶段 B（SimdVF，第一遍，求平方和）：
```
acc = T.simd.alloc_var(f32); acc = T.simd.vdup(0.0, f32, full)
for i in T.serial(D // 64):
    v  = T.simd.vld(x_ub[row*D + i*64], dist="UNPK_B16")
    vf = T.simd.vcvt(v, "float32", part=0)
    T.simd.vmula(acc, vf, vf, full)      # acc += vf*vf  [源码证据]
```
再对 `acc` 做跨 lane 归约 → 标量 s；`rstd = 1/sqrt(s/D + eps)`；`rb = vdup(rstd)`。**[未验证：归约链与标量广播]**

阶段 C（SimdVF，第二遍，缩放）：
```
for i in ...:
    v = vld(...); vf = vcvt(v, f32, part=0)
    wv = vld(w_ub[i*64], dist="UNPK_B16"); wf = vcvt(wv, f32, part=0)
    y = vmul(vmul(vf, rb), wf)
    vsts(y_ub[..., i*64], vcvt(y, bf16), dist=??)   # [未验证: bf16 写回 dist]
```
阶段 D（DMA）：`T.copy(y_ub, Y)`。

## 2. 为什么可能显著变快

- 现内核把 (rows, 4096) 的 **fp32 fragment** 物化在寄存器/UB，逐元素走 SimtVF 的线程模拟；SimdVF 直接用真向量指令，理论上去掉模拟开销。
- 第一遍与第二遍可以**融合**（在线算法）：若边读边累加同时在第二遍复用 tile，仍受 UB 限制；至少可省掉一次 fp32 fragment 物化。
- 但**不要预设会达标**：厂商是手写 C++ 内核，且本任务 bf16 需 fp32 统计。**[未验证：实际加速比]**

## 3. 风险

1. 跨 lane 归约链若有额外指令开销，可能吃掉收益。
2. bf16 写回的 `vsts` dist/掩码若用错会静默产生错误结果 → 必须先过精度门。
3. 每行 64 个 64 元素分块 = 64 次 vld/vcvt，指令数固定；若归约/转换是标量瓶颈，可能反而不如 SimtVF 的批量 fragment。
4. UB 预算：SimdVF 需要 `x_ub`、`w_ub`、`y_ub` 与可能的 `acc` 局部；与现内核同量级，但仍需按实测验证（UB 上限目前只有单缓冲探针观测，非通用硬顶）。

## 4. 门控实验顺序（正确性优先）

1. **E-A**：复现 bf16 `vld+vcvt` 正确性（已通过，作为回归基线）。
2. **E-B**：单行 4096 的平方和 → 标量，与 CPU float64 比，要求相对误差 ≤ 2^-7（一个 bf16 ulp）。
3. **E-C**：单行完整 RMSNorm（含 w、rstd、bf16 回写），与 CPU float64 比。
4. **E-D**：多行/多核映射正确性（64 核、rows per core）。
5. **E-E**：与现 SimtVF cw4 内核**同窗口配对**计时；只用配对差值，不用跨 run 绝对时间。
6. **E-F**：通过独立验收门（15+ 配对窗口、多进程、见证不足则 RETEST）。

任何一步失败即停在原地，不进入下一步；绝不把"看起来更快"当作达标。

## 5. 下一步优先级

1. 用探针钉死 **vcadd 跨 lane 归约到标量的确切写法**（最高未知）。
2. 钉死 **bf16 的 vsts 写回 dist**。
3. 写 E-C 单行完整 RMSNorm 的正确性原型。
4. 通过后再谈多核映射与计时。
