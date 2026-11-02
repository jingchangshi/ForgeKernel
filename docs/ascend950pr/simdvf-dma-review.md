# SimdVF + DMA 方案：独立复核（Lead）

本文件是对 `simdvf-dma-plan.md` 的独立复核。原定由 accept-verify 小组完成，但其回合连续失败未产出，故由 Lead 直接复核。

## 已确认（源码或实测）

1. **DMA 在 SimdVF 外**：`examples/ascend/flash_attention/core.py` 中 `T.copy(...)`（86-88 行）在 `with T.SimdVF()`（94 行）之前；`testing/ascend/language/simdvf/` 各测试一致。因此"SimdVF 与 DMA 组合"是**顺序组合**，不是嵌套。
2. **逐 lane 累加可行**：`serial_vreg_accum_kernel` 用 `alloc_var + vdup + vmula + vsts` 完成跨迭代累加。
3. **bf16→fp32 载入正确**（本次实测）：`vld(dist="UNPK_B16")` + `vcvt("float32", part=0)` 对 64 元素给正确值，逐元素与 torch 相等。

## 仍未验证（不得据此下结论）

1. **跨 lane 归约到标量**：`vcadd` 是成对减半（32→16→…），到标量需链式调用；链的写法与开销未测。
2. **bf16 写回**：`vsts` 对 bf16 的 dist/掩码未测。
3. **rstd 标量广播**：`vdup` 一个由标量算出的值在 SimdVF 内的写法未测。
4. **性能**：无任何 SimdVF 版 RMSNorm 的计时；当前"会更快"只是机理推断。

## 对计划的两点质疑

1. 计划称"可省掉一次 fp32 fragment 物化"——但若两遍都要重读 `x_ub`，第二遍的 vld/vcvt 开销可能抵消收益；需在 E-E 用配对计时判断，不能预设。
2. 计划把 UB 预算视为"同量级"——但 UB 上限目前只有单缓冲探针观测（262144 B 通过、263168 B 被拒），**不是通用硬顶**；新内核必须按自身分配实测，不能套用。

## 复核结论

方向正确且关键未知已被本次实测部分消除（bf16 载入）。但**归约到标量与 bf16 写回仍是硬门槛**，在 E-B/E-C 通过之前不应宣称任何性能预期。建议按计划的门控顺序推进，先钉死归约。
