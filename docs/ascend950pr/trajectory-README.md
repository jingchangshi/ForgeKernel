# 优化轨迹文档集：导读

**日期**：2026-11-03。**作者**：lead。

本目录里关于 RMSNorm 优化轨迹的文档不止一份，读者容易混。这份导读说清每份是什么、该先读哪份。

## 按需求选

| 你想知道 | 读这份 |
|---|---|
| **Agent 每次在想什么、判断错在哪** | [`optimization-narrative.md`](optimization-narrative.md) ← **先读这份** |
| **每个优化点对应哪个 kernel 版本** | [`optimization-points-and-kernels.md`](optimization-points-and-kernels.md) |
| **逐轮原始记录（R1–R84 + v34）** | [`raw-forgekernel-iterations.md`](raw-forgekernel-iterations.md) |
| **三次干预的取证（含操作者原话与 seq）** | [`evidence/intervention-timeline.md`](evidence/intervention-timeline.md) |
| **参考资料（asc-devkit）到底给了什么** | [`reference-materials-role.md`](reference-materials-role.md) |
| **机器可读的 40 条账本** | [`evidence/optimization-ledger-2026-11-03.json`](evidence/optimization-ledger-2026-11-03.json) |
| 旧版按「信念时代」组织的长文 | [`optimization-trajectory-R1-R84.md`](optimization-trajectory-R1-R84.md) |
| 56核 + L2 的技术细节 | [`core-count-and-l2-breakthrough.md`](core-count-and-l2-breakthrough.md) |

## 这批文档的分工

- **narrative** = 解释性。讲决策与误判，不堆数字。
- **points-and-kernels** = 目录式。每个优化点 → 版本 → 代码落点 → 效果 → 存留。
- **raw-iterations** = 流水式。逐轮改了啥、测了啥、判了啥。
- **intervention-timeline** = 取证式。只放能锚定到会话事件的东西。
- **reference-materials-role** = 归因式。防止把探针的功劳记到文档头上。

## 三条一致的结论（五份文档互相对得上）

1. **39 条（现 40 条）里真正有效的优化只有 6 条。** 大部分是负结果与方法修正——这是轨迹的真实形状，不是失败。
2. **把 prefill 从 0.39× 抬过 1.0 的两下（56 核、NOTALLOC_CLEAN）都不在 kernel 体里。** 80 轮调 kernel 体只到 0.85×。
3. **两次最大的方向修正都来自「换提问层次」**：一次是操作者指出范式（SIMT→SIMD），一次是参考资料让人去看 host 侧。工具能把每次尝试判准，但换层次它自己不会做。

## 关于 v34

本会话新增 `candidates/rmsnorm_simdvf_v34/`：把 5 条 6×`vcadd` 链坍缩成 1 次（30→5 次调用）。**两形状 bit-identical**，prefill +0.66%、decode +0.60%。它是**候选**，不是认证结论——0.6% 可能在认证噪声内，其价值在于「可证不改答案且少 25 条指令」。

## 一个已知的账本疑点

`optimization-ledger` 的 R1 条写 `float32 vector = 32 lanes`，本会话实测是 **64 lanes**（并已在 `discrepancies` 字段标注，未静默改）。细节见 [`evidence/vcadd-semantics-probe.md`](evidence/vcadd-semantics-probe.md)。