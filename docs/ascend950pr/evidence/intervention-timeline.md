# 操作者干预时间线（取证）

**作者**：lead。**日期**：2026-11-03。**方法**：原会话 `session-34744689-56f0-433d-a874-44f74a2d5fe5`（「昇腾服务器运行ForgeKernel方案讨论」）的事件级检索 + 逐字引用。

标签：**[verbatim]** 原文引用 / **[seq]** 事件序号 / **[compacted]** 原文已被上下文压缩，仅在 checkpoint 摘要中留存。

> **为什么要这份文件**：操作者记忆中有三次关键干预。叙事报告不得凭记忆写干预，必须落到事件上。以下每条都给出证据定位；查不到的如实写「未记录」。

---

## 干预 1：SIMTVF -> SIMDVF（最关键的一次）

**操作者原话** [verbatim, 存于 checkpoint seq 17752 @ 2026-11-02T13:09:49]：

> 我看了 `docs/ascend950pr/candidates/rmsnorm_cw4/kernel.py`，使用了 **SIMTVF**，而 rmsnorm 类算子理论上使用 **SIMDVF+DMA** 访存才能达成极致性能。按照这个思路，使用 agent team 审视接下来的性能优化工作。

**原始事件已 [compacted]**：直接搜索 `极致性能` / `rmsnorm类算子` 在 user/message 中均无命中，该原始事件被上下文压缩替换；原文确定性来自两份 checkpoint 摘要（seq 13063 @11:02 与 seq 17752 @13:09）的一致引用。

**干预前的状态** [seq 12242 @ 2026-11-02T10:37:28，assistant/reasoning 原文]：

> Round 8 done: E10 negative, consolidated log updated (10 experiments, only cw4 wins)… Decode: ACHIEVED (certified EXCEED 1.095x). Prefill: NOT achieved (0.388x), structurally stuck after 10 experiments.

即：**10 个实验（E1–E10）后自认撞上结构性墙，prefill 停在 0.388×**，只有 decode 达标。

**⚠️ 归因的精确形态（必须照实写，不要简化）**：在同一事件 seq 12242 的推理**末尾**，agent 自己写道：

> Actually — \"SimtVF\" vs \"SimdVF\": there's also SimdVF (SI…

并且 seq 12243（10:37:28）紧接着就发起工具调用「Inspect SimdVF API and usage examples」。**在这条消息之前最近的 user/message 是 seq 12228（10:36:55），内容是系统时间戳，不是操作者指令。**

**因此准确的说法是**：agent 与操作者在约半小时内几乎同时到达 SimdVF —— agent 在 round 8 收尾时自行注意到 `SimtVF` vs `SimdVF`，操作者的消息把它从「agent 的一个念头」**升级为明确指令**，并追加了「使用 agent team 审视」这一组织要求。写成「操作者教了 agent 一个它完全不知道的东西」或「agent 完全自己想到」都不准确。

**干预后的动作** [seq 15782 @12:05:35]：单行 SimdVF RMSNorm「与 CPU float64 完全一致（max_abs = 0.0）」，随后扩展多核；[seq 15881] 完整 8192×4096 正确，同窗口配对显示 **SimdVF 实质快于 SimtVF**（6 组交替 ABBA，中位 ~1.208×）。

---

## 干预 2：40 轮工作计划

**操作者原话** [verbatim, 同存于 checkpoint]：

> 在之前工作的基础上，制定最大 40 轮工作计划，持续优化算子性能直至达成既定性能目标。

**产物**：`docs/ascend950pr/plan-40-rounds.md`；目标恢复为最多 80 轮。时间点：SIMD 转向之后（seq 15761 @12:04 的对话里已引用该计划的阶段划分）。

---

## 干预 3：指向 asc-devkit 参考仓库（第二次关键干预）

**操作者原话** [verbatim, seq 20119, user/message, 2026-11-03T01:16:16.649Z]：

> 在你的上述分析之外，`/home/s00653124/workspace/asc-devkit-master` 是 AscendC 开发算子的教程仓库，其中 `…/examples/01_simd_cpp_api/05_best_practices/02_reg_compute/softmax_high_performance` 应该对你有参考意义，其他子目录可能也有，你来读取后学习经验，思考对你当前的 rmsnorm 算子有何收益。

**这是三条干预里唯一能用事件序号精确锚定的**（另两条原文已压缩）。

**干预前后**：这是唯一一次**在 R80「杠杆已穷尽」宣告之后**又挖出东西的干预。R80（轮次预算 80/80）时的结论是 prefill 在四个窗口上 FAIL（0.771–0.850）、`objective_met=false`、**杠杆清单宣告耗尽**。之后 R81–R83 是模拟器研究（以「模拟器无法区分 12% 差异」的决定性否定收尾），**R84 才拿到 56 核 + NOTALLOC_CLEAN 两个突破**。

---

## 顺序关系（这是要点）

| 序 | 事件 | 时间 | 前后状态变化 |
|---|---|---|---|
| 1 | 10 个实验后撞墙，prefill 0.388× | ~10:37 | SimtVF 天花板 |
| 2 | 干预1：SimtVF→SimdVF（+use agent team） | ~11:00–13:09 之间（原文压缩） | 转入 SIMD，prefill 到 0.62×→0.72× |
| 3 | 干预2：40 轮计划 | 之后 | 系统化推进 R1–R80 |
| 4 | **R80 宣告杠杆耗尽** | — | prefill 仍 FAIL |
| 5 | 干预3：指向 asc-devkit | 2026-11-03 01:16 | **R84 出 56核+L2，prefill 首次 EXCEED** |

**结论（可直接引用）**：第一次干预把方向从 SIMT 模拟改成 SIMD 硬件路径，但只把 prefill 从 0.39× 提到 ~0.81×；**真正跨过 1.0 的是第三次干预（参考资料）之后才出现的 56 核与 L2 策略**。也就是说，第二次「换范式」是必要条件，第三次「换视角（读厂商怎么想）」才是充分条件。

---

## 未记录/存疑

- 干预 1、2 的**原始事件 seq 已被压缩**，无法给出精确到秒的发送时刻；只能给出 checkpoint 留存的原话与其时间戳（13:09 / 11:02）。
- 干预 1 与 agent 自主察觉之间**无法确定严格先后**（agent 的自主察觉在 10:37:28，操作者消息原文时间戳晚于它但其原始事件不可见）。本文按「几乎同时、以操作者消息为指令化节点」表述，不作更强的断言。