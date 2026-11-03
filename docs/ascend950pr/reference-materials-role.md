# 参考资料到底给了什么：asc-devkit 对 RMSNorm 优化的真实贡献

**作者**：lead。**日期**：2026-11-03。
**依据**：/home/s00653124/workspace/asc-devkit-master、core-count-and-l2-breakthrough.md §1–§2、devkit-lessons-reg-compute.md、devkit-lessons-memory.md、原会话 seq 20119。

> **为什么要单写这一份**：操作者把「指向参考资料」列为关键干预。但**把功劳记错，会让教训也记错**。这份文件只回答一件事：两个突破里，哪些真是 devkit 给的，哪些是 agent 自己的探针测的。

---

## 0. 结论

| 突破 | devkit 给了吗 | 真实来源 |
|---|---|---|
| **56 个 AIV 核（grid 64→56）** | **没有直接给数字** | agent 的 step-detection 探针；devkit 只提示了「去看厂商怎么组织」 |
| **L2 读路径 NOTALLOC_CLEAN** | **给了机制，没给对的枚举** | devkit 文档讲清了「只读一次的数据别进 L2」；但**按文档映射到的 NOTALLOC_KEEP 实测无效**，真正有效的是 NOTALLOC_CLEAN，靠实测找到 |

**一句话**：devkit 的作用是**换提问层次**，不是**给答案**。把它说成「文档里有答案」会掩盖真正的教训。

---

## 1. 56 核：devkit 没写这个数字

core-count-and-l2-breakthrough.md §1.1 记录了三处**独立**声明 vector_core_cnt=56 / ai_core_cnt=28：平台 config、torch_npu 报告、以及 step-detection 探针。

探针是关键证据（§1.2）：**固定每块工作量，grid 从 1 扫到 224，6 次重复** —— grid 1..56 **平坦（~10.0 µs）**，到 57 **跳 +82%（~18.5 µs）**，之后到 84 又平。**这个形状直接说明并发 AIV 核数是 56**：第 57 块起要排队等第二波。

**这是 agent 自己设计并跑的探针。** devkit 里没有任何一处写「950PR 有 56 个 vector core」。devkit 的作用是让人想到「该去看看硬件到底几个核」，**不是提供了这个数**。

> 诚实地说：即使没有 devkit，一个称职的性能工程师也该问「我按 64 块启动，芯片真有 64 个核吗」。**这个错误和 devkit 无关，是没验证启动假设。** devkit 只是把注意力引到 host 侧。

---

## 2. L2 读路径：机制是文档的，枚举是实测定出来的

### 2.1 devkit 确实写清了机制

- l2_cache_mode_config.md:33：工作集超过 L2 时，「旧数据必须先写回 GM，这占用 GM 带宽」。
- l2_cache_mode_config.md:35：**对只访问一次的数据，设成不进 L2，「避免一次性访问的数据替换掉有效数据」**。
- add_high_performance Case 5 是同一思路，也是 devkit **最大**的实测赢：A2 264.02→187.68 µs（**-28.9%**），950 混合带宽 1.712→2.256 TB/s。

**这就是我们读路径策略的机制来源，没有疑问。** 并且这印证了 §2.3 的话：「机制是文档的，不是这里发明的」。

### 2.2 但按文档映射到的枚举，实测是**错的**

这是最值得留下的一段。devkit-lessons-memory.md 把 NOTALLOC_KEEP 排第一，理由是 devkit 的 CACHE_MODE_DISABLE 映射到 load 枚举 4 == NOTALLOC_KEEP（SetL2CacheHint.md:82、kernel_utils_macros.h:233-244）。

**门控实测**：NOTALLOC_KEEP **0.6600 vs NORMAL_FV 0.6626 —— 1.004×，即什么都没有**。预测里 Case-5 量级的赢**没出现**；出现的是 NOTALLOC_CLEAN。

**这说明两件事**：
1. **文档给的映射是对的（语义上）**，但「不进 L2」这个家族里有多个枚举，**哪个在这个 kernel 上有效，文档答不了，只能实测**。
2. 真正的赢来自**消除脏行回写**（CLEAN 做的事，KEEP 不做），而不是笼统的「绕过 L2」——**这个更细的判断是文档给不出的，是实测买来的。**

---

## 3. 可迁移的技术（与两个具体数字分开）

devkit 真正教的是**结构**，这部分迁移到了我们内核里（devkit-lessons-reg-compute.md）：

- **两阶段结构 + 每边界一个 barrier**：load phase / compute phase 之间显式同步，不做细粒度交错。→ 对应我们 Era B 的 v15 无竞态两相。
- **整问题装进 UB 就不做复杂 tiling**：softmax 示例「deliberately NOT a pipeline example」，因为 shape 装得下。→ 提醒我们 UB 预算是硬约束。
- **每条指令对应一个硬件动作的写法**（vld/vsts/vmula 逐条），而不是让编译器猜。→ 这是我们 SimdVF 路线的范式。
- **overlap 结构在 GELU 示例里、不在 softmax 里**：文档自己分流了「结构示例」和「流水示例」。

---

## 4. 什么没迁移，为什么

- **devkit 是手写 AscendC，我们是 TileLang-Ascend**：devkit 能直接调 asc_* C-API 与 CACHE_MODE_* 宏；我们只有 TileLang 暴露的那一层。**这就是为什么 devkit 的 Case-5 赢（-28.9%）我们只拿到 1.103×——量级不同不是我们做错了，是命令面不同。**
- **devkit 的 softmax 是 UB 内整块**，RMSNorm 的 128 MB 工作集装不下，必须走流水——**照搬它的 tiling 会错。**
- **两个最快的 L2 策略在 devkit 里没提是危险的**，但在我们的 kernel 上**破坏正确性**（core-count-and-l2-breakthrough.md §3 红线）。**文档的沉默不等于安全。**

---

## 5. 给下一次的判据

> **值得读，但不要指望它给数字。** 它的价值是让你**换一个问题**——从「我这个 kernel 还能调什么」换成「厂商在同样硬件上靠什么结构」。换了之后，两个答案（56 核、NOTALLOC_CLEAN）**仍然要自己测**，而且**文档直接指向的那个枚举（NOTALLOC_KEEP）恰好是错的**。
>
> 所以正确的记法是：**参考资料改变提问层次，实测给出答案，两者缺一不可。** 把功劳全给 devkit，下次就会以为「多读文档就行」；把功劳全给探针，下次就会以为「不用读别人的实现」。