# vcadd 语义实测：质疑成立，但内核仍然正确

**作者**：lead（原 task-13/task-14 的两个执行者因模型额度耗尽中断，探针脚本已落盘但证据文件未产出；本文件由我复跑并重写）。
**日期**：2026-11-03。**设备**：Ascend950PR_9579，device 1。

标签：**[measured]** / **[verified: 路径:行]** / **[inferred]**。

---

## 0. 结论

| 问题 | 结论 |
|---|---|
| `T.simd.vcadd` 是"把全部活跃元素求和写入最低车道"吗？ | **是。质疑成立。** [measured] |
| 内核把它当"对折求和"用了吗？ | **用了 6 次链式调用，但这不是对折语义。** |
| 因此内核结果错了吗？ | **没有。输出 bit-identical。** [measured] |
| 那 6 次调用有害吗？ | **纯冗余：每次归约浪费 5 条指令。** |
| 值得改吗？ | **值得——prefill 实测 +0.60%，且零风险**（可证 bit-identical）。 |

**一句话**：质疑在语义上完全正确，在后果上不成立；它发现的不是正确性 bug，而是一处**可安全删除的冗余**，顺带暴露了上游文档的一处错误。

---

## 1. 实测：`vcadd` 是全归约，不是对折

三次输入，每次读**全部 64 车道**（不是只看 lane 0）。输入 = 全 1，float32，64 车道。

| 输入 | 调用次数 | lane0 | **非零车道数** | low8 |
|---|---|---|---|---|
| ones (全 1) | 1 | **64.0** | **1** | [64,0,0,0,0,0,0,0] |
| ones | 2 | 64.0 | 1 | [64,0,...] |
| ones | 3 | 64.0 | 1 | [64,0,...] |
| ones | 6 | 64.0 | 1 | [64,0,...] |
| ramp (0..63) | 1 | **2016.0** | 1 | [2016,0,...] |
| tail (1..63) | 1 | **63.0** | 1 | [63,0,...] |

- ones → **64.0** = 全部 64 个 1.0 之和，写入 lane 0，其余车道清零。
- ramp → **2016.0** = sum(0..63) = 63×64/2，与 CPU 参照**精确相等**。
- **一次调用就得到完整和**，2/3/6 次调用结果完全相同 → **幂等**。

**所以操作者的判断是对的**：`vcadd` 确实是 "add all active elements together and write the result to the lowest element of the destination vector register"。

### 1.1 顺带澄清三个同族算子的真实分工 [measured]

同样输入（ramp 0..63），只换算子：

| 算子 | 1 次调用 | 非零车道 | 语义 |
|---|---|---|---|
| `vcadd` | lane0=2016, nz=**1** | 1 | **全归约到 lane 0** |
| `vcpadd` | low8=[1,5,9,13,...], nz=**32** | 32 | **相邻两车道相加，打包进低半**（真正的"对折"） |
| `vcgadd` | low8=[28,92,156,...], nz=**8** | 8 | **每 8 车道分组求和** |

`vcpadd` 才是"对折"：nz 依次 32→16→8→1，第 6 次才收敛到 2016。**内核想要的对折语义应该用 `vcpadd`，用的是 `vcadd`。**

### 1.2 发射出的指令 [measured]

编译内核后读 `get_kernel_source()`：

```
auto v1 = simd_inst::vcadd(v0, full, MODE_ZEROING);
```

`MODE_ZEROING` 意味着非目标车道被清零——这就是"额外调用幂等"的机制来源：第二次调用对 {total, 0, 0, …} 求和仍得 total。

---

## 2. 为什么内核结果**没有**错

三条独立证据，全部 [measured]：

**证据 1 — 幂等性。** 上表：`vcadd` 调用 1/2/3/6 次，lane0 恒为 64.0 / 2016.0 / 63.0。链式重复调用**不改变数值**。

**证据 2 — 真实内核 bit-identical。** 把 kernel.py 里全部 **5 条** 6-调用链（rows==4 分支的 acc0..acc3 四条 + decode 分支的一条，共 30 次 `vcadd`）压成**每条 1 次**（共 5 次），编译两个变体，在同一输入上跑 8192×4096：

```
max_abs      : 0.0
numel        : 33554432
mismatches   : 0
bit-identical: True
```

**3355 万个 bf16 元素，零差异。**

**证据 3 — 对内核对 CPU 参照。** 用**设备实际吃进去的同一份输入**（关键：NPU 与 CPU 的 RNG 不通用，重新 seed 会比对到不同数据，我第一次就踩了这个坑）算 float32 参考 RMSNorm：

```
max_abs : 0.015625   <- 恰好是 2^-6，bf16 在 ~1.0 量级上的 1 个 ULP
mean_abs: 1.39e-08
```

**内核算的就是正确的 RMSNorm。**

### 2.1 所以质疑的"后果"部分不成立，但"语义"部分完全成立

把这两件事分开是本文最重要的一句：**质疑者对了语义、错了后果；内核没坏，但确实在浪费指令。** 二者不矛盾。

---

## 3. 冗余的代价：配对计时实测

**单次 A→B 读数不可采信**（效应只有 0.6%）。用配对交替法：同进程内 A/B 交替、每对翻转先后、24 对、报每对比值中位数 + 自助法区间。

| 形状 | A（30 vcadd） | B（5 vcadd） | 每对 A/B 中位数 | 区间 | A 更慢的对数 |
|---|---|---|---|---|---|
| **prefill** 8192×4096 | 44.55 µs | 44.32 µs | **1.0060** | [1.0036, 1.0083] | **23/24** |
| **decode** 64×4096 | 4.30 µs | 4.31 µs | 0.9968 | [0.9941, 1.0014] | 9/24 |

- **prefill：删掉 25 次 `vcadd` 得 +0.60% [0.36%, 0.83%]，24 对里 23 对方向一致** → 真实且可复现。
- **decode：无可靠变化** —— 区间跨过 1.0，9/24 对方向一致（抛硬币）。原因明确：decode 只有 4.3 µs、64 行铺在 56 核上，**受发射开销支配**，归约指令不在关键路径上。

> **一个必须记下的坑**：`torch.npu.Event.elapsed_time()` 返回**毫秒**不是微秒。我第一版把它标成 µs，得出"8192×4096 内核 0.045 µs"这种物理上不可能的数字。是绝对计时自检（wall 45.5 µs）发现的。**任何只报相对比值、不核绝对量级的计时都可能这样错下去。**

---

## 4. 上游文档有错

`/home/s00653124/workspace/tilelang/tilelang/ascend/language/simd.py:957`：

```
def vcadd(src, mask=None, mode="MODE_ZEROING"):
    """Pairwise add reduction: dst = vcadd(src, mask)."""
```

**这句是错的。** 同文件 `:945` 的 `vcpadd` 才写对了："Pairwise adjacent-lane add. The sums of adjacent source lanes are packed into the low half of the result vector."

也就是说：**上游把 `vcpadd` 的描述错安到了 `vcadd` 头上，而两个函数就在同一段代码里、相隔 12 行。** 内核作者（上一个会话）依此写成 6 次链式对折，是可以理解的——这不是粗心，是**跟着上游文档走**。

**建议**：向上游报一个文档修正。这不影响任何已认证结果（因为幂等），但会持续误导下一个读者。

---

## 5. 该不该改

**该改，理由是零风险而非性能。**

1. **正确性风险为零**：可证 bit-identical（§2 证据 2），且已用 3355 万元素验证。
2. **性能 +0.60%**（prefill），decode 无变化。相对已认证的 prefill 1.1883，若叠加约 →1.195，属于**小幅但真实**的改善。
3. **代码更贴近意图**：一条 `vcadd` 表达"全归约"，比 6 条表达得更清楚；`vdupv(..., POS_LOWEST)` 在两种写法下都正确。
4. **顺带减少寄存器压力与指令数**（每归约 6→1）。

**但必须按项目的认证纪律走**：本文只证明"删掉是安全的、且约 +0.6%"，**不等于新的认证结论**。若要落到 `candidates/` 里作为新版本，仍需过完整的配对认证流程（多轮、争用修正、精度 6/6）。

**不建议**直接改 v33 原地——它是已认证版本，应新增一个 v34 候选来承载。

---

## 6. 复现

```bash
cd /home/s00653124/workspace/ForgeKernel/docs/ascend950pr/probes
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES=1
source /data/pri/cann-9.2.0/cann/set_env.sh

# 语义（三方算子对比 + 全车道读出 + 发射指令）
/home/s00653124/.conda/envs/s00653124_tilelang/bin/python3 vcadd_semantics_probe.py

# 正确性（5 条链压成 1 条，比对 bit-identical + CPU 参照）
.../python3 vcadd_chain_collapse_check.py

# 性能（配对交替 A/B，24 对）
.../python3 vcadd_chain_paired_timing.py
```

**注**：`probes/_vcadd_chain_kernels.py` 是 `vcadd_semantics_probe.py` 的被导入依赖（被保留）。
中断前另写的 `_vcadd_chain_cost.py`、`_vcadd_full_collapse.py` 等已删除——它们只做单次 A→B 读数，
按 §3 的理由不可采信。本文三个结论分别来自上面三个可直接运行的脚本。