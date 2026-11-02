# 验收协议（accept Gate）

最后更新：2026-11-02。实现：`agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py`（worktree `ascend950pr-ag`，untracked）。
host-only 回归：`tests/test_acceptance_rmsnorm.py`（51 passed，不占卡）。
负例实测记录：`evidence/negative-controls-2026-11-02.json`。

本文件是**判据的定义**。代码里改了判据必须同步改这里；两者不一致时以本文件为准，代码视为缺陷。

---

## 0. 一句话

候选必须**用自己的 TileLang kernel 算出** BF16 RMSNorm，**数值上不差于厂商**，**在同一测量窗口内配对计时不慢于厂商**；三者缺一不可，任何一项失败就是 FAIL，噪声不足则 RETEST，**永不用几何平均掩盖单 shape 失败**。

---

## 1. 三方分离（为什么必须是三方）

| 角色 | 是什么 | 作用 | 不能做什么 |
|---|---|---|---|
| 数学 judge | 本脚本自己用 **CPU float64** 算的算子定义 | 判对错 | **绝不作为速度对手** |
| 厂商基线 | `torch_npu.npu_rms_norm(x, w, 1e-6)[0]` | 速度目标 | 不能被豁免精度判据 |
| 候选 | 从归档源码**重新编译**的 TileLang kernel | 被验收对象 | 不能复用任何已编译产物 |

任意两个合并就会出现"优化到错误的数"——把 judge 当对手会让目标变成"和 fp64 比快"，把厂商豁免精度会让厂商的舍入误差成为候选的免罪符。

### 1.1 数学 judge 必须在 CPU 上算 float64

昇腾上 `.double()` 会把 float32 张量**静默降级**为 float32（不报错、不警告）。更麻烦的是 `torch.zeros(dtype=float64, device="npu")` **是真的 float64**，所以"设备上能建 float64 张量"不能证明"设备上的 .double() 是 float64"。实测：

`@
float32 张量 .double().dtype  ->  torch.float32   <- 降级，无异常
float64 直接分配 .dtype       ->  torch.float64   <- 真 float64
`@

**对策**：真值和一切距离都在 CPU 上用真 float64 算，拷回在前、计时在后。拷贝与判据成本**绝不进入设备计时**（计时读 msprof 的 kernel 时长，本身不含 host 工作）。

实测：8192x4096 一次完整 fp64 距离计算约 **0.10 s**（CPU），相对 30 s 级设备测量可忽略。

---

## 2. 精度判据

### 2.1 两条规则

设 `e_cand`、`e_ref` 为相对 L2 误差（对 CPU fp64 真值，真值末尾舍入到 BF16 一次——因为该算子的输出**就是** bf16 张量，拿未舍入的 fp64 比会让每个实现平白多背一次舍入）：

`@
P1  e_cand <= K_RATIO * e_ref + max(FLOOR_ULPS * 2^-8, 2^-17)     K_RATIO=3.0, FLOOR_ULPS=1.0
P2  max_rel_cand <= max(2^-7, 1.5 * max_rel_ref)
`@

两条都要过。**P1 是相对 L2（整体），P2 是逐元素最坏值（尾部）**，各自漏掉对方的失败模式。

### 2.2 2^-7 到底是什么：**它是 1 个 BF16 ulp，比较必须用 <=**

`2^-7 = 0.0078125` 是 BF16 在一个 binade 顶端的相对间隔，即**一个 ulp 的相对上界**。正确舍入的实现**可以恰好落在它上面**。实测证据：

`@
8192x4096 / normal      vendor_max_rel = 0.0078125 = 2^-7 精确相等
                        fp32   max_rel = 0.0078125 = 2^-7 精确相等
`@

**所以判据必须是 `<=`，写成 `<` 会把厂商自己和正确的 fp32 实现判为失败。** 这条由三个测试锁死：`test_a_max_rel_exactly_at_the_floor_passes`、`test_the_measured_matrix_really_lands_on_the_boundary`（断言实测值确实等于 2^-7，不是假设）、`test_the_ulp_floor_is_exactly_one_ulp`。

### 2.3 判别力必须用 **ulp 倍数** 论证，不能用"余量百分比"

报告里每个 (shape, family) 都同时给 `ulp_cand`、`ulp_ref`、`ulp_limit`（= 值 / 2^-7）。**这是唯一有意义的单位**。本机实测 6 个组合全表：

| shape / family | vendor ulp | 正确 fp32 ulp | bf16 累加 ulp | 有效上限 ulp | bf16/上限 |
|---|---|---|---|---|---|
| 8192x4096 / normal | 1.000 | 1.000 | **5.525** | 1.500 | 3.68x |
| 8192x4096 / near_zero_rows | 0.992 | 0.992 | **3.391** | 1.488 | 2.28x |
| 8192x4096 / outlier_channels | 0.992 | 0.992 | **3.737** | 1.488 | 2.51x |
| 64x4096 / normal | 0.000 | 0.000 | **4.129** | 1.000 | 4.13x |
| 64x4096 / near_zero_rows | 0.529 | 0.703 | **2.054** | 1.000 | 2.05x |
| 64x4096 / outlier_channels | 0.601 | 0.525 | **3.657** | 1.000 | 3.66x |

结论：
- 正确实现 **0.00-1.00 ulp**，bf16 累加 **2.05-5.53 ulp**，**6/6 全部被拒**；
- **最窄的分离是 2.05x**（64x4096/near_zero_rows），不是 3.4x。引用"3.4-5.5 ulp"会**高估余量**，以本表为准；
- P1 的 `mean_abs` 分离更极端（1e5-1e6 倍），但 max_rel 的 ulp 倍数才是能直接对上 `2^-7` 阈值的量。

### 2.4 P2 的 1.5x 只在厂商贴着地板时才起作用（必须人工看 ulp 列）

`max(2^-7, 1.5 * max_rel_ref)` 的有效上限**从未超过 1.50 ulp**：

- 厂商 1.000 ulp -> 上限 1.500 ulp（放宽）；
- 厂商 0.000 ulp -> 上限 1.000 ulp（floor binds）；
- 厂商 0.601 ulp -> 1.5x0.601 = 0.901 < 1.0 -> **floor 仍然 binds**。

**残余风险（必须写进任何引用本判据的结论里）**：若将来某 shape 让厂商误差涨到 3 ulp，候选上限会变成 4.5 ulp，就能放进较粗糙的实现。因此：

> **判定不能只看 PASS/FAIL，必须逐 shape 复核 `ulp_cand` / `ulp_ref` / `ulp_limit` 三列。**

放宽的边界在 **0.6667 ulp**（推导：`1.5 * (u * 2^-7) <= 2^-7  <=>  u <= 1/1.5`）。实测 6 个组合里：

- **3 个已经放宽**：`8192x4096` 全族（vendor 0.992-1.000 ulp -> limit 1.488-1.500 ulp）；
- **3 个 floor 仍然 binds**：`64x4096` 全族（limit 1.000 ulp）；
- **最紧的一格是 `64x4096/outlier_channels`：厂商 0.6009 ulp，距 0.6667 只剩 0.0658 ulp。**

所以 1.5x 目前是**在实测的这两个 shape 上安全，不是结构性安全**。这是最该盯的一格。

### 2.5 e_ref -> 0 时 floor 够不够？（实测：余量 220 倍）

`e_ref` 极小时 P1 退化成 floor = `max(1.0*2^-8, 2^-17) = 0.003906`。会不会误判正确的 fp32 实现？实测 **不会**：

| shape / family | e_ref | limit | 正确 fp32 的 e_cand | 占 limit |
|---|---|---|---|---|
| 8192x4096 / normal | 1.520e-05 | 0.003952 | 1.454e-05 | **0.37%** |
| 8192x4096 / near_zero_rows | 1.591e-05 | 0.003954 | 1.792e-05 | **0.45%** |
| 8192x4096 / outlier_channels | 4.120e-06 | 0.003919 | 4.350e-06 | **0.11%** |
| 64x4096 / normal | 0.000e+00 | 0.003906 | 0.000e+00 | **0.00%** |
| 64x4096 / near_zero_rows | 2.690e-06 | 0.003914 | 1.108e-05 | **0.28%** |
| 64x4096 / outlier_channels | 6.500e-06 | 0.003926 | 2.170e-06 | **0.06%** |

最差 0.45%，**没有误判风险**。测试 `test_the_floor_has_room_for_an_honest_kernel` 断言不超过 5%。

### 2.6 保留集（held-out families）

三族输入，**每族用同一 seed**，所以三方看到完全相同的输入：

| family | 构造 | 想打掉的捷径 |
|---|---|---|
| `normal` | `randn` | 基准 |
| `near_zero_rows` | `randn * 1e-3` | 方差/epsilon 处理错（分母趋 0）、`E[x^2]-E[x]^2` 式算法 |
| `outlier_channels` | 第 0 列置 300.0 | 粗缩放、窄累加器压平离群通道 |

保留集的作用**不是**补容差漏洞（容差本身有区分力，见 2.3），而是**防过拟合**：只针对公开输入调优的候选仍要过这三族。

---

## 3. 计时判据

### 3.1 同窗口配对，绝对值跨 run 无意义

**跨 run 的绝对设备时间漂移 2.5 倍**（同一厂商内核实测 0.0206-0.0514 ms），因为共享机器上租户来来去去。所以任何"这次 0.021 ms、上次 0.051 ms"的对比都是伪结论。

**同一窗口内**的配对比值稳定在 **+/-8%**。可审计邻接负载（心跳 0->72660、AICore 利用率 0->90%）下配对比值只偏移 **3.3%**。

协议：
- 一个窗口内 A/B（候选先）或 B/A（厂商先）**交替**；
- 每个窗口两侧各一次 msprof 测量；
- 判定只用**逐窗口配对比值** `speedup = vendor_ms / candidate_ms`；
- 点估计 = 逐窗口比值的**中位数**；区间 = 该比值的**配对 bootstrap 95%**（4000 次，固定 seed）。

### 3.2 每个窗口必须有证词，无法分类的窗口一律作废

每个窗口记录：
- 窗口前后 `npu-smi info -t usages -i <device>` 的 `Aicore Usage Rate(%)`；
- 若提供 `--neighbour-heartbeat`，窗口前后的心跳值。

分类：

| 分类 | 条件 | 处理 |
|---|---|---|
| `quiet` | 利用率前后都 < 20% | 计入 |
| `contended` | 任一侧 >= 20%，或心跳推进 | **计入但标注**（实测争用只偏 3.3%，可用） |
| `unclassifiable` | 前后都读不到，且无心跳 | **作废（VOID）** |

**无法事后分类的窗口一律作废**不是洁癖：曾经出现过一个 +70% 的异常窗口，原因不可考，正是"邻居进程 stderr 被丢弃、没有 liveness 检查"造成的。一个无法标注的窗口不可解释，平均它只会把不可解释性摊进结论。

`npu-smi` 读数必须**校验它确实在描述目标卡**（输出里 `NPU ID : <device>` 匹配），否则读的是别的卡，报 unknown 而不是当作 quiet。

### 3.3 阈值与四种结论

`@
MATCH   = 点估计 >= 1.0  且  配对 bootstrap 95% 下界 >= 0.98
EXCEED  = 下界 > 1.0
FAIL    = 95% 上界 < 1.0        <- 板上钉钉地在噪声之外更慢
RETEST  = 其余                    <- 证据不足以判定，不是失败
`@

- **0.98 是预先声明的 2% 工程等价容忍**，不是看到结果后调的。
- `FAIL` 只在乐观端都输时给出：噪声救不回来才是真慢。
- **可用窗口 < 9 个 -> RETEST，永不 FAIL**。证据不足既不能判过也不能判失败。
- **p90/p10 > 1.25 -> RETEST**：窗口之间自相矛盾（超 2% 容忍能吸收的范围），不能当结论。

### 3.4 逐 shape 判定，禁止几何平均

每个 frozen shape 各自出结论；`decide()` **只要有一个 shape FAIL 就整体 FAIL**，部分超越如实写进 `verdict.reasons`。测试 `test_a_partial_win_does_not_become_a_pass` 锁死：8192 MATCH + 64x4096 FAIL -> 整体 FAIL。

### 3.5 计时 basis

`tilelang.profiler.msprof.bench_with_msprof`，返回**逐 kernel 设备时长**，天然 host-free（不含 host 入队时间）。

- `cache` 必须是 **int32**：msprof 自己的 flush kernel 在 Ascend 上无法 lower int8 向量（`SimtVF cannot convert vector type int8x8`）。
- 返回值可能不是 list（单 kernel 时是 float），聚合前要 `isinstance` 判断。
- 冷启动（空 cache 重编译）与 warm 稳态**分开报**；编译成本不进稳态。

---


### 3.6 窗口内的争用观测：canary kernel（为什么不用高频 npu-smi 采样）

**先记录一个被推翻的设计。** 早期要求是"窗口内每 100-200 ms 采一次 AICore 利用率轨迹"。**这个要求在物理上无法实现**，实测：

`@
npu-smi info -t usages -i 1 :  0.80 / 0.83 / 1.04 / 0.80 / 0.79 s
npu-smi info (full)        :  2.39 / 1.73 / 0.80 s
parallel 4 calls           :  1.26 s -> 3.2 samples/s （最好情况）
`@

单次调用 **0.5-3.2 秒**。声称拿到 100-200 ms 的采样密度就是**伪造 witness**。所以保留三条互不相同的证据：

| 证据 | 原理 | 失效模式 |
|---|---|---|
| **witness 1: AICore/HBM 采样** | 进程外、独立于我们的测量进程 | 慢，会漏掉突发；采样率受限 |
| **witness 2: 设备进程表** | 驱动生成，脚本无法伪造 | 只答"有没有别的进程"，不答"它有多忙" |
| **witness 3: canary kernel** | **在窗口内跑**，分辨率 = 测量本身的分辨率 | 进程内，与被测对象共享执行环境 |

若 AICore 利用率读数完全缺失，空进程表或未膨胀的 canary 不能证明窗口无争用，窗口标记为 `unclassifiable`；可观测到的正向干扰信号仍优先标记为 `contended`。共享卡可正常运行，不要求其他进程退出。

**canary 的做法**：每个窗口在 A/B 之后，用**同一个厂商算子**测一个**固定小块** `x[:64]`（同一类工作：向量单元 + HBM），得到 `canaryMs`。全 run 取中位数作基线，逐窗口算

`@
canaryInflation = canaryMs / median(canaryMs over run)
`@

**`canaryInflation >= CANARY_INFLATION = 1.15` -> 该窗口标 `contended`。**

为什么这比高频采样强：它**没有采样率问题**（观测发生在窗口内、与测量同分辨率），而且它测的是**"卡是否忙"这个我们真正关心的事实**，不是"有没有别的进程"这个代理指标（代理指标会漏掉同进程占多卡，也会把空闲进程误报成争用）。

为什么 `1.15` 是个可证伪的阈值：它是 15% 的膨胀，远大于实测的同窗口配对比值稳定性（+/-8%）和邻接负载偏移（3.3%），又小于一个真实争用租户造成的膨胀。**阈值写进常量，被测试锁死，可以被质疑。**

**canary 挡不住的**：如果邻居负载恰好与 canary 的执行时段错开（canary 每次只跑几毫秒），仍可能漏掉。**所以三条 witness 全部保留**，它们的失效模式互不相同；两条以上都无法取得时窗口作废，只有一条时标 `degraded` 降级使用。

### 3.7 `quiet` 必须可被质疑

`MIN_WITNESS_SAMPLES_PER_SECOND = 0.5`：窗口内采样率低于此时，该窗口**不允许标 `quiet`**，标记为 **`sparse`**。

`sparse` 的窗口**仍然计入判定**（它没有错误，只是观测不足），但报告里必须同时给出：

- `witnessSamplingRateHz`（min / median / max）——让读者知道采样率是被工具限制的，不是选择；
- `sparseWindows` 计数；
- `quietWindowsWithSparseSampling` 计数——标记 `quiet` 但采样不足的窗口数，**这个数必须为 0**，否则说明分类逻辑漏了 sparse 分支；
- `canaryInflation`（min / max / windows / threshold）。

**这条的意义**：一个只在开头结尾各看一眼的窗口，看起来"全程安静"，实际是"中间没看"。这两个结论在报告里必须长得不一样。



---


### 3.8 争用会不会被配对吸收？（受控正对照实测）

**方法**：同一个**未改动**的候选，跑两次——一次在干净卡上，一次在**真实持续争用**下（独立后台进程在**同一张卡**跑 180 s NPU 乘加循环）。同一个 artifact、同一份代码、同一个 device。

| | 干净（12:53） | 争用（13:03，AICore=99%） | 变化 |
|---|---|---|---|
| **8192x4096 点估计** | **0.3224** | **0.6615** | **x2.052** |
| 8192x4096 vendor 中位时长 | 0.0527 ms | 0.1412 ms | x2.68 |
| 64x4096 vendor 中位时长 | 0.0037 ms | 0.0583 ms | **x15.7** |
| 判据结果 | FAIL | **RETEST**（新规则） | — |

**结论 1（重要，推翻一条既有说法）：配对协议能吸收轻争用，不能吸收持续重争用。**

早先测得的"配对比值稳定 +/-8%、邻接负载只偏 3.3%"**只在轻负载下成立**。在 AICore 被占满时，**同一个未改动的候选**的配对比值从 0.3224 变到 0.6615 —— **2.05 倍**。

**原因**：两个臂对争用的敏感度**不对称**。候选是 64 核 SimtVF kernel，厂商是高度优化的小 kernel，AICore 饱和时受害程度不同，比值**不会抵消**。

**结论 2（协议变更）：没有 quiet 窗口的 run 不能下性能结论。**

`judge_timing` 现在规则：

`@
若可用窗口 > 0 且 quiet 窗口数 == 0
  -> RETEST，理由 "no quiet window; pairing's ability to cancel contention is
     unverified, and contention was measured to shift the paired ratio by 2.05x"
`@

这条把上面那次争用 run 判为 **RETEST** 而不是 FAIL ——**这才诚实**：它没有资格把争用算成候选的错。测试 `test_a_run_with_no_quiet_window_decides_nothing` 锁死。

### 3.9 canary 作为检测器**不灵敏**（实测，必须如实记录）

争用全程 AICore = 99%，canary 膨胀的实测结果：

| shape | canary 膨胀序列 | 超 1.15 的窗口 |
|---|---|---|
| 8192x4096 | 1.086, 0.139, 0.984, 1.158, 1.000, 1.181, 0.543, 1.156, 0.213 | **3 / 9（33%）** |
| 64x4096 | 1.121, 0.965, 1.083, 0.888, 0.506, 1.125, 1.000, 0.983, 1.186 | **1 / 9（11%）** |

**原因**：canary 测的是 ~0.06 ms 的 kernel，**它自身的噪声与被测信号同量级**。

**成本约束（Lead 实测）**：一次 canary 的 wall clock 是 **0.64 s**，而它要裁决的设备时间只有 **0.004 ms** —— **16 万倍成本**。一个窗口放 10 次 canary 要 6.4 秒，等于为了测 4 微秒占用 6.4 秒，违背"只在真正用卡时占用"。**所以 canary 保持单次。**

**因此 canary 的证据强度是单向的**：

> **膨胀 -> 强证据，判 `contended`。**
> **不膨胀 -> 弱证据，不得据此判 `quiet`。**

如果只用 canary，这次 99% 争用的 run 会有 **6/9 到 8/9 的窗口被错判为 quiet**。**窗口级争用检测实际靠的是 AICore 轨迹**（这次 9/9 窗口都是 99%，全部正确标 `contended`）。canary 是补充，不是主力。

### 3.10 witness 采样率实测（回应"保护过严导致失效"的担忧）

`@
64x4096:   samplingRateHz min 1.522  median 3.153  max 3.882   sparseWindows 0
8192x4096: samplingRateHz min 1.157  median 3.014  max 3.542   sparseWindows 0
`@

远高于 `MIN_WITNESS_SAMPLES_PER_SECOND = 0.5`，**没有把正常窗口误标 sparse**。原因：`_CardSampler` 是 **run 级后台线程**持续采样，而单个窗口约 1 秒，所以每个窗口能夹到 5-8 个采样点——而不是只在窗口首尾各看一眼。

### 3.11 一次归因错误（记录在案）

受控争用实验里，我起的负载进程 **PID 147892** 被 Lead 短暂误判为"第三方租户"，并据此认为"第一次抓到真实第三方争用"。**那个进程是我们的。**

**教训**：共享机器上从 PID/进程名倒推属主是不可靠的，属主需要 owner 级证据。本证据文件已把该实验标注为"受控实验（我们主动制造）"，而不是"自然争用"。**这条不影响实验结论的有效性**——争用是真实的、witness 是独立发现的——但影响"这是第三方负载"这个陈述的真实性。

### 3.12 判定样本规则（contended / sparse 窗口一律剔除）

3.8 实测出争用会把配对比值改变 **2.05 倍且方向随 shape 反转**，由此得到一条必须写进实现的硬规则：

```
判定样本 = 仅 quiet 且非 sparse 的窗口
contended 窗口  -> 记录、统计、展示，但不进点估计、不进 bootstrap
sparse  窗口    -> 同上（不是 quiet，"没看清"不等于"确实安静"）
若 quiet 窗口数 < 9            -> RETEST
若 quiet 窗口占全部窗口 < 70%  -> RETEST
```

**为什么是"剔除"而不是"用中位数抗掉"**：中位数抗随机噪声，不抗**系统性偏移**。contended 窗口的偏移是系统的（同一 artifact 在同一争用水平下稳定偏 2.05×），混入样本会整体搬移中位数，而不是互相抵消。

**为什么还要 70% 占比门槛**：即使凑够 9 个 quiet 窗口，如果另外 11 个全是 contended，剩下的样本就是**被系统性筛选过的**子集，代表性无法保证。

**剔除不掉的两个数字必须印在判定旁边**：`contendedWindows` / `quietWindows` 计数；若同一 artifact 同时有争用 run 与安静 run，还要报告两者的比值差（本例 `0.3224` vs `0.6615`）。理由：让读者一眼看出这个判定是在什么条件下得出的。

#### 这条规则的两项实测验证

**（a）重放已发布的数据——判定不能因为改规则而漂移**

干净 run（12:53）是规则生效**之前**跑的，所以必须用它自己的记录窗口重算，不能假设旧结论仍然成立：

| shape | 旧判定 | 新判定 | 判定样本 | quiet / contended | 是否改变 |
|---|---|---|---|---|---|
| 64x4096 | RETEST | RETEST | 10 | 10 / 0 | 否 |
| 8192x4096 | FAIL | FAIL | 10 | 10 / 0 | 否 |

干净 run 全部窗口都是 quiet，规则剔除不掉任何东西，**正式性能结论原样成立**。

争用 run（13:03）则相反：

| shape | 旧判定 | 新判定 | 判定样本 | quiet / contended | 是否改变 |
|---|---|---|---|---|---|
| 64x4096 | RETEST | RETEST | 0 | 0 / 9 | 否 |
| 8192x4096 | FAIL | **RETEST** | 0 | 0 / 9 | **是** |

**旧代码在 9/9 窗口全争用、没有任何 quiet 参照的情况下仍然输出了 FAIL**，这正是本规则要修的缺陷。争用 run 从此只作 witness 验证实验存证，**永不作为性能结论**。

**（b）回归探针——保护是不是真的在护**

把实现临时改回 `sample = list(usable)`（即恢复"contended 窗口计入样本"），**5 个测试立刻失败**，包括 `test_contended_windows_never_reach_the_verdict`。恢复后 74 passed。**没做这一步，就无法排除"测试只是看起来在测"。**

另有一个污染探针：15 个 quiet（1.02×）+ 5 个 contended（0.20×），quiet 占比 75% 已过门槛 → `decisionSampleWindows == 15`、`pointSpeedup == 1.02`，contended 臂**没有泄漏**进中位数。

**（c）一条同时修掉的不可达代码**

`usable < 9` 的检查原先排在 quiet 规则**之后**。由于 quiet 样本是 usable 的子集，`usable < 9` 时 quiet 规则必然先返回，那个分支**永远不可达**——又是一个"看起来有守护、实际不执行"。现已前移，并由 `test_an_unlabelled_shortfall_is_not_reported_as_a_contention_shortfall` 钉死。

#### wash-to-PASS 的具体算术

```
真实比值 0.478  ×  2.052（争用膨胀）  =  0.981
MATCH 的 95% 区间下界要求              =  0.98
=> 一个慢一倍的真实实现，会在争用下越过 MATCH 门槛
```

这不是理论担忧，是 3.8 那次实测的直接推论。

---

## 4. 设备基线有效性审计（host-bound）

### 4.1 判据

`host_bound_ratio = host_enqueue_ms / device_median_ms`，阈值 **0.5**（与 ForgeKernel `timing_audit.HOST_BOUND_RATIO_WARN` 一致）。

- **逐 workload 判，最大值 binding**：大 shape 不能替小 shape 平均掉停滞。decode（4 us 级）正是最先 host-bound 的那个。
- **没有任何一个 workload 报出 clock -> 基线无效**（不是"比值为 0"，是"无法 qualify"）。

### 4.2 聚合陷阱（实测会让判定翻转）

`timing.py` 发射的是**逐 workload 最大值**，但 `leaderboard.py:2426-2444 _task_ref_alone` **不发射**该最大值与逐 workload 审计。归档后重算必然得到：

`@
geomean(a_i) / geomean(b_i)  ===  逐 workload 比值的几何平均
`@

**恒等**，且**永远不等于最大值**。实例 `[0.04, 0.88]`：

| 算法 | 值 | 判定 |
|---|---|---|
| 逐 workload 最大值 | **0.88** | **invalid（阻断）** |
| geomean/geomean | **0.187617** | **valid（放行）** |

**判定相反。** 实测 verify：`rederived = 0.187617` 与 `sqrt(0.88*0.04) = 0.187617` 精确一致，`verdictFlips = true`。

所以 `audit_host_bound()` 除了按最大值判，还**额外判一条**：归档里发射的聚合值若与逐 workload 最大值不一致，**该归档本身就是缺陷**（`aggregateInconsistent = true`），基线无效。这样"消灭审计字段"这一动作会被发现，而不是被用来洗白。

---

## 5. 实现真实性检查（反作弊）

### 5.1 允许什么

`import torch_npu` **明确允许**：不注册后端就建不出 `"npu"` 张量，事件/分配/cast helper 都在那里。禁止的是**把计算委托给厂商融合算子**。

### 5.2 静态正则不够（实测零命中）

实测：用 `importlib` + `getattr` + 字符串拼接调用 `npu_rms_norm`，**名字黑名单全部零命中**，运行期却解析到禁用算子。

所以本实现做 **AST 可达性分析 + 常量折叠**：

- 绑定别名：`import torch_npu as X`、`from torch_npu import npu_rms_norm as Y`；
- 折叠字符串：`("npu_rms" + "_norm")`、`("".join([...]))`、f-string；
- **跟随动态导入的结果绑定**：`backend = importlib.import_module("torch" + "_npu")` 之后，`backend` 就是 `torch_npu`；
- **跟随 `getattr(<vendor module>, <常量>)` 的结果绑定**：`_fusion = getattr(_backend, "npu_rms"+"_norm")` 之后，`_fusion` 就是 `torch_npu.npu_rms_norm`；
- **折叠不出来的动态调用（`eval` / `exec` / `globals().get(name)` / 动态 `import_module`）判为 `unresolved -> FAIL`**：无法界定可达代码就是证据不足，放行等于放弃检查。

实测该 walk 对负例 #1 的命中（`evidence/negative-controls`）：

`@
kernel.py:16  resolves torch_npu.npu_rms_norm through getattr (alias of a vendor fusion operator)
kernel.py:20  calls the vendor fusion operator torch_npu.npu_rms_norm
`@

### 5.3 不能只扫本轮 diff：归档是自包含的

按契约，候选需要的 helper 必须在归档内，所以**归档里的每个 `.py` 按定义都可达**，全部扫描。归档外的 import 全部记录（那是 walk 唯一看不到的代码），并在报告里列出 `externalImports`。

### 5.4 三条腿都要，缺一不可

| 腿 | 证明什么 | 单独为何可被攻破 |
|---|---|---|
| 静态可达性 | 源码没有委托计算 | 源码可以存在但从不执行 |
| 编译产物 | **本次**验收在**空 cache** 里重新编译出了 `.aibin` | 有 `.aibin` 不等于它被执行 |
| 运行期证词 | 计时那次真的启动了 TileLang kernel（`*_kernel`），且**没有**启动厂商融合算子 | 单独看名字可以被同名的假 kernel 骗 |

运行期证词用 torch_npu profiler 导出 chrome trace，取事件名。实测：

`@
候选: ["AscendCL@aclrtLaunchKernelWithHostArgs", "main_kernel", "Computing", ...]
厂商: ["AscendCL@aclnnRmsNorm", "RmsNorm", ...]
`@

候选出现 `main_kernel`、**没有** `RmsNorm` / `aclnnRmsNorm` —— 这是把"真的用自己的 kernel 算"从源码层面提升到运行层面的关键一条。

---

## 6. job 证据判据（负例 #3）

以下**每一条单独即 FAIL**，且**不得降级为 SKIP**：

| 条件 | 为什么不能放过 |
|---|---|
| `timedOut` | 超时的 run 仍有日志，半截 run 不能报测量 |
| `interrupted` | 同上 |
| `exitCode != 0` | 非零退出的 run 也可能打印出像模像样的 JSON |
| 缺完成 marker（`task=<mode> finished`） | 可能在分配完张量之后就死了 |
| 无输出 / 输出不是单个 JSON object | 无可读报告 |
| 输出 > 64 KiB | evaluator 读不了 |
| `collected == 0` | 什么都没收集 |
| `skipped >= collected` | 全跳过不是结果 |
| 任一 `observationCheck` 不为 `true` | 观测项为假就是没证明 |

---

## 7. 输出契约

- stdout **恰好一个 JSON object**，全部诊断走 stderr；
- **<= 64 KiB**（evaluator 上限）。超限时按序降级并把丢弃的字段记进 `sizeGuard.dropped`：
  1. 逐窗口 `order` / `index`；
  2. 逐窗口 `witness`；
  3. `authenticity.runtimeWitness`；
  4. `job.report` / `job.record` / `static.imports` / `hostBoundAudit.aggregateDeclared`；
  5. **白名单兜底**：删掉所有非契约顶层键（`schema` / `protocol` / `verdict` / `timing` / `precision` / `authenticity`）。
- **逐 shape 原始样本、区间、判定永不丢弃**；放不下就 raise（拒绝打印一个 evaluator 读不了的对象，比打印一个残缺判定诚实）。

### 7.1 退出码

| 码 | 含义 |
|---|---|
| 0 | MATCH / EXCEED |
| 1 | FAIL（真缺陷：作弊 / 数值错 / 更慢 / 证据无效） |
| 2 | 配置或用法错误：什么都没判 |
| 4 | RETEST：证据不足以判定（**不是失败**） |
| 75 | BLOCKED_ENV：没有可用设备，什么都没测 |

---

## 8. 五个负例（必须真的失败）

实测记录：`evidence/negative-controls-2026-11-02.json`（`scripts/negative_controls.py --json` 重跑）。**五个全部真的被拒**。

| # | 负例 | 期望 | 实测 |
|---|---|---|---|
| 1 | 别名调用 `npu_rms_norm` 的假内核 | 真实性检查拒绝 | `FAIL` @ `authenticity`，命中 2 条 delegation，exit 1 |
| 2 | 故意 bf16 累加 | 精度门拒绝 | 6/6 family FAIL（2.05-5.53 ulp），同时 fp32 臂 PASS（最差占 limit 0.45%） |
| 3 | 缺 marker / 非零退出 / 超时 / 全跳过 | FAIL，不能 PASS | 四种全 FAIL；证据目录不存在 -> exit 2 |
| 4 | `host_bound_ratio [0.04, 0.88]` | device 基线无效 | 最大值 0.88 -> invalid；重算 0.187617 -> valid；**判定翻转确认**；gate -> `INVALID_EVIDENCE` |
| 5 | 性能不足的候选 | FAIL 且保留最佳正确产物 | 0.3146x（0.3006-0.3241），`FAIL @ performance`，`bestCorrectArtifact` 保留 |

负例 #2、#5 使用**本机实测数据**（`artifacts/precision-matrix.json`、`artifacts/probe/`）离线复用，文件里已标注。**#2 的对照臂（fp32 累加）在同一条管线、同一套容差下 PASS**——这是让负例有意义的关键，否则"永远返回 fail"也能通过负例。

---

## 9. 已知限制（必须随结论一起引用）

1. **P2 的 1.5x 在厂商误差大时会放宽**：见 2.4，必须人工复核 ulp 列。
2. **最窄 ulp 分离 2.05x**（64x4096/near_zero_rows），不是 3.4x。
3. **P2 的放宽边界是 0.6667 ulp**；最紧的 `64x4096/outlier_channels` 厂商 0.6009 ulp，只剩 **0.0658 ulp** 余量。6 个组合里已有 3 个处于放宽状态。
4. **运行期证词依赖 profiler**：若 profiler 不可用，`runtimeWitness` 记录 error，此时真实性只靠静态 + 编译两条腿（仍拒绝放行，但证词更弱）。
5. **跨 run 的绝对值不可比**：任何"上次 vs 这次"的快慢对比都是伪结论，只有同窗口比值算数。
6. **本机所有精度/计时数字来自共享机器**：绝对值随租户漂移，比值在同窗口内才稳定。
7. **脚本是 untracked 文件**：证据绑定它的 sha256；回放依赖该文件存在于工作树。

---

## 10. 协议自身的失效模式（这套验收防不住什么）

前面九节只说了"能防什么"。**一份只说自己能防什么的协议会让人误以为它完备。** 下面每一条都是这套 gate **确实防不住**的，引用本协议结论时必须一并考虑。

### 10.1 真实内核里藏了未声明的近似

静态可达性只能证明"没有委托给厂商算子"，**不能证明这个 kernel 对所有输入都算的是同一个函数**。一个真实 TileLang kernel 完全可以在 `if batch == 8192` 分支里走快路径、别的 shape 走慢路径，或者对某类数值范围悄悄降精度。

**能挡住的**：三族 held-out 输入（`normal` / `near_zero_rows` / `outlier_channels`）会打掉一部分——针对公开发布输入过拟合的实现会在这里露馅。

**挡不住的**：如果近似只在你没有构造的第四类输入上触发（例如 2.6 之外的相关性结构、NaN/Inf、极端指数范围），本协议看不见。**根本原因**：判据是对**有限输入集合**的采样，不是对函数全集的证明。要真正覆盖需要做输入空间的差分测试或形式化，超出本协议范围。

### 10.2 cache 命中导致时长偏乐观

验收在**空 cache** 里重新编译（5.4 的编译产物腿），但 msprof/编译缓存之外的层面仍可能偏乐观：

- kernel 二进制缓存命中时，**第一次 launch 的加载成本**可能被摊掉；
- 多 run 复用时，device 上的内存分配器已预热，**首次分配成本**不出现在稳态里。

**对治**：本协议区分**冷启动**与 **warm 稳态**并分开报（3.5），判定只用 warm 稳态——因为**生产环境也是 warm**。但这意味着：**一个只在冷启动更慢的候选会被放过**。若冷启动成本对目标场景重要，必须另外声明并单独测，本 gate 不覆盖。

### 10.3 窗口有 witness 但仍被同卡大流量任务干扰到超出 +/-8%

witness 能告诉你"窗口期间卡不干净"，但**不能定量扣除干扰**。实测：可审计邻接负载下配对比值偏移 3.3%，协议靠**配对 + 中位数 + bootstrap** 吸收；但**如果邻居的流量模式恰好与 A/B 交替同相**（例如它的负载以固定周期波动，正好与你的窗口周期共振），配对就失去保护——两个臂各自落在波峰或波谷的不同位置。

**能挡住的**：p90/p10 > 1.25 会判 RETEST（3.3），极端不稳定会被拒。

**挡不住的**：**系统性同相偏差**。两个臂都稳定地偏同一个方向时，比值是稳的、p90/p10 是小的、bootstrap 区间是窄的——**而结论是错的**。这是本协议最危险的一类失效，因为它伪装成"高质量证据"。**缓解**：对关键结论做**相位随机化**重测（改变 pairs 数或窗口顺序），若结论随相位变化则无效。

### 10.4 候选只在当前 shape 快

已在 3.4 写过，这里重申：**逐 shape 独立判定，禁止几何平均**。一个候选在 8192x4096 上 1.5x、在 64x4096 上 0.6x，**整体判 FAIL**（测试 `test_a_partial_win_does_not_become_a_pass` 锁死）。

**挡不住的**：冻结的两个 shape 之外的 shape。候选可以针对这两个 shape 特化，在其他 shape 上退化。**这是"冻结 shape 验收"固有的局限**，不是实现缺陷——但任何"该候选整体更快"的结论都**不能**从本验收推出，只能从"这两个 shape 上不慢于厂商"推出。

### 10.5 精度门只覆盖两个 shape x 三族输入

2.3 的 6 个组合是**采样**，不是全表。P2 的 1.5x 放宽边界（0.6667 ulp）说明：**厂商误差越大，判定越松**。若某个未测 shape 上厂商误差显著大于 1 ulp，候选的有效容差会被放大（2.4 的残余风险）。**必须人工复核 ulp 列**，不能只看 PASS。

### 10.6 反作弊依赖"归档自包含"这一契约

5.3 的全部论证建立在"候选需要的 helper 必须在归档内"。若候选在**归档外**有可导入的模块（已安装的包、`PYTHONPATH` 上的路径、`site-packages`），静态 walk 看不到它。报告会列出 `externalImports`，但**不会因此拒绝**——因为拒绝所有外部 import 也就拒绝了 `torch` 本身。

**挡不住的**：把计算藏在一个**已安装的第三方包**里，让归档只 import 它。要防这个需要冻结环境并审计每个被导入的包，超出本协议范围。

### 10.7 witness 2（邻居心跳）在单租户环境不可得

本协议原设计用"邻居心跳文件"作为第二个 witness。**这台机器上当前没有可观测的协作邻居**，而**自己起一个进程冒充邻居是循环论证**。

已改用 **驱动自身的进程表**（`npu-smi info` 的 Process 表）作为 witness 2：它由驱动生成、脚本无法伪造、能直接回答"这期间有没有别人的进程在卡上出现/消失/常驻"。**它比心跳文件更强**（不需要邻居配合），但**只覆盖"有没有别的进程"，不覆盖"别的进程有多忙"**——后者仍只能靠 AICore 利用率这个粗粒度读数。

**挡不住的**：一个常驻但负载剧烈波动的邻居。进程表看到它一直在，利用率采样可能每次都撞在低谷。

---

## 11. 关于 pytest 占卡的澄清（记录在案）

窗口期内一次 `npu-smi` 快照里出现 `| 0 | 4066195 | pytest | 700 |`，曾被怀疑是本协议的测试套件占卡。**核实结论：不是。**

证据：

1. 测试文件**不 import torch / torch_npu / tilelang**（唯一出现这些名字的地方是**字符串常量**，用作静态扫描的假源码）；
2. 运行时插桩直接验证：`ASCEND_RT_VISIBLE_DEVICES=""` 下跑完 57 个测试后，`sys.modules` 里**既无 `torch` 也无 `torch_npu`**，因此不可能初始化 device context、不可能出现在 npu-smi 进程表里；
3. 该快照时间戳 `2026-11-02T04:40:45Z`（UTC，= 北京时间 12:40:45）落在设备 job 已被 kill 之后，且同一快照中 npu0 另有 `4072612 python3`、npu1 有 `4076382/4076869` 两个 python3——卡上有多个第三方进程。

**但这不改变操作纪律**：窗口期内任何**可能**占卡的操作（包括 pytest，即使事后证明不占卡）都应先询问。本协议的测试套件现在被明确设计为**离线可跑**，并已用 `ASCEND_RT_VISIBLE_DEVICES=""` 验证在**无设备**环境下通过（57 passed）。
### 3.13 真实性不可判定 ≠ 真实性失败

当候选**完全没有产出测量**，且窗口中**没有观测到任何 kernel**（既无候选自己的 TileLang kernel，也无厂商算子）时，多半是设备在争用下拒绝了发射。此时不能把「沉默」当成委托证据：真实性记为 `inconclusive`，整体判 **RETEST**。

保留 FAIL 的两条路径不变：①确实启动了 kernel 但不是候选自己的；②触发厂商融合算子（`aclnnRmsNorm` 等）。静态可达性分析发现委托时，即使这一轮什么都没跑，仍然 FAIL。
