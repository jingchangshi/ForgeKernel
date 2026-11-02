# Ascend950PR RMSNorm（TileLang-Ascend SimdVF）最终状态

> 生成于第 69 轮。所有性能数字均来自 AgentGates 同窗口配对门控（15 配对、15/15 quiet、bootstrap 95% 下界），逐条可追溯至 `docs/ascend950pr/evidence/` 的原始 JSON。

## 1. 目标与达成情况

| 形状 | 目标 | 实测 | 判定 |
|---|---|---|---|
| **decode 64×4096** | 点估计 ≥1.0 且 95% 下界 ≥0.98 | **1.3726×** / 复现 **1.4056×**，下界 1.3561 / 1.3783 | **EXCEED ✅ 达标（两窗口复现）** |
| **prefill 8192×4096** | 同上 | 四个窗口比值 **0.771–0.850**（均值 0.810），下界最高 **0.829** | **FAIL ❌ 未达标（四窗口均不达标）** |

**结论：目标未完全达成。** decode 明确达标；prefill 仍差约 20%，本报告不把它表述为达标。

## 2. 最终候选

`docs/ascend950pr/candidates/rmsnorm_simdvf_v23/kernel.py`

- 单阶段逐行 RMSNorm，rows=4 路径**源码显式展开**（每行独立累加器与 scale 寄存器）
- 每核连续分区（`base = (core*iters+step)*rows*d`）
- x 四缓冲、输出双缓冲、4 条独立平方累加链、6 级 `vcadd` 归约
- **四条 DMA 路径全部使用 `l2_cache_ctrl="NORMAL_FV"`**
- 精度：两形状 15/15 quiet 门控中 precision 与 authenticity 均 pass；对 CPU float64 相对误差 decode ≤1e-3、prefill ≤3e-3（阈值 2⁻⁷≈7.8e-3）

## 3. 演进轨迹（都是稳定可比的量）

| 版本 | 关键改动 | prefill 候选时长 | decode 点估计 |
|---|---|---|---|
| v4 | 单阶段 + 交错分区 | 0.0801 ms | 1.2809× |
| v5 | 每核连续分区 | 0.0794 ms | （路径同 v4） |
| v15 | 两阶段 + 每行寄存器 | 0.0755 ms | — |
| v16 | + x 四缓冲 | 0.0745 ms | 1.3160× |
| v20/v21 | + L2 策略（全路径） | 0.0690 ms | 1.3497× / 1.3559× |
| **v23** | **stages=4 + 全 `NORMAL_FV`** | **0.0651 ms** | **1.3726×** |

prefill 候选时长累计下降 **19%**；decode 点估计从 1.2809 提升到 **1.3726**。

## 4. 关键正面发现

1. **L2 缓存策略是决定性旋钮**：写路径未设策略时 1.13 TB/s，显式 `NORMAL_FV` 后 **3.08 TB/s（2.7×）**；读路径最佳约 2.35 TB/s。
2. **两阶段 + 每行寄存器**在不引入竞争的前提下把向量工作量拉近厂商（曾多 63%，后仅多 10%）。
3. **每核连续分区**（而非交错步长）消除了跨核跳跃访问。

## 5. 关键否定结论（均有归档证据）

- 两阶段 + **共享缓冲**传 rstd：真实竞争，rows=2 全尺寸错误且 batch=128 不确定 → 已否定（`r59-two-phase-race.json`）
- `annotate_unlimit_memory` 不放宽 UB 启动限制（286720 B 必拒）
- `vsqz`/`vusqz` 不是平方和原语；bf16 `vmula` 无混合精度；`T.gemm` 设备编译失败原因未隔离
- 每行 8 条累加链、rows=2 细 tile、y 三缓冲、读/写异构策略组合：**均不如 v23**
- 微基准排名不能直接搬到内核（读路径微基准最优的 `NORMAL_LV` 在内核中反而更慢）

## 6. 未达标的原因（量化）

- 归因（同窗口）：vec 约 40.5 µs、DMA 双向仍比厂商慢约 18%；总时长 65 µs vs 厂商 53 µs
- 若 DMA 与向量能完全重叠，理论上限约 40–45 µs（对应比值 1.2–1.3），说明**剩余空间在重叠效率**，但 TileLang 当前暴露的流水控制无法进一步压缩
- 此结论限定为"本机 + 本工具链 + 本次实测"，不外推为硬件能力上限

## 7. 复现方式

```bash
source /data/pri/cann-9.2.0/cann/set_env.sh
export PATH=<tilelang-env>/bin:$PATH TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 <AgentGates>/agent_gates/adapters/forgekernel-ascend-npu/scripts/accept_rmsnorm.py \
  --candidate <repo>/docs/ascend950pr/candidates/rmsnorm_simdvf_v23 \
  --shape 8192x4096 --device 0 --pairs 15 --repeat 8 --child-timeout 600 --cache <empty dir>
```

注意：该机为共享卡，多数时间存在**仅存在型**共租户（AICore 峰值低、canary 平坦），门控会正确判 RETEST；需要反复重试直到出现 15/15 quiet 的窗口（`docs/ascend950pr/scripts/patient_gate.sh`）。

## 8. 复现性（第 69 轮独立验证）

| 形状 | 第一窗口 | 独立复现窗口 | 结论 |
|---|---|---|---|
| decode 64×4096 | 1.3726× [1.3561, 1.3829] | **1.4056× [1.3783, 1.4493]** | 两窗口下界均远高于 1.0，达标可复现 |
| prefill 8192×4096 | 0.8074× [0.8006, 0.8343]（候选 0.0651 ms） | **0.8133× [0.7956, 0.8432]**（候选 0.0694 ms） | 两窗口均低于 1.0，未达标可复现 |

原始报告：`evidence/accept-v23-prefill-r69-repeat-raw.json`（sha256 `16dd59c8849da6a1ca00384f2d71e376cb5474df220692f85acf969d5d38af54`）等。

## 9. 本轮新增否定结果

- **x 五缓冲（stages=5）**：候选 0.0809 ms，明显更差 → 四缓冲是正确深度。

## 10. 测量稳定性（重要，R74 更正）

跨 3 个 15/15 quiet 独立窗口：

| 量 | 最小值 | 最大值 | 跨度 |
|---|---|---|---|
| prefill 候选时长 | 0.065149 ms | 0.078896 ms | **21.1%** |
| prefill 厂商时长 | 0.052974 ms | 0.061855 ms | 16.8% |
| **prefill 比值** | **0.7711** | **0.8133** | **5.5%** |
| **decode 候选时长** | 0.002640 ms | 0.002673 ms | **1.3%** |
| **decode 比值** | **1.3550** | **1.4056** | 3.7% |

**正确读法**：

- **prefill**：两侧绝对时长同向漂移约 ±20%，因此**配对比值才是稳定量**（0.771–0.813，慢约 19–23%）。任何小于约 20% 的单窗口差异都不能作为改进证据——这正好解释了 R68–R70 那些"看起来更好"的变体为何都不成立。
- **decode**：候选时长真正稳定（±1.3%），三个窗口的点估计 1.355–1.406，下界均远高于 0.98。

## 11. 差距的物理来源（R73）

| 场景 | 聚合吞吐 |
|---|---|
| 只读 | 2.35 TB/s |
| 只写 | 3.08 TB/s |
| 纯拷贝（读+写） | 1.76 TB/s |
| 我们内核（读+写） | **1.97 TB/s** |
| 厂商算子（读+写） | **2.42 TB/s** |

两个方向各自远未触顶，但并发时聚合吞吐下降；厂商在混合负载下高约 23%（与 20% 的时间差距同量级）。64 核 × (读+写) = 128 条并发 DRAM 流，页局部性差异是最可能的原因，而 **TileLang 未暴露页局部性、流数量或拷贝引擎选择的旋钮**。

## 12. prefill 四窗口终表（R77）

| 窗口 | 比值 | 95% 区间 | 候选 | 厂商 | quiet |
|---|---|---|---|---|---|
| R67 | 0.8074 | [0.8006, 0.8343] | 0.0651 ms | 0.0530 ms | 15/15 |
| R69 复现 | 0.8133 | [0.7956, 0.8432] | 0.0694 ms | 0.0563 ms | 15/15 |
| R74 | 0.7711 | [0.7651, 0.8196] | 0.0789 ms | 0.0619 ms | 15/15 |
| **R77** | **0.8502** | **[0.8294, 0.8830]** | 0.0761 ms | 0.0646 ms | 15/15 |

**四个独立窗口无一达到 1.0**；最好的一次下界也只有 0.8294，距离 0.98 的达标线很远。慢的幅度为 **15–23%**。这是本任务在 Ascend950PR + TileLang-Ascend 上的最终 prefill 结果。

## 13. 最终交付物清单

| 类别 | 路径 |
|---|---|
| 最终候选内核 | `docs/ascend950pr/candidates/rmsnorm_simdvf_v23/kernel.py` |
| 最终汇报（本文） | `docs/ascend950pr/final-status.md` |
| 不可达性评估 | `docs/ascend950pr/d3-assessment.md` |
| 归档清单（含 sha256） | `docs/ascend950pr/archive-manifest.json` |
| 逐轮日志 | `docs/ascend950pr/plan-40-rounds.md` |
| 原始门控报告 | `docs/ascend950pr/evidence/accept-*-raw.json`（20+ 份） |
| 复现脚本 | `docs/ascend950pr/scripts/{patient_gate,final_verify,repro_check}.sh` |
| 验收门（含本轮修复） | AgentGates `agent_gates/adapters/forgekernel-ascend-npu/` |
