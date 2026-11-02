# Ascend950PR RMSNorm（TileLang-Ascend SimdVF）最终状态

> 生成于第 69 轮。所有性能数字均来自 AgentGates 同窗口配对门控（15 配对、15/15 quiet、bootstrap 95% 下界），逐条可追溯至 `docs/ascend950pr/evidence/` 的原始 JSON。

## 1. 目标与达成情况

| 形状 | 目标 | 实测 | 判定 |
|---|---|---|---|
| **decode 64×4096** | 点估计 ≥1.0 且 95% 下界 ≥0.98 | **1.3726×** / 复现 **1.4056×**，下界 1.3561 / 1.3783 | **EXCEED ✅ 达标（两窗口复现）** |
| **prefill 8192×4096** | 同上 | 候选 **0.0651 / 0.0694 ms**，比值 **0.807 / 0.813** | **FAIL ❌ 未达标（两窗口复现）** |

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
