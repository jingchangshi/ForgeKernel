# ForgeKernel × TileLang-Ascend：Ascend950PR 适配设计与验收记录

本目录是 ForgeKernel 适配昇腾 Ascend950PR 的**设计、实施记录与测量证据**。
首轮"打通"已验收；**RMSNorm 性能尚未达标**（见 [性能报告](perf-report.md)）。

> **唯一源约定**：验收协议在两处各有一份副本，**AgentGates 仓库中的副本是唯一源**
> （`agent_gates/adapters/forgekernel-ascend-npu/`）。本目录的
> [acceptance-protocol.md](acceptance-protocol.md) 是镜像，如需修改请改 AgentGates 侧再同步。

## 状态

| 项 | 状态 |
| --- | --- |
| ForgeKernel 补丁（P1/P2/P3） | **完成**，分支 `ascend950pr` |
| AgentGates adapter `forgekernel-ascend-npu` | **完成**，AgentGates 仓库 |
| 真卡 build + smoke gate | **双双 PASS**（8/8、9/9） |
| FK 端到端驱动昇腾 job | **已验证**（真实 job 链路） |
| 验收协议 | **完成**（74 测试通过；排除规则经回归探针验证） |
| **RMSNorm 性能达标** | **未达成**：预填充 0.3224×（FAIL）、解码 0.9939×（RETEST） |
| P4–P9 补丁组 | **未开始** |

## 文档

- [完整方案](plan.md)：adapter 职责、TileLang 构建复用、FK 补丁、BF16 任务、计时与厂商性能门槛、实施顺序。
- [实施前待确认事项](questions.md)：需决定的问题与两个会改变预期的事实（已答复）。
- [实施风险登记表](pitfalls.md)：历史问题、更正与当前状态（含"Lead 被协作者纠正"的元级教训）。
- [RMSNorm 性能报告与优化轨迹](perf-report.md)：起点 0.3148×、管道归因、row 扫描 1.76×、争用偏差 2.05×、最终判定与优化方向。**区分"已落地"与"已定位"**。
- [验收协议](acceptance-protocol.md)：12 节，含判定样本排除规则与协议自身的失效模式。**镜像，唯一源在 AgentGates**。

## 证据与收据

`evidence/` 下为全部测量的原始记录；`artifacts/` 下为关键收据。文档中的数字可追溯到这两个目录。

| 文件 | 内容 |
| --- | --- |
| `evidence/probe-bf16-2026-11-02.json` | 真卡编译、数值、性能差距、计时协议、精度门、反作弊、采样量、ulp 界、工具可用性、共享机租户与争用偏差 |
| `evidence/e2e-fk-ascend-2026-11-02.json` | `kind=npu` → `ASCEND_RT_VISIBLE_DEVICES` → `npu0` 锁 → job Succeeded → TileLang 编译执行 → msprof 时长 → 完成 marker |
| `evidence/precision-ulp-matrix-2026-11-02.json` | 6 组合 ulp 全表 |
| `evidence/negative-controls-2026-11-02.json` | 5 个负例的真实拒绝输出 |
| `evidence/contention-positive-control-2026-11-02.json` | 争用偏差 2.05× 与方向反转 |
| `evidence/exclusion-rule-replay-2026-11-02.json` | 剔除规则的回归探针与旧数据重放 |
| `artifacts/accept-genuine.json` | 干净 run 的正式性能数据 |
| `artifacts/contended-run.json` | 争用 run（仅作 witness 验证，不作性能结论） |

## 脚本

[scripts/](scripts/) 下为可复现测量与校验脚本，路径已改为可移植（见 `scripts/_paths.py`），
并通过环境变量 `FORGEKERNEL_ROOT` / `TILELANG_ROOT` / `AGENTGATES_ROOT` 允许覆盖。

重跑示例（本目录下执行）：

```bash
source "$CANN_ENV"            # CANN 的 set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 scripts/inspect_environment.py > evidence/environment-next.json 2> evidence/environment-next.stderr.log
```

## 说明

- 本目录只放**文本与小型 JSON**：`cache/`、`artifacts/*/` 等可重建的编译缓存已由 `.gitignore` 排除。
- 早期证据在 `/tmp` 丢失后按会话重建，保留了更正痕迹，**不伪造旧实验日志**。