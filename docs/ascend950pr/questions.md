# 实施前待确认事项（2026-11-02）

本文件只列需要你决定、且决定错了会造成返工的问题。每条给了我的建议；回复“同意”或指出要改的编号即可。
下面 A 部分是已自行核实的事实，其中 A1、A2 会改变你对结果的预期，请先看。

## D. 已确认（2026-11-02 用户答复）

| 编号 | 决定 |
|---|---|
| B1 | 共享机器：所有卡都可使用，不必固定卡0；他人用卡不代表不可用。已据此改掉独占与安静卡的设计（见 plan.md 0.1）。 |
| B4 | 只在真正跑 kernel 时占用，不整段独占。设备租约改为窗口级协作式。 |
| B5 | 优化算子的 agent 使用 magpie 网关 auto/claude-sonnet-5-5（canary 通过）。 |
| 其余 | 按建议执行。 |

## A. 已核实，不需要你回答

| 项 | 结论 | 证据 |
|---|---|---|
| NPU 健康 | 卡0、卡1 均 OK | npu-smi 25.7.rc1 |
| 卡1 占用 | 其他用户 l00898962 的 compare.py 在跑（599MB） | ps |
| 卡0 | 可用，本轮全部实测都在卡0 上跑通 | 探针 |
| 磁盘 | 8.7T 可用（68% 已用） | df |
| 模型 API | magpie 网关 canary 通过，claude-opus-5-5 回 CANARY_OK | canary_provider.py |
| BF16 可编译 | 8192x4096 / 64x4096 均编译通过且数值正确 | probe-bf16-2026-11-02.json |
| 计时 basis | TileLang msprof 后端给出逐 kernel 设备时长，天然 host-free | msprof.py |
| FK 测试基线 | 12 failed / 2041 passed（非全绿） | fk-baseline-pytest-2026-11-02.log |

### A1【改变预期】起跑差距是 3.18 倍，不是接线就能达标

| shape | 仓库示例内核 | 厂商 | 相对厂商 |
|---|---|---|---|
| 8192x4096 bf16 | 0.1634 ms | 0.0514 ms | 0.315x（慢 3.18 倍） |
| 64x4096 bf16 | 0.004683 ms | 0.004019 ms | 0.858x（慢 1.17 倍） |

意思是：能跑通 ≠ 能达标。要持平厂商，必须真的做昇腾内核优化（每核行数、UB 预算、流水、reduce 方式、VF 切分）。
方案里的“打通”只能保证闭环可运行、可测量、可复验；性能达标是第二阶段的工作量。请确认你接受这个预期。

### A2【重要】测试基线本来就不是全绿

CANN 已 source 且 TORCH_DEVICE_BACKEND_AUTOLOAD=0 时：12 failed / 2041 passed。
- 2 个是 safetensors 未安装；
- 10 个是 driver/harness_guard 的顺序依赖与 git 权限问题，属仓库 HEAD 既有状态（commit 5030503）。

所以验收口径只能是“不新增失败”，不能是“全绿”。若你要求基线全绿，需要先修 FK 仓库本身，那是另一件事。

## B. 需要你决定

### B1 设备分配：固定用卡0 吗？
- 现象：卡0 HBM 显示已用 7849MB，但进程列表为空（可能是残留）；卡1 有别人在跑。
- 需确认：卡0 是分配给你、且允许长时间占用吗？还是应随空闲情况动态选卡？
- 建议：机器配置里显式写死卡0，每次 run 前重查占用；不自动漂移，被占即 BLOCKED。

### B2 允许改 AgentGates 核心代码吗？
新 adapter 不是放个目录就行，必须在 core 加映射：engine.py 的 ENV_KEYS / OWN_PREFIX / DEVICE_ADAPTERS / EXCLUSIVE_VARIABLE、machine_env.py 的键集合与 ADAPTERS、pyproject.toml 的打包清单与 tests 打包断言。
- 建议：允许改，但限定在这几处映射 + 对应测试，不改既有 adapter 行为，每条改动配一个测试。

### B3 补丁放哪里、怎么审？
(a) 补丁文件存在 adapter 内（patches/NNNN-*.patch），apply 到独立工作树，FK 仓库不动；
(b) 在 ForgeKernel 仓库开分支 ascend950pr 提交，adapter 只记录 commit。
- 建议：两者结合 —— FK 仓库开分支便于你逐条 review diff，adapter 同时保存 patch 文件与 SHA 保证可复现。你更希望哪种？

### B4 首次优化运行的时间预算与占用
当前 models.toml：teacher 单轮 14400s、student 7200s、budget 4h。
- 需确认：首次真正跑，允许连续占用卡0 多久？（4 小时？还是先只跑 1 小时 smoke 级？）
- 建议：先跑 60–90 分钟受控 run 验证全链路，再放开到 4 小时；期间卡0 不与他人共享。

### B5 模型与成本
沿用 magpie 网关 + claude-opus-5-5，teacher effort=high。
- 需确认：模型与 effort 就用现状？有无 token / 费用上限要加进预算门？
- 建议：沿用现状，但在 adapter 加“每轮 token 与总调用数”上限，超限即停并记录。

### B6 网络与依赖策略
- 需确认：优化期间 student 是否允许联网（查昇腾文档）？
- 建议：online，因为本任务需要查 TileLang-Ascend / CANN 文档；但依赖 frozen（不在优化回合 pip 升级共享环境），正式 GPU 执行仍走 FK 记录的 submit 通道。
- 另：要不要装 safetensors 让测试基线少 2 个失败？建议装，纯 Python 包，不影响工具链。

### B7 性能验收口径
逐 shape 判定，不用几何平均掩盖：
- 达标（MATCH）：每个 shape 点估计 ≥ 1.0，且配对 bootstrap 95% 下界 ≥ 0.98（2% 等价容忍）。
- 超越（EXCEED）：每个 shape 的 95% 下界 > 1.0。
- 需确认：这个口径可以冻结吗？还是要求更严（下界 ≥ 1.0 才算达标）？

### B8 decode 这个 shape 的定位
64x4096 实测在 4µs 量级，基本由 launch/固定开销主导（我们 4.68µs vs 厂商 4.02µs）。
- 需确认：decode 必须是硬门槛吗？还是可作次要指标（例如要求不退化超过 X%，而非严格超越）？
- 建议：两个 shape 都要过 MATCH，但 decode 用“不慢于厂商 + 2% 容忍”判定，因为 4µs 级差异极易被噪声主导。

### B9 实施顺序：先纵切还是先大改
(a) 先做最小纵切（adapter 骨架 + P1/P2/P3 + BF16 smoke），让你看到一次真实 agent 回合和一次真实 job，再推进 eval 设备化（leaderboard 2767 行、timing 913 行）与其余补丁；
(b) 一次性把 9 个补丁 + adapter 都做完再给你看。
- 建议：(a)。理由：A1 的 3.18 倍差距说明真正的风险在内核优化阶段，先把测量与闭环跑通拿到真实反馈，比先堆完所有补丁更有价值。

### B10 团队分工确认
拟三层分工，写作用域互不重叠：
1. fk-patch：ForgeKernel 补丁（P1–P9），只写 FK 分支；
2. ag-adapter：AgentGates adapter + core 映射，只写 AgentGates；
3. accept-verify：独立验收脚本、厂商基线协议、负例测试，只写 adapter 的 tests/ 与 scripts/accept*；
4. Lead 负责最终 diff 复核、真实设备 run 与收口。
- 需确认：这个划分可以吗？是否要限定成员数量或模型？

## C. 默认假设（若不反对就按这些执行）

1. 卡0 专用，每 run 前重查占用，被占即 BLOCKED 不换卡。
2. 允许改 AgentGates core 的映射与测试。
3. 补丁双轨：FK 分支 + adapter patch 文件与 SHA。
4. 先 60–90 分钟受控 run，再放开预算。
5. 沿用 claude-opus-5-5 / teacher high effort，加 token 上限。
6. online + 依赖 frozen，装 safetensors。
7. 逐 shape 判定，下界 0.98 容忍。
8. 先纵切再大改。
9. 三个 teammate + Lead 收口。
