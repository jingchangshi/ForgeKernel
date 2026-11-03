# 如何运行 tla-npu adapter：从裸机到 PASS 的完整路径

**作者**：lead。**日期**：2026-11-03。
**性质**：运行手册（runbook）。**本文含一次实测运行**（run id 见 §7），其余为源码与配置核对。
**适用**：/home/s00653124/workspace/AgentGates/agent_gates/adapters/tla-npu/，本机（Ascend950PR_9579）。

标签：**[verified: 路径:行]** / **[measured]** / **[source-only]** / **[inferred]**。

> **本文与 `README.md` 的关系**：那份文档回答"该不该迁"；本文只回答"怎么把它跑起来"。

---

## 0. 一句话

**跑一次只要一条命令，但前提是四样东西齐全：容器、容器内的 conda 环境、`env.sh`、以及
一次性的 `--device-id` 授权。** 四样都已在本机就位，所以现在直接：

    cd /home/s00653124/workspace/AgentGates
    ./.venv/bin/agent-gates run --adapter tla-npu --gate tla-npu-e2e \
        --source /home/s00653124/workspace/triton-ascend --device-id 1

**[measured]** 实测结果：`PASS  run 9e8c59f1c356badc32a0e0d396af4659`，容器内 runner 耗时 12.45 s。

---

## 1. 运行架构：谁在哪里跑

这是最容易搞错的一点——**门脚本本身在主机上跑，它再去 `docker exec` 进容器跑测试**。

    agent-gates run            (主机, AgentGates .venv)
      └─ python3 tla_gate.py   (主机, 见 recipe/process-config.json)
           └─ docker exec -u root -e HOME=... -w <workdir> s00653124_tla bash -c '<script>'
                └─ conda activate s00653124_tla; python3 - test_tla.py
                     └─ 真实 NPU 执行 + 编译

**[verified: tla_gate.py:1091]** prefix 构造，**:1104** subprocess.run。

三个要点：

1. **`tla_gate.py` 在主机执行**，不在容器里 [verified: recipe/process-config.json 的 argv 是主机路径]。
   所以它需要主机的 `python3`，且主机的 CANN/conda 与它无关——环境准备发生在容器侧。
2. **`-u root` 不是随意的。** 代码注释说明：Ascend 驱动按 user namespace 持有 davinci0/1
   （`uda_occupy_dev_by_ns`），一旦某 namespace 占住卡，非 root 打开会被 EBUSY 拒绝，
   即使它是 device 组成员。root 不受该检查约束 [verified: tla_gate.py:581-590]。
   **这是本手册最容易踩的坑**：以普通用户在容器里手动跑 `python3` 会遇到完全不同的报错。
3. **`-e HOME=...` 也必须带上。** 少了它，容器内 Python 找不到 `/data/miniconda3` 下的
   site-packages 路径。**[measured]** 我本人踩过这个坑，现象是
   `ModuleNotFoundError: No module named 'pybind11'`。

---

## 2. 前置条件清单（四样，缺一即 BLOCKED_ENV）

| # | 依赖 | 本机状态 | 如何检查 |
|---|---|---|---|
| 1 | 容器 `s00653124_tla` | **运行中（Up 29 hours）** | `docker ps --filter name=s00653124_tla` |
| 2 | conda 环境 `s00653124_tla` | **存在** | `ls ~/.conda/envs/s00653124_tla` |
| 3 | `env.sh`（git-ignored） | **存在，2603 B** | `ls agent_gates/adapters/tla-npu/env.sh` |
| 4 | `--device-id` 授权 | **运行期参数，不落盘** | 见 §3 |

**[verified]** 前三条均实查；第 4 条是设计选择，见 §3。

### 2.1 `env.sh` 的关键键 [verified: env.sh 实读]

| 键 | 本机值 | 说明 |
|---|---|---|
| `TLA_NPU_CONTAINER_NAME` | `s00653124_tla` | 必须与 `docker/config.env` 一致 |
| `TLA_NPU_CONDA_ENV` | `s00653124_tla` | **注意与 ForgeKernel 的 `s00653124_tilelang` 不是同一个** |
| `TLA_NPU_CONDA_ENVS_PATH` | `/home/s00653124/.conda/envs` | 供 `conda activate` 定位 |
| `TLA_NPU_CANN_SETUP` | `/data/pri/cann-9.1.0/set_env.sh` | **本机存在**（9.1.0，非 9.2.0） |
| `TLA_NPU_TEST_DIR` | `/home/s00653124/workspace/agent_tla` | 必须有 `test_tla.py` |
| `TLA_NPU_CACHE_DIR` | `~/.cache/tla-npu-gate` | **不得**位于 source 或 test 树内 |
| `TLA_NPU_WHEEL_DIR` | `/home/s00653124/Packages` | 仅 `setup-python-env.sh` 用 |
| `TLA_NPU_TRITON_WHEEL` | 钉定某个 dev 构建 | 见 §6.3 的踩坑 |

---

## 3. 为什么 `--device-id` 是运行期参数

`env.sh` 里**故意不含** `TLA_NPU_DEVICE_ID`。第一次不带它运行时，doctor 返回：

    BLOCKED_ENV  run 95336d0b...
      CONFIG_CHECK_FAILED: tla-npu.e2e
      stderr: tla-npu-gate: missing required environment variable TLA_NPU_DEVICE_ID

**[measured]** 这是我第一次尝试的真实结果。原因写在 README 里 [verified: README.md:58-60]：
该值 "authorizes one run, not a machine"，因此不属于机器配置。

三种给法（任一）：

    --device-id 1                                    # 推荐，显式
    TLA_NPU_DEVICE_ID=1                              # 环境变量
    TLA_NPU_ENV_FILE=.../env.sh TLA_NPU_DEVICE_ID=1  # 直调 gate 脚本时

**本机用 1**：device 0 报 Alarm，1 是健康卡（本项目其余所有测量也都在 1 上）。

---

## 4. 三条命令：doctor → run → verify

### 4.1 先探一次：doctor（不跑真门，只查配置与机器）

    cd /home/s00653124/workspace/AgentGates
    ./.venv/bin/agent-gates doctor \
        --adapter tla-npu \
        --source /home/s00653124/workspace/triton-ascend \
        --device-id 1

**[measured]** 输出：`PASS  run 6175f074...`

`--source` 是被测对象（`triton-ascend` 检出），**不是**测试目录。两者是两个不同的输入：
测试在 `TLA_NPU_TEST_DIR`，源码在 `--source` [verified: tla_gate.py:349-366 的注释明确解释
"Deliberately not the source root"]。

### 4.2 跑真门：run

    ./.venv/bin/agent-gates run \
        --adapter tla-npu --gate tla-npu-e2e \
        --source /home/s00653124/workspace/triton-ascend \
        --device-id 1

**[measured]** 输出：

    PASS  run 9e8c59f1c356badc32a0e0d396af4659
    Receipt: /home/s00653124/.local/state/agent-gates/runs/9e8c59f1c356badc32a0e0d396af4659/receipt.json
      tla-npu-e2e: PASS

命令形态 [verified: `agent-gates run --help`]：
`run [--json] [--evidence-root DIR] --adapter ADAPTER --source SOURCE [--env-file FILE] [--device-id ID] (--gate GATE | --all)`

- `--gate tla-npu-e2e`：跑指定门；`--all` 跑该 adapter 全部门（本 adapter 只有一个）。
- `--env-file`：不给时读 adapter 目录下 git-ignored 的 `env.sh`。
- `--json`：结构化输出，便于脚本化。

适配器清单 [measured: `agent-gates list`]：

    ascendnpu-ir: ascend-build, ascend-regression
    forgekernel-ascend-npu: forgekernel-ascend-npu-smoke, forgekernel-ascend-npu-build
    tilelang-hivm: tilelang-hivm-build, tilelang-hivm-regression
    tilelang-hivm-npu: tilelang-hivm-npu-sanity
    tla-npu: tla-npu-e2e

### 4.3 复核回执：report / verify

    ./.venv/bin/agent-gates report --run-id 9e8c59f1c356badc32a0e0d396af4659
    ./.venv/bin/agent-gates verify --run-id 9e8c59f1c356badc32a0e0d396af4659

**[measured]** 两者均 `PASS`。

**两者的区别值得说清**：`report` 读已发布的回执；`verify` **就地重新校验**回执里的哈希与机器状态。
前者是"当时记了什么"，后者是"现在还算不算数"。

---

## 5. 回执里到底记了什么（一次实测的完整读数）

**[measured] run 9e8c59f1c356badc32a0e0d396af4659**，11 个 check **全 true**：

| 字段 | 值 | 意义 |
|---|---|---|
| `castBackOk` / `tritonCastBackOk` | true / true | 两条路径都算出正确结果 |
| `tensorMovesToDevice` | 6 | 进程内计数，真搬到设备 |
| `npuPeakMemoryBytes` | 50366464（约 48 MB） | 设备真的分配了内存 |
| `compiledArtifactsPresent` | true（`.o` 数 = 0） | 产出了 `.npubin`；`.o` 只计数不要求 |
| `npuDeviceUsed` | true | 由 `npuPeakMemoryBytes > 0` 推出 |
| `deviceCountOne` | true | 单卡可见 |
| `synchronizeOk` | true | 设备同步成功 |
| `deviceName` | `Ascend950PR_9579` | 记录，不门控 |
| `tritonVersion` | `3.6.0` | 记录 |
| `tlaImplementationFileCount` | 21 | TLA 实现文件数 |
| `tlaImplementationSha256` | `15090006…` | **绑定到产生本次 PASS 的那份 TLA 代码** |
| `tlaHasDma` | true | tla.dma 存在 |
| `visibleDeviceEcho` | `1` | 记录，不门控 |

源码身份：`head 2a00c605…`、`git-files` 模式、3275 文件 [verified: receipt.sourceBefore]。

**注意 `deviceUuid` / `deviceName` / `visibleDeviceEcho` 是"记录但不门控"**——
本机两张卡报同一 UUID，门它等于门一个常量。这写在代码注释里 [verified: tla_gate.py:949-975]，
是这道门诚实性的核心证据之一（详见 `adapter-audit.md` §4）。

---

## 6. 若要从零重建环境

三步，全是**操作者动作，门永远不会替你调用** [verified: docker/README.md 开头明写]。

### 6.1 建镜像与容器

    cd agent_gates/adapters/tla-npu/docker
    docker build -t tla-npu-dev:24.04 .
    cp config.example.env config.env      # 按本机身份修改
    ./create-container.sh --config ./config.env --dry-run
    ./create-container.sh --config ./config.env

`config.env`（本机已有）关键值：镜像 `tla-npu-dev:24.04`、容器名 `s00653124_tla`、
`PRIVILEGED=true`、`NETWORK=host`、`BIND_MOUNTS="/data:/data /home:/home"`、
`ASCEND_VISIBLE_DEVICES=0,1`。

镜像**不装** LLVM/clang/cmake/ninja：厂商 Triton wheel 自带预编译 LLVM 与 BishengIR，
容器内不编译 C++ [verified: docker/README.md、Dockerfile 97 行]。

### 6.2 装 Python 环境（容器内，以容器用户）

    bash agent_gates/adapters/tla-npu/scripts/setup-python-env.sh

它读同一份 `env.sh`，从 `TLA_NPU_WHEEL_DIR` 装厂商 `torch` / `torch_npu` / `triton_ascend`，
最后**验证** `triton.language.extra.tla` 导出该门需要的 API 且 NPU 可见，否则失败退出，
不留半可用环境 [verified: setup-python-env.sh]。

### 6.3 wheel 钉版——这里有个真实的坑

**不能"取最新"。** 两个同日构建的对照 [verified: README.md:139-158]：

| wheel | 结果 |
|---|---|
| `...post20261030+gita414a5cb` | **不可用**：连 `tla.jit` 都没有；native library 只导出 34 个 builder 中的 9 个 |
| `...dev20261030+git0c9d99ca` | **可用**：跑到两个 marker，零 mismatch |

因为 TLA API 跨厂商构建不稳定，所以 `TLA_NPU_TRITON_WHEEL` **按名钉定**，取最新只是回退路径。

> **一个必须更正的旧说法**：我此前判断"本机没有可安装的 TLA"。**那不准确。**
> 本机 `~/.conda/envs/s00653124_tla` 里装的是
> `triton_ascend-3.6.0.dev20261101+git2a00c605`，**它含完整 tla 且能跑**（§7 实测）。
> 我当时的检索只看了 `s00653124_tilelang` 环境与磁盘上的 wheel 文件，漏掉了这个 env。
> 正确的表述是：**默认的 tilelang 环境里没有 tla，需要那个专用 conda 环境 + 容器 + root。**

---

## 7. 一次完整的实测记录

**[measured] 2026-11-03**

    $ ./.venv/bin/agent-gates run --adapter tla-npu --gate tla-npu-e2e \
          --source /home/s00653124/workspace/triton-ascend --device-id 1
    PASS  run 9e8c59f1c356badc32a0e0d396af4659
      tla-npu-e2e: PASS

- 容器内 runner 耗时 **12.45 s**（`receipt.json` → `gates[0].process.elapsed`；注意 `recipe/process-result.json` 只有退出码，没有 elapsed）
- 11/11 check 通过，`tlaImplementationSha256 = 15090006ba89d18f1e7f52ffd6f1402d2cf2cb116d1cec945a2df1aa722c258b`

**容器内直接验证 TLA 可用**（绕过门，手工）：

    docker exec -u root -e HOME=/home/s00653124 \
        -w /home/s00653124/workspace/agent_tla s00653124_tla bash -c '
      source /data/miniconda3/etc/profile.d/conda.sh
      export CONDA_ENVS_PATH=/home/s00653124/.conda/envs
      conda activate s00653124_tla
      python3 -c "from triton.language.extra import tla; print(tla.__file__)"'

**[measured]** 输出 tla 路径，且 `hasattr(tla,"jit")` 与 `hasattr(tla,"dma")` 均为 `True`。

---

## 8. 手动跑单步（调试用）

| 目的 | 命令 |
|---|---|
| 只查配置 | `TLA_NPU_ENV_FILE=<abs>/env.sh TLA_NPU_DEVICE_ID=1 python3 -B <pack>/scripts/tla_gate.py --check-config` |
| 直跑门脚本（跳过 AgentGates） | 同上，去掉 `--check-config`，加 `--timeout 1200` |
| 看适配器清单 | `agent-gates list` |
| 看上次运行产物 | `~/.local/state/agent-gates/runs/<run-id>/` 下有 `receipt.json`、`tla-npu-e2e/recipe/{stdout,stderr}.log` |

**[measured]** `--check-config` 输出：

    tla-npu-gate: configuration ready: identity file .../env.sh
      container=s00653124_tla home=/home/s00653124 device=1 npuId=1
      occupancy=shared(4) exclusive=False

最后一项说明卡是**共享**的（4 路占用）。门**报告而不拒绝**——因为这类机器上卡就是共享的，
为别人的占用挡住自己的测试是不合理的 [verified: README.md:65-70]。
要让占用变成 `BLOCKED_ENV`，设 `TLA_NPU_REQUIRE_EXCLUSIVE_DEVICE=1`。

---

## 9. 常见失败与定位

| 现象 | 原因 | 处置 |
|---|---|---|
| `missing required environment variable TLA_NPU_DEVICE_ID` | 没给 `--device-id` | 加 `--device-id 1`（§3） |
| `BLOCKED_ENV / CONFIG_CHECK_FAILED` | `--check-config` 非零退出 | 读 `runs/<id>/doctor/tla-npu.e2e/stderr.log` |
| 容器里 `import triton` 报 `No module named 'pybind11'` | **用了普通用户或少了 `-e HOME`** | 用 `-u root -e HOME=<container_home>`（§1 第 2、3 点） |
| `conda activate ... failed` | `CONDA_ENVS_PATH` 未导出或 env 名不符 | 检查 `env.sh` 的 `TLA_NPU_CONDA_ENV(S_PATH)` |
| wheel 装了但 `@tla.jit` 报 `AttributeError` | 装上了**不可用**的那个构建（§6.3） | 按名钉 `TLA_NPU_TRITON_WHEEL`，别取最新 |
| `TLA_NPU_CACHE_DIR ... is inside the ...` | 缓存目录落在源码/测试树内 | 换到树外（本机 `~/.cache/tla-npu-gate`） |

**通用定位原则**：这道门**缺失数据一律为假**，所以"没有输出"会表现为 FAIL 而不是 PASS
[verified: `observations()` docstring, tla_gate.py:906]。看到 FAIL 时先找"哪一项没有读数"，
而不是先怀疑数值。

---

## 10. 边界：这道门不做什么

**必须说清，否则容易误用**（详见 `adapter-audit.md` §3）：

| 它做 | 它不做 |
|---|---|
| 跑**固定**的 `test_tla.py` | 跑**你的**内核。`GATE_TEST='test_tla.py'` 是模块级常量 [verified: tla_gate.py:124]，`parse_argv` **只**接受 `--timeout` [verified: :565-573] |
| 判定 11 个布尔 check | 判定性能。**拒绝**把速度阈值做成门（设计选择，见 README/PLAN §3） |
| 记录 `tlaImplementationSha256` 等 | 用它们门控（它们是"记录不门控"，§5） |
| 报告卡占用 | 强制独占（除非显式要求） |

**所以"跑通 tla-npu"= 证明这台机器能编译并正确执行一个 TLA 内核。
它不等于"我们的 rmsnorm 已在 TLA 上被验收"。** 后者需要改门的契约（`adapter-audit.md` §3 给了最小改造面）。

---

## 11. 复现本文所有核对

    cd /home/s00653124/workspace/AgentGates

    # 1) 适配器已注册
    ./.venv/bin/agent-gates list

    # 2) 配置就绪（含容器/conda/卡占用）
    ./.venv/bin/agent-gates doctor --adapter tla-npu \
        --source /home/s00653124/workspace/triton-ascend --device-id 1

    # 3) 跑门
    ./.venv/bin/agent-gates run --adapter tla-npu --gate tla-npu-e2e \
        --source /home/s00653124/workspace/triton-ascend --device-id 1

    # 4) 复核
    ./.venv/bin/agent-gates report --run-id <run-id>
    ./.venv/bin/agent-gates verify --run-id <run-id>

    # 5) 门为什么只能跑固定文件
    grep -n "GATE_TEST\|parse_argv" \
        agent_gates/adapters/tla-npu/scripts/tla_gate.py

    # 6) 容器侧环境可用性（注意 -u root -e HOME）
    docker exec -u root -e HOME=/home/s00653124 \
        -w /home/s00653124/workspace/agent_tla s00653124_tla bash -c \
        'source /data/miniconda3/etc/profile.d/conda.sh; \
         export CONDA_ENVS_PATH=/home/s00653124/.conda/envs; \
         conda activate s00653124_tla; \
         python3 -c "from triton.language.extra import tla; print(tla.__file__)"'
