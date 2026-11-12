# 上游对齐与分批优化设计（2026-11-12）

状态：设计已获批准，待生成实施计划。
范围：用户选择「全部分阶段做 —— 先完成完整设计与分批路线，再从低风险项开始逐批实施」。

## 0. 背景与关键发现

- 上游项目：`/Users/apple/Downloads/lean/ProjectionPhysics`，HEAD `3a7775a27b52581aa6a04d50f58c2c4e5d61277e`。
  本仓 `docs/ALIGN-LEAN.md:4` 称其为 Hibs-Physics 并引用提交 `d28cd41`；
  `docs/S2-VERDICTS.md:102`、`docs/S3-NONLOCALITY.md:3`、`gym_mu_frc/s3_nonlocality.py:4`
  引用 `77c794f`，均已过期。
- 上游新增（2026-11-12）`docs/wiki/fusion-sim-workbench.md`，并已有一套更完整的
  `fusion_sim/` 工作台：`backends.py`（JAX/NumPy 抽象）、`config.py`（JSON 往返）、
  `transport.py`（1-D 后向 Euler 输运）、`envs/mu_frc_env.py`（obs 8 / act 2）、
  `torax_adapter.py`（如实降级）、CLI 与 FS1–FS15 共 47 条门禁。它是本次优化的规格来源。
- 上游其它直接依据：`docs/wiki/theory-mu-topology.md`（非局部性、指数和 e、阈值能量
  1.23e6 / 4.94e6 / 1.23e9 J、第二输入缺口）、`docs/wiki/theory-mu-power.md`
  （`W = ε·gross(1−μ)`、ε ≈ 6.17e8 J、`gross ≡ net (mod 2)`）、
  `docs/wiki/fusion-program-roadmap.md`（G0/G1、1.29%、μ 阶梯）、
  `docs/wiki/design-minimal-mu-experiment.md`（D1 判决量）。
- 本仓存在未提交改动：`gym_mu_frc/s3_nonlocality.py` 的 `reconnection_event` 新增第 5 个参数
  `band: float = 0.30`，并把重联电阻带从 0.6 改为 `band`。该改动**必须保留**。
- 已知缺陷：`gym_mu_frc/s3_nonlocality.py:369` 在干净合并分支从 `hist` 取 `欧姆耗散` 键会
  KeyError；`scripts/verify_plan_gates.py:31` 与 `scripts/verify_s2_gates.py:31` 把
  `HUSH` 硬编码到本机 `/Users/apple/Downloads/hushfusion`，导致远端 T7/S2-7 静默跳过。

## 1. 事实来源、架构原则与验收约束

- **单一事实来源**：`/Users/apple/Downloads/lean/ProjectionPhysics` @ `3a7775a…`。
  本仓文档中的 commit 引用统一改为「路径 + HEAD + 文件 + 校验日期」，不再写死旧 commit。
- **两套独立实现互证**：遵循 `docs/ALIGN-LEAN.md:5`，**不整体复制**上游 `fusion_sim/`；
  只移植口径、常量、接口契约与门禁设计。`gym_mu_frc/` 继续是独立实现，G1 继续数值互证。
- **新增不替换**：`ledger` 代价、`v0.1` 环境、配置表新字段一律以新增键 / 新增类 / 新增门加入；
  `work`/`locking`、旧 `MuFrcEnv`、现有 JSON 键全部保留。
  特别是 `gym_mu_frc/gates.py:340-344` 的 G7b 会读 `docs/ALIGN-LEAN.md` 中
  `[work]` / `[locking]` 两行「已知缺口 · 未解决」，必须原样保留。
- **S3 与 band 兼容**：`band` 保持第 5 个参数、默认 `0.30`，后续新参数一律加在它之后并给默认值；
  `scripts/verify_s3_gates.py` 用 `d.get("band", 0.6)` 兼容旧 `report.json`。
  `band=0.30` 很可能首次跑出 `N: 2→1`，进入 `scripts/verify_s3_gates.py:78-85` 的
  `has_event` 分支，因此必须**同批修掉** `s3_nonlocality.py:369` 的 KeyError
  （改为在 `run()` 的每条 hist 里记 `sim.ohmic`）。
- **门禁与产物**：每批至少跑 `make fast` + 对应专项门禁，P0/P1 加跑 `make s3`；
  JSON **只增键不删键**；`scripts/verify_sim.py` 的通过/失败/跳过汇总口径不变。
- **不碰的边界**：不重启 / 不写远端容器、不安装 GPU 版 JAX、不改 TORAX 物理、
  不把 A100 设成默认后端；GPU 只作为后续可复现实验脚本存在。
- **提交纪律**：按仓库安全规则不自动 `git commit`；提交需用户明确授权。

## 2. 分批路线选择

| 路线 | 说明 | 结论 |
|---|---|---|
| **A（采用）按风险递增 + 上游工作台对齐，四批** | P0 对齐与止血 → P1 上游口径落地 → P2 环境/RL v0.1 → P3 工程化 | **推荐**：先修 band 触发的崩溃，再逐步引入规格，每批独立可验收 |
| B 文档先行、代码后补 | 先一次性对齐所有文档，再统一改代码 | 否决：P0 的 KeyError 会持续挡 `make s3`，文档易先于实现变成「声称已做」 |
| C 按模块垂直切 | μ 动力学 / 两流体 / S3 / TORAX 各自端到端 | 否决：`gates.py`、`ALIGN-LEAN.md`、`artifacts/*.json` 被多线共享，门禁漂移与冲突风险最高 |

## 3. 四批逐文件改动清单

### P0 对齐与止血

| 文件 | 改动 | 上游依据 |
|---|---|---|
| `docs/ALIGN-LEAN.md:4,46,79-81` | 旧 commit 改「路径+HEAD+校验日期」；罚分 `5.0` → `100.0`（与 `env.py:33` 一致）；v0.1 待办标为已排期；保留 `[work]`/`[locking]` 两行 | `fusion-sim-workbench.md` |
| `docs/S2-VERDICTS.md:102,120-122`；`docs/S3-NONLOCALITY.md:3,44-55,73`；`gym_mu_frc/s3_nonlocality.py:4` | commit 引用同步；「0.6 宽」改「`band=0.30` 默认，`0.6` 可复现旧口径」；配置枚举 / 第二输入缺口状态改为「P1 落地」 | `theory-mu-topology.md:48-60,62-76,86` |
| `gym_mu_frc/physics.py:22` | **只加注释**：上游残余比 `2.19432e-4`，本仓沿用脚本口径 `2.194e-4`；不改数值，否则 `close_step=165` 与 G1 全崩 | `theory-mu-power.md:44-47` |
| `gym_mu_frc/s3_nonlocality.py:302-321,354-379,426-429` | `run()` 每条 hist 增记 `"欧姆耗散": sim.ohmic`；`reconnection_event` 返回增 `"band"`、`"eta0"`；阈值代价增 `[E_J_min, E_J_max]` 换算 | `theory-mu-topology.md:53-59` |
| `scripts/verify_s3_gates.py:37-38,58-85` | `ev.get("band", 0.6)` 兼容旧产物；新增 S3-9：`band` 已记录且 `0 < band < sep/2` | 同上 |
| `scripts/verify_plan_gates.py:31,129-140`；`scripts/verify_s2_gates.py:31,149-158`；`docs/REMOTE-RUN.md:47-49` | `HUSH_DIR` 环境变量；T7/S2-7 改读上游 `fusion-program-roadmap.md:31-33,62-78`，不再静默 SKIP | `fusion-program-roadmap.md` |

### P1 上游口径落地

| 文件 | 改动 | 上游依据 |
|---|---|---|
| `gym_mu_frc/env.py`、`gym_mu_frc/gates.py:351` | 新增第三路 `cost_model="ledger"`，按 `W = ε·gross(1−μ)` 记账；`run_all` 默认仍是 `("work","locking")`，新门另起 G8 组 | `theory-mu-power.md:75-93`；`fusion-sim-workbench.md:62-75` |
| `gym_mu_frc/s3_nonlocality.py:382-393` | `config_table()` 扩成 `(e, gross, net, μ)` 并校验 `gross ≡ net (mod 2)`；保留「模型替身」字样（S3-5 门依赖） | `theory-mu-power.md:49-59`；`theory-mu-topology.md:21-40` |
| `scripts/run_torax_smoke.py` | 报告显式区分 `engine="torax"` / `"reduced"`，不把降阶路径当 TORAX 本体 | `torax_adapter.py:45-52`；`fusion-sim-workbench.md:29-30,197-209` |

### P2 环境 / RL v0.1

| 文件 | 改动 | 上游依据 |
|---|---|---|
| `gym_mu_frc/env.py`、`agents.py`、`scripts/verify_sim.py` | 新增 `MuFrcEnvV01`（obs 8 / act 2，钳制 `μ_target`）、非平稳参数漂移；旧 `MuFrcEnv` 保留为 legacy；标准 RL 走可选 `--rl`，缺席仍按「已登记缺口」跳过 | `fusion-sim-workbench.md:119-143`；`design-minimal-mu-experiment.md:31-37`；`ALIGN-LEAN.md:79-81` |

### P3 工程化

| 文件 | 改动 | 上游依据 |
|---|---|---|
| `Makefile`、新增可选 `scripts/doctor.py`、新增可选 `scripts/bench_gpu_spectral.py`、`docs/REMOTE-RUN.md:78` | `make doctor` 如实报后端/TORAX/GPU；把 GPU 临时移植收编为**默认不跑**的基准脚本；补记「N=512 时 GPU 反而更慢，需更大网格 + 空闲卡」 | `fusion-sim-workbench.md:23-30,170-195,222-230` |

## 4. 数据流、错误处理与测试

- **数据流**：上游 `docs/wiki/*` + Lean + `artifacts/*` →（只读引用，不复制、不 import）→
  本仓 `docs/*` 口径登记 → 常量 / 配置 → 求解器与环境 → `artifacts/*.json` →
  门禁读取 → `scripts/verify_sim.py` 汇总。
- **可选依赖**：JAX / Gymnasium / TORAX / hushfusion 缺席时一律**如实报告**
  `available=false` / `engine="reduced"`；显式请求而缺席才抛错，不静默降级
  （`fusion-sim-workbench.md:23-30`）。
- **数值异常**：S3 发散继续返回 `N=-1`，绝不把 NaN 读成合并；`band` 旧产物用
  `get(..., 0.6)`；`ME_MI_DT` 不改值。
- **测试**：P0 = `make fast` + `make plan` + `make s3` + `bash scripts/teeth.sh`；
  P1 = `make fast` + `make s2` + `make s3`；P2 = `make test` + `--rl` 冒烟；
  P3 = `make doctor` + 基准脚本。每批额外做一次 `artifacts/*.json` 键集合 diff，
  确认**只增键不删键**。

## 5. 风险与回滚

| 风险 | 处理 |
|---|---|
| `band=0.30` 首次跑出 `N: 2→1`，进入 `verify_s3_gates.py:78-85` 的 `has_event` 分支 | P0 同批修 `欧姆耗散` 记账，不拆批 |
| 改 `ME_MI_DT` 会破坏 `close_step=165`、G1 网格 | 只加注释，不改值 |
| 改 `ALIGN-LEAN.md` / `S2-VERDICTS.md` 触发 G7b、S2-8 关键词门 | 保留 `[work]`/`[locking]`、`口径变更登记`、`配置枚举` 等关键词 |
| 上游 `theory-mu-topology.md:8-9` 仍是 `stage: draft / confidence: low` | 所有新增口径标「条件结论 / 模型替身」，不写成物理事实 |
| `run_all` 加 `ledger` 改变 `sim_report.json` 条目数 | 默认 `cost_models` 不变，新门另起组名 |
| 共享机 / GPU / 远端 | 不重启、不写远端、不安装 GPU JAX、不改默认后端 |

**回滚**：每批只改少量文件、产物 JSON 向后兼容、不自动提交；逐批 `git diff` 审阅即可回退。

## 6. 批次门槛

每批开始前置条件：上一批全部验收命令通过、`artifacts/*.json` 无删键、
用户未提交的 `band` 改动仍在工作区。任一门槛不满足即停止后续批次，先修复再继续。

## 7. 明确不做的事

- 不整体复制 / 不 import 上游 `fusion_sim/` 代码；
- 不把上游 draft 级 wiki 结论写成已证物理；
- 不改 TORAX 物理、不把 A100 设为默认后端、不重启远端容器；
- 不自动 `git commit`、不自动改 git 配置。
