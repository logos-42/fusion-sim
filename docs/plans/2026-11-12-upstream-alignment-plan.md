# 上游对齐与分批优化 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 以 `/Users/apple/Downloads/lean/ProjectionPhysics` @ `3a7775a27b52581aa6a04d50f58c2c4e5d61277e` 为规格来源，分四批对齐 `fusion-sim` 的口径、缺陷、环境与工程化，同时保持 `make test` / `make s3` 全绿、产物 JSON 只增键、保留未提交的 `band=0.30` 改动。

**Architecture:** 上游只读引用、不复制、不 import；`gym_mu_frc/` 保持独立实现；所有新口径以「新增不替换」进入。每批先建立回归基线或会红的门禁断言，再实现，再跑绿，最后审阅 diff。

**Tech Stack:** Python 3.12 / NumPy / Gymnasium / 自写 `check()` 门禁 / Makefile / JSON 产物；JAX、TORAX、PyTorch 仅可选或基准用。

---

## 0. 执行前提与偏差登记

- 设计依据：`docs/plans/2026-11-12-upstream-alignment-design.md`（已批准）。
- **未使用独立 worktree**：当前工作区有未提交改动 `gym_mu_frc/s3_nonlocality.py`（`band`），且创建 worktree 未获授权；计划直接在当前工作区执行。
- **无 pytest**：本仓门禁是自写 `check()` + `make` 目标，按 YAGNI 不引入测试框架；TDD 等价循环 = 「加会红的门禁 → 跑红 → 最小实现 → 跑绿」。
- **不自动 `git commit`**：每个 Task 末步是 `git --no-pager diff` 审阅；提交需用户明确授权。
- 上游 `theory-mu-topology.md` 仍是 `stage: draft / confidence: low`，所有新增口径必须带「条件结论 / 模型替身」字样。
- 全局硬约束：`band` 保持第 5 参数、默认 `0.30`；`docs/ALIGN-LEAN.md` 的 `[work]` / `[locking]` 两行原样保留（`gym_mu_frc/gates.py:340-344` G7b 依赖）；`ME_MI_DT` 数值不改。
- 每批开始前置条件：上一批验收全通过、`artifacts/*.json` 无删键、`band` 改动仍在工作区。

---

# P0 —— 对齐与止血

### Task 1: 同步 `docs/ALIGN-LEAN.md` 口径

**Files:**
- Modify: `docs/ALIGN-LEAN.md:4,46,79-81`
- Test: `make fast`（G1–G8，含 G7b）

**Step 1: 记录基线**
Run: `make fast`
Expected: 末尾汇总通过/失败/跳过，G7b 为绿；记录当前数字。

**Step 2: 改第 4 行**
Replace:
`左侧是 Hibs-Physics(\`~/Downloads/lean/ProjectionPhysics\`,提交 \`d28cd41\`)的定理名,`
With:
`左侧是上游 \`/Users/apple/Downloads/lean/ProjectionPhysics\`（HEAD \`3a7775a27b52581aa6a04d50f58c2c4e5d61277e\`, 校验日期 2026-11-12）的定理名,`

**Step 3: 改第 46 行罚分**
Replace: `窗口关闭 = 罚 5.0 并终止;步数上限 400(n_max)。`
With: `窗口关闭 = 罚 100.0 并终止(与 \`gym_mu_frc/env.py:33\` 的 \`PENALTY_CLOSED\` 一致);步数上限 400(n_max)。`

**Step 4: 改 §4.1 待办**
在 `**v0.1 待办(按优先级)**` 段首追加一行：
`> 状态：已排期（P2 落地），上游依据见 \`docs/plans/2026-11-12-upstream-alignment-design.md\`。`

**Step 5: 验证并审阅**
Run: `make fast`
Expected: G7b 仍绿（`[work]`、`[locking]` 两行未动），汇总数字与 Step 1 一致。
Run: `git --no-pager diff -- docs/ALIGN-LEAN.md`
不提交。

---

### Task 2: 同步 S2/S3/README 的 commit 与 band 口径

**Files:**
- Modify: `docs/S2-VERDICTS.md:102,120-122`
- Modify: `docs/S3-NONLOCALITY.md:3,44-55,73`
- Modify: `gym_mu_frc/s3_nonlocality.py:4`
- Modify: `README.zh-CN.md:57-63`
- Test: `make s2` + `make s3`

**Step 1: 记录基线**
Run: `make s2`
Expected: S2-8（口径变更登记）为绿。

**Step 2: 替换三处旧 commit**
把 `77c794f` 统一改为 `3a7775a27b52581aa6a04d50f58c2c4e5d61277e（校验 2026-11-12）`；`docs/S2-VERDICTS.md:102` 的「提交 77c794f」改为「HEAD」。保留 `theory-mu-topology.md` 文件名（S2-8 门依赖该关键词）。

**Step 3: 改 S3 文档口径**
Replace: `设 sep = 2.0、η 只在两岛之间 0.6 宽的窄带`
With: `设 sep = 2.0、η 只在两岛之间 band=0.30 的窄带（默认；band=0.6 可复现旧口径）`
`docs/S3-NONLOCALITY.md:73` 的「尚缺一句可辩护的归一化」改为「P1 落地 [E_J_min, E_J_max] 换算（上游 1.23e6 / 4.94e6 / 1.23e9 J）」。

**Step 4: 改 S2 未做登记**
`docs/S2-VERDICTS.md:120-122` 的「尚未做：把上面的『配置枚举』口径落成一张可执行的配置表」改为「已在 P1 排期落地：见设计文档 §3」。

**Step 5: 验证**
Run: `make s2`
Expected: S2-8 仍绿（关键词 `口径变更登记` / `配置枚举` / `theory-mu-topology.md` / `局部变形` / `阈值能量` 全在）。
Run: `git --no-pager diff -- docs/S2-VERDICTS.md docs/S3-NONLOCALITY.md gym_mu_frc/s3_nonlocality.py README.zh-CN.md`
不提交。

---

### Task 3: 修 S3 欧姆耗散 KeyError 并留痕 band / eta0

**Files:**
- Modify: `gym_mu_frc/s3_nonlocality.py:302-321,354-379,426-429`
- Modify: `scripts/verify_s3_gates.py:35-38,58-85,104-108`
- Test: `make s3`（约 5–8 分钟，本批唯一长步骤）

**Step 1: 先加会红的门禁**
在 `scripts/verify_s3_gates.py` 的 `main()` 中、`ev = d["S3-2_重联事件"]` 之后插入：

```python
    band = ev.get("band")
    check("S3-9: 重联带宽已记录（0 < band < sep/2）",
          isinstance(band, (int, float)) and 0 < band < 1.0,
          {"band": band})
```

Run: `make s3`
Expected: 预期红 —— 要么 `S3-9 = FAIL`（`band: None`），要么实验先在
`gym_mu_frc/s3_nonlocality.py:369` 崩在 `r["欧姆耗散"]` KeyError
（`band=0.30` 首次给出干净合并时必然触发）。两者都算「先红」，不要用旧产物兜底掩盖。

**Step 2: `run()` 每条 hist 增记欧姆耗散**
`gym_mu_frc/s3_nonlocality.py:307`：
```python
    hist = [{"t": 0.0, **snapshot(g, sim.psi, thr, sep), "欧姆耗散": sim.ohmic}]
```
`:319`：
```python
            hist.append({"t": round(t, 4), **snapshot(g, sim.psi, thr, sep),
                         "欧姆耗散": sim.ohmic})
```
`:315-316` 的发散分支补 `"欧姆耗散": float("nan")`。

**Step 3: `reconnection_event` 返回增键（保留 `band` 参数位置与默认值）**
`gym_mu_frc/s3_nonlocality.py:372-379` 的 return 字典末尾追加：
```python
            "步数": sim.nsteps, "最小有效 dt": sim.dt_min,
            "band": band, "eta0": eta0}
```
`:369` 的 `r["欧姆耗散"]` 保持不变（Step 2 已保证键存在）。

**Step 4: 阈值代价换算**
`gym_mu_frc/s3_nonlocality.py:426-429` 的 `dim` 字典追加：
```python
           "上游阈值能量 [J]": {"电阻层 δ=L/√S": 1.23e6,
                              "离子惯性长 d_i": 4.94e6,
                              "整管上界": 1.23e9},
           "口径": "量级口径，非推导；层厚假设跨度 500×（上游 theory-mu-topology.md §4）"}
```

**Step 5: S3 门禁兼容旧产物（门名保持不变）**
把 Step 1 的 S3-9 断言替换为兼容读取，旧 `report.json` 用 `0.6` 兜底：

```python
    has_band = isinstance(ev.get("band"), (int, float))
    band = ev.get("band", 0.6)
    check("S3-9: 重联带宽已记录（0 < band < sep/2）",
          0 < band < 1.0,
          {"band": band, "来源": "new" if has_band else "legacy"})
```

`scripts/verify_s3_gates.py:37` 之后的打印同样用 `ev.get("band", 0.6)`。
新产物走 `来源=new`，旧产物走 `legacy`，两者都绿。

**Step 6: 跑绿并审阅**
Run: `make s3`
Expected: S3-0..S3-9 全绿；若 `band=0.30` 跑出 `N: 2→1`，S3-2 打印「干净合并窗口」、S3-4 走 `has_event` 分支且不再 KeyError。
Run: `git --no-pager diff -- gym_mu_frc/s3_nonlocality.py scripts/verify_s3_gates.py`
Expected: diff 中 `band: float = 0.30` 与 `:364` 的 eta 公式**未被改动**。
不提交。

---

### Task 4: 跨仓路径改为可配置并接上游原文

**Files:**
- Modify: `scripts/verify_plan_gates.py:31,129-140`
- Modify: `scripts/verify_s2_gates.py:31,149-158`
- Modify: `docs/REMOTE-RUN.md:47-49`
- Test: `make plan` + `make s2`

**Step 1: 两个脚本第 31 行统一替换**
```python
import os
HUSH = pathlib.Path(os.environ.get("HUSH_DIR", "/Users/apple/Downloads/hushfusion"))
```

**Step 2: T7 / S2-7 增加上游兜底**
在 `if prog.exists():` 分支之后增加 `elif` 分支：读
`/Users/apple/Downloads/lean/ProjectionPhysics/docs/wiki/fusion-program-roadmap.md`，
检查 `["判决漏斗", "G1", "1.29%", "1e−4", "R_ci"]` 是否命中；命中则以
`check(name, True, {"来源": "上游 fusion-program-roadmap.md"})` 通过，不再打印 `SKIP`。
保持门名字符串完全不变（`artifacts/*.json` 依赖）。

**Step 3: 更新远端文档**
`docs/REMOTE-RUN.md:47-49` 改为：`HUSH_DIR` 可指向随包传上去的上游 wiki；两门不再静默跳过。

**Step 4: 验证**
Run: `make plan` → `T7` 绿；`make s2` → `S2-7`/`S2-8` 绿。
Run: `git --no-pager diff -- scripts/verify_plan_gates.py scripts/verify_s2_gates.py docs/REMOTE-RUN.md`
不提交。

---

### Task 5: `ME_MI_DT` 注释与 P0 批次验收

**Files:**
- Modify: `gym_mu_frc/physics.py:22`

**Step 1: 只加注释**
```python
# D-T 的电子/离子质量比:m_e/m_i = 2.194e-4(FC11b 里的硬地板)
# 上游 theory-mu-power.md §2 的残余比为 2.19432e-4;本仓沿用 Hibs-Physics
# 脚本口径 2.194e-4,不改数值(否则 close_step 的 165 与 G1 网格会整体漂移)。
ME_MI_DT: float = 2.194e-4
```

**Step 2: P0 全量验收**
Run: `make fast`、`make plan`、`make s2`、`make s3`、`bash scripts/teeth.sh`
Expected: 全绿；`teeth.sh` 注入缺陷仍变红并复原。
Run: `python3 -c "import json,pathlib; p=pathlib.Path('artifacts'); print(sorted(x.name for x in p.glob('*.json')))"`
Expected: 与 P0 前一致，且各 JSON 无删键（人工 diff 键集合）。
Run: `git status --short`
Expected: `M gym_mu_frc/s3_nonlocality.py`（用户改动 + 本次改动合并）、`?? docs/plans/`、以及本批修改文件；**无自动 commit**。

---

# P1 —— 上游口径落地

### Task 6: 新增 `cost_model="ledger"`（第三路，不替换旧两路）

**Files:** Modify `gym_mu_frc/env.py:50-146`、`gym_mu_frc/gates.py:351-380`

**Step 1: 先加会红的 G8 门**
在 `gates.py` 新增 `def ledger_gates() -> list[Check]`，断言
`W = ε·gross(1−μ)` 单调递减、`μ→1 ⇒ W→0`、`gross ≡ net (mod 2)`；
此时 `env.py` 无 `ledger`，门红。

**Step 2: 最小实现**
`env.py` 的 `cost_model` 白名单扩为 `("work", "locking", "ledger")`；
`ledger` 分支：`cost = ε * gross * (1 - mu_new)` 的逐步差分，ε 默认 `6.17e8`（上游
`theory-mu-power.md:87-93`），`gross` 默认 4557.22（上游 §2）。
`gates.py:351` 的 `run_all` 默认参数**保持** `("work", "locking")`，新增可选
`include_ledger: bool = False` 才跑 G8。

**Step 3: 验证**
Run: `make fast` → 汇总条目数与 P0 完全一致；加跑
`.venv/bin/python -c "from gym_mu_frc.gates import run_all; c,_=run_all(include_ledger=True); print(sum(x.ok for x in c))"` → G8 绿。
Run: `git --no-pager diff`；不提交。

---

### Task 7: 配置表扩成 `(e, gross, net, μ)` 并校验奇偶律

**Files:** Modify `gym_mu_frc/s3_nonlocality.py:382-393`、`scripts/verify_s3_gates.py:87-91`、`docs/S3-NONLOCALITY.md:57-66`、`docs/S2-VERDICTS.md:120-122`

**Step 1: 加会红的门**
新增 `S3-10`：每行必须有 `e/gross/net/μ` 且 `gross % 2 == net % 2`；当前 `config_table()` 无 `e` 键以外的完整字段 → 红。

**Step 2: 实现**
`config_table()` 每行补 `e = gross`，`μ = 1 - net/gross`，新增
`{"奇偶律违例": sum(1 for r in rows if (r["gross"] - r["net"]) % 2 != 0)}`；
**保留** `"口径"` 里的「模型替身」四字（S3-5 门依赖）。

**Step 3: 验证**
Run: `make s3` → S3-5、S3-10 全绿；`git --no-pager diff`；不提交。

---

### Task 8: A 线如实标记引擎

**Files:** Modify `scripts/run_torax_smoke.py:94-143`

**Step 1:** `report` 增 `"engine": "reduced"` 与 `"torax_available": bool`；`versions()` 保持。
**Step 2:** 打印行加 `engine=...`，不把降阶路径写成 TORAX 本体。
**Step 3:** Run `make smoke-torax`（需 A 线环境）或仅 `python -m py_compile scripts/run_torax_smoke.py`；
`git --no-pager diff`；不提交。

---

# P2 —— 环境 / RL v0.1

### Task 9: `MuFrcEnvV01`（obs 8 / act 2）

**Files:** Modify `gym_mu_frc/env.py`（新增类，保留 `MuFrcEnv`）

**Step 1:** 新增会红的契约门：obs 长度 8、act 长度 2、`μ_target=0.999` 钳制不越界。
**Step 2:** 实现 `MuFrcEnvV01`，观测按上游 `fusion-sim-workbench.md:122`，控制律按
`:125-129`（`η = min(η_req, (μ_target−μ)/(1−μ))`），终止按 `:131`。
**Step 3:** `make fast` 仍绿（旧环境未动）；新门绿；不提交。

### Task 10: 非平稳参数漂移

**Files:** Modify `gym_mu_frc/env.py`、`gym_mu_frc/gates.py`

**Step 1:** 新增 `drift: bool = False` 与 `default_rng(seed)`；开启后 `me_mi` / `eta` 实现值 / 代价带噪声。
**Step 2:** 新增 G9：同 seed 轨迹逐位一致、不同 seed 轨迹不同。
**Step 3:** `make fast` 绿（默认 `drift=False`）；不提交。

### Task 11: 可选标准 RL 基线

**Files:** Modify `scripts/verify_sim.py`、`gym_mu_frc/agents.py`

**Step 1:** 加 `--rl` 开关；缺席或没装 `stable-baselines3` 时打印「已登记缺口」并计入跳过，不伪装通过。
**Step 2:** `make test` 默认汇总不变；`--rl` 冒烟可跑；不提交。

---

# P3 —— 工程化

### Task 12: `make doctor` 与后端如实报告

**Files:** Create `scripts/doctor.py`（可选）、Modify `Makefile:5-14`

**Step 1:** 输出 NumPy / Gymnasium / JAX / TORAX / GPU 可用性与 `engine` 标记，缺依赖不崩。
**Step 2:** `make doctor` 退出 0；`make fast` 仍绿；不提交。

### Task 13: 收编 GPU 基准并更新远端文档

**Files:** Create `scripts/bench_gpu_spectral.py`（可选、默认不跑）、Modify `docs/REMOTE-RUN.md:78`

**Step 1:** 脚本用 PyTorch FFT 复现 1-D 谱/RK4，显存上限 2%，只在显式调用时运行。
**Step 2:** 文档补记「N=512 实测 GPU 约 7.0 s 慢于 CPU 约 2.4 s；需更大网格 + 空闲卡」；
不改默认 CPU 后端；不提交。

---

## 批次门槛与总验收

| 批次 | 命令 | 通过条件 |
|---|---|---|
| P0 | `make fast` `make plan` `make s2` `make s3` `bash scripts/teeth.sh` | 全绿，JSON 无删键，`band` 改动保留 |
| P1 | `make fast` `make s2` `make s3` + G8 冒烟 | 全绿，`sim_report.json` 条目数不变 |
| P2 | `make test` + `--rl` 冒烟 | 汇总口径不变，新门绿 |
| P3 | `make doctor` + 基准脚本显式调用 | 默认 CPU 后端不变 |

任一门槛不满足即停止后续批次，先修复再继续。

## 执行交接

计划保存后提供两种执行方式（见对话中的选项）：
1. **本会话逐任务执行**（每个 Task 一个新子代理 + 任务间审阅）；
2. **另开会话批量执行**（在独立 worktree 中用 `@executing-plans`，带检查点）。
