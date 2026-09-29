# fusion-sim —— 聚变数字仿真工作台(v0)

[English](./README.md) · **中文**

两条线,一条仓:

| 线 | 内容 | 入口 |
|---|---|---|
| **A · 现成 AI 仿真栈** | TORAX(可微 1-D 输运, JAX)+ Gym-TORAX(Gymnasium 控制环境)在本机的可运行性 + ITER ramp-up 环境 | `scripts/run_torax_smoke.py`,报告 `docs/RUN-A-TORAX.md` |
| **B · 我们自己的环境** | `gym-μ-FRC v0`:μ 动力学(TD1–TD21)+ FRC 窗口(FC11b/FC5b)+ RMF 代价 + Gymnasium 环境 + RL + 门禁 | `gym_mu_frc/`,门禁 `make test` |

## 为什么要两条线

A 回答的是「开源生态里带 AI 加持的那套东西能不能在本机跑起来」(链路问题);
B 回答的是「我们自己的物理(μ 修正)能不能做成一个可训练、可验证、可对照的控制问题」(差异化问题)。

A 里的代码对 μ 一无所知;B 里的 μ 修正**没有文献可对照**,因此 B 的验证只能靠三件事:
① 与 Hibs-Physics 的 Lean/数值逐条对齐(两套独立实现互证);
② μ→0 极限回归(关掉 μ 修正必须退化回经典 FRC 标度);
③ 多 seed + 随机基线 + 穷举参考解对照(单 seed 结论不作数)。

## 安装

```bash
python3 -m venv .venv && source .venv/bin/activate

# B(轻:只要 numpy + gymnasium)
pip install -r requirements-sim.txt

# A(重:jax / flax / torax / gymtorax)
pip install -r requirements-torax.txt
```

依赖版本说明:A 线在本机(Intel macOS)上**必须钉版本**才对得上 ——
`torax 1.4.3` 要求 `jaxlib>=0.10`,而 x86_64 macOS 上可装的最高 `jaxlib` 是 **0.4.38**,
所以钉 `jax/jaxlib==0.4.38 + torax==1.0.3 + gymtorax==1.0.0`(torax 1.0.3 要求 `jaxlib>=0.4.32` ✓)。
Apple Silicon / Linux 上可用最新版(见 `docs/RUN-A-TORAX.md` 的版本表)。

## 跑

```bash
make test          # B 线全门禁(5 seeds, 400 回合训练 + 对照 + 穷举参考)
make fast          # 冒烟(1 seed, 60 回合)
make smoke-torax   # A 线:TORAX 冒烟 + Gym-TORAX 环境跑一回合
bash scripts/teeth.sh   # 门齿:注入两个真缺陷(μ 更新忽略 η / 关闭步 +1)→ 必须变红 → 复原
```

产物:门禁汇总行 + `artifacts/sim_report.json`(逐条门的名字/红绿/载荷)+ A 线的 `artifacts/torax_smoke.json`。

**门齿实测**(`scripts/teeth.sh`):把 `mu_step` 改成忽略 η(η≡1)→ **7 条红**;把 `close_step` 偏一步 → **1 条红**(期望 165 / 实际 166);复原后全绿。
注入型缺陷若让契约函数直接抛异常,组级兜底会把它记成**一条红**而不是崩掉整轮(崩掉的输出最像"什么都没跑")。

**已知缺口**:v0 的 RL 学习器没有解决控制问题(实测比例与失败方式见 `docs/ALIGN-LEAN.md` §4.1)。
门禁把这两条按「已登记缺口」**计数并打印**(`跳过 K 项(已登记缺口)`),不隐藏、也不假装通过;
登记门 G7b 会核对文档里确实写了这条缺口 —— 想靠"写一句还没达到"糊过去是不行的。

## 诚实边界(引用结果时必须一起引用)

- μ 的演化方程 `μ ↦ μ + η(1−μ)` 与 `η = 抹平进展` 是**模型选择**,不是从物理推出的;
- 代价模型(抹平功怎么算)同样是选择项,故门禁里跑两路消融(`work` / `locking`);
- FRC 侧只用了 Hibs-Physics 已证的两条(`FC11b` 窗口条件、`FC5b` 标度因子),**没有**引入新的物理机制;
- 本仓不含任何实验数据,也**不做**实验拟合;所有结论都是「在这个模型下」的条件结论;
- 不是一个电站级仿真,也不替代中子学/氚/材料那三层(那是另一个尺度的问题)。
