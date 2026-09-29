"""gym-μ-FRC v0 —— μ 修正的 FRC 0-D 动力学 + Gymnasium 控制环境.

分层(与 Hibs-Physics 的形式化一一对应,见 docs/ALIGN-LEAN.md):
  physics.py  μ 的演化(TD1–TD21)· FRC 窗口判据(FC11b / FC5b)· 代价模型
  env.py      MuFrcEnv:Gymnasium 环境(动作 = 增益/停,观测 = 状态 + 窗口余量 + 锁定因子)
  agents.py   表格 Q-learning(纯 numpy,无额外依赖)+ 随机基线
  gates.py    门禁断言(与 Lean 数值一致 / μ→0 回归 / 单调性 / 环境契约 / 多 seed)

诚实口径(必须随结果一起引用):
  * μ 的演化方程与 η 的定义是**模型选择**,不是从物理推出的(Hibs-Physics
    docs/wiki/theory-mu-dynamics.md §四);
  * 现有开源代码(FROM/TORAX/Gkeyll…)对 μ 一无所知 → 本仓结果**没有文献可对照**,
    只有 μ→0 极限回归 + 内部自洽;
  * 代价模型的系数是选择项,故默认给两路消融('work' / 'locking')。
"""

from .physics import (  # noqa: F401
    ME_MI_DT,
    close_step,
    flatten_progress,
    locking_factor,
    m_eff_sq,
    mu_closed_form,
    mu_step,
    order_difference,
    window_margin,
    window_open,
)
from .env import MuFrcEnv  # noqa: F401
from .agents import RandomPolicy, TabularQ  # noqa: F401

__all__ = [
    "ME_MI_DT",
    "mu_step",
    "mu_closed_form",
    "window_margin",
    "window_open",
    "close_step",
    "locking_factor",
    "m_eff_sq",
    "flatten_progress",
    "order_difference",
    "MuFrcEnv",
    "RandomPolicy",
    "TabularQ",
]
__version__ = "0.1.0"
