"""? 动力学 + FRC 0-D 判据 —— 与 Hibs-Physics 的 Lean 定理逐条对应.

对应表(定理名 → 这里的函数/断言):
  TD1a/TD2/TD3/TD3b/TD5/TD7/TD8/TD9/TD10  mu_step / mu_closed_form / locking_factor
  TD11/TD12/TD13/TD14                     flatten_progress
  TD15/TD16/TD17                          order_difference + apply_order()
  TD18                                    cost_model('work') 的零代价端点
  TD19/TD20/TD20b/TD21                    window_margin / locking_factor / close_step
  FC11b(窗口存在 ⟺ m_e < m_eff;关闭 ⟺ μ ≥ 1 − m_e/m_i)   window_open / window_closed
  FC5b(μ 标度因子 = 1/√(1−μ),同时作用于 S* 与 τ_E)        locking_factor

数值常量来自 Hibs-Physics(scripts/verify_mu_dynamics.py / artifacts/mudynamics/report.json);
本模块不发明新数字,只做同一批断言的独立实现(两套实现互证,见 docs/ALIGN-LEAN.md §4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# D-T 的电子/离子质量比:m_e/m_i = 2.194e-4(FC11b 里的硬地板)
ME_MI_DT: float = 2.194e-4

# 允许的增益上限:η ≤ 1 才是"无超调"区(TD3b)
ETA_STABLE_MAX: float = 1.0
# 收敛上界:η ≥ 2 时 |1−η| ≥ 1,轨道不收敛(N10)
ETA_CONVERGENT_MAX: float = 2.0


def _check_mu(mu: float) -> None:
    """状态域:μ ∈ [0,1].μ=1 是不动点(TD4),允许作为**值**存在。"""
    if not (0.0 <= mu <= 1.0):
        raise ValueError(f"μ 必须在 [0,1] 内: {mu!r}")


def _check_mu_open(mu: float) -> None:
    """需要"还没到 1"的量(锁定因子、m_eff²)用这个:μ ∈ [0,1).

    锁定因子 1/√(1−μ) 在 μ=1 处发散;而 μ=1 在实数域有限增益下不可达(TD3/TD8),
    所以这是"不该被调用"而不是"该给个数"。
    """
    if not (0.0 <= mu < 1.0):
        raise ValueError(f"该量只在 μ ∈ [0,1) 上良定义(μ=1 是极限,不是状态): {mu!r}")


def _check_eta(eta: float) -> None:
    if not (0.0 <= eta):
        raise ValueError(f"η 必须 ≥ 0: {eta!r}")


def mu_step(mu: float, eta: float) -> float:
    """饱和增长一步:μ' = μ + η(1−μ).  (TD1a 有界 / TD2 严格推进 / TD5 η=1 一步到 1)"""
    _check_mu(mu)
    _check_eta(eta)
    return mu + eta * (1.0 - mu)


def mu_closed_form(mu0: float, eta: float, n: int) -> float:
    """闭式解 μ_n = 1 − (1−η)^n (1−μ₀).  (TD7)"""
    if n < 0:
        raise ValueError(f"步数必须 ≥ 0: {n!r}")
    _check_mu(mu0)
    _check_eta(eta)
    return 1.0 - (1.0 - eta) ** n * (1.0 - mu0)


def window_margin(mu: float) -> float:
    """窗口余量 m_i(1−μ).  (TD19:沿轨道严格递减;μ=1 时余量为 0 = 必然关闭)"""
    _check_mu(mu)
    return 1.0 - mu


def window_open(mu: float, me_mi: float = ME_MI_DT) -> bool:
    """窗口是否存在:1 − μ > m_e/m_i.  (FC11b 的等价形式)"""
    _check_mu(mu)
    return (1.0 - mu) > me_mi


def window_closed(mu: float, me_mi: float = ME_MI_DT) -> bool:
    return not window_open(mu, me_mi)


def locking_factor(mu: float) -> float:
    """μ 标度因子 1/√(1−μ),同时作用于 S* 与 τ_E.  (FC5b;TD20 分母恒正 = 良定义)"""
    _check_mu_open(mu)
    return 1.0 / math.sqrt(1.0 - mu)


def m_eff_sq(mu: float, s: float = 1.0) -> float:
    """m_eff² = s²(1−μ)².TD10:m_eff² > 0 ⟺ 质量永不归零(μ<1 时恒成立)."""
    _check_mu_open(mu)
    return (s * (1.0 - mu)) ** 2


def close_step(mu0: float = 0.0, eta: float = 0.05, me_mi: float = ME_MI_DT) -> int:
    """窗口关闭步 n*:最小 n 使 (1−η)^n (1−μ₀) ≤ m_e/m_i.  (TD21)

    η=0.05、μ₀=0、D-T ⟹ 165(Hibs-Physics 数值口径).
    """
    _check_mu(mu0)
    _check_eta(eta)
    if not (0.0 < eta < 1.0):
        raise ValueError(f"close_step 只在 0<η<1 上有意义(η={eta!r})")
    if not (0.0 < me_mi < 1.0):
        raise ValueError(f"m_e/m_i 必须在 (0,1) 内:{me_mi!r}")
    n = math.log(me_mi / (1.0 - mu0)) / math.log(1.0 - eta)
    return int(math.ceil(n - 1e-12))


def flatten_progress(q_now: float, q_ref: float) -> float:
    """η = 抹平进展 = 1 − Q_A(v)/Q_A(v₀).  (TD11/TD12/TD13)

    TD11a 未抹平 ⟹ η=0;TD11b/TD12 已平坦 ⟹ η=1(复用 GCA2c)。
    """
    if q_ref <= 0:
        raise ValueError(f"参考二次型 Q_A(v₀) 必须 > 0:{q_ref!r}")
    if q_now < 0:
        raise ValueError(f"二次型非负:Q_A(v) = {q_now!r} < 0")
    if q_now > q_ref:
        raise ValueError("抹平不会让起伏变大:Q_A(v) ≤ Q_A(v₀)")
    return 1.0 - q_now / q_ref


def order_difference(mu: float, eta_before: float) -> float:
    """顺序差 = (1−μ)(1−η_before).  (TD16;只来自增益的差)"""
    _check_mu(mu)
    _check_eta(eta_before)
    return (1.0 - mu) * (1.0 - eta_before)


def apply_flatten_then_update(mu: float, q_now: float, q_ref: float) -> float:
    """先抹平、再更新 μ(抹平把增益拉到 1 ⟹ μ 一步到 1).  (TD14a / TD17 见证 A)"""
    eta = flatten_progress(q_now, q_ref)
    return mu_step(mu, eta)


def apply_update_then_flatten(mu: float, eta: float) -> float:
    """先更新 μ、再抹平(抹平只改场、不改已推进的 μ).  (TD17 见证 B)"""
    return mu_step(mu, eta)


@dataclass(frozen=True)
class Ledger:
    """一步的账:状态 + 余量 + 锁定因子 + 是否关闭.

    Hibs-Physics 的 N11/TD19–TD21 口径:余量单调收窄、锁定因子良定义且单调增。
    """

    n: int
    mu: float
    margin: float
    locking: float
    closed: bool
    m_eff_sq: float

    @staticmethod
    def of(n: int, mu: float, me_mi: float = ME_MI_DT, s: float = 1.0) -> "Ledger":
        """μ=1 时锁定因子发散 → 这里直接拒绝(而不是给一个 inf 让下游算错)。"""
        return Ledger(
            n=n,
            mu=mu,
            margin=window_margin(mu),
            locking=locking_factor(mu),
            closed=window_closed(mu, me_mi),
            m_eff_sq=m_eff_sq(mu, s),
        )


def trajectory(mu0: float = 0.0, eta: float = 0.05, steps: int = 300,
               me_mi: float = ME_MI_DT) -> list[Ledger]:
    """μ 轨道上的账本序列(逐步递推,不用闭式解 —— 两条路要能互证)."""
    out: list[Ledger] = []
    mu = mu0
    for n in range(steps + 1):
        out.append(Ledger.of(n, mu, me_mi))
        mu = mu_step(mu, eta)
    return out
