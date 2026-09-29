"""MuFrcEnv —— 把「推进 μ 还是停在窗口关闭前」做成一个 Gymnasium 控制问题.

这一层是**控制问题**,不是新物理:物理全部来自 physics.py(TD/FC 系列)。
环境里唯一的新东西是两个**选择项**(在门禁里各跑一路消融,见 docs/ALIGN-LEAN.md §3):

  ① 收益:每步的 τ_E 红利 = 锁定因子 LF(μ) = 1/√(1−μ)     (FC5b 的标度因子)
     ——注意是**每步照收**(运行在 LF(μ) 上),不是"增量的和":增量口径会退化成
     "只要最后 μ 一样,回报就一样",于是"什么时候停"变得无所谓,控制问题就假了。
  ② 代价:抹平功 = c · η · (1 或 LF(μ))                     (TD18:满增益 ⟺ 零代价 ⟺ 已平坦,
     所以"代价随 η"本身是选择项;两路消融 = 'work' / 'locking')

动作(Discrete 4):0 = 停;1/2/3 = 用 η = 0.01 / 0.05 / 0.20 推进一步。
  · 窗口关闭(1−μ ≤ m_e/m_i,TB21/FC11b)⟹ 立即终止 + 罚分(RMF 失效,不是"稳态运行")
  · 步数上限 N_max ⟹ 截断(不算"成功")
目标:最大化 Σ(红利 − 代价),窗口关闭 = 失败。于是「推到哪一步停」有非平凡最优解,
参考解由**穷举**(η, 停步)得到 —— 拿穷举当基准,不去手推闭式解。
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .physics import ME_MI_DT, locking_factor, mu_step, window_closed, window_margin

# 奖励归一化:除以"窗口边缘的锁定因子",让不同 m_e/m_i 下的量级可比
LF_REF = 1.0 / math.sqrt(ME_MI_DT)

PENALTY_CLOSED = 100.0    # 窗口关闭的罚分(单位:归一化后的红利)

# 为什么罚分要这么"大":这个环境的问题是「在哪一步停」——
# 把窗口打爆不是"少赚一点",而是 RMF 失效、装置失去约束(FC11b 的窗口没了)。
# 罚分必须**大于任何可达红利上限**(实测最优红利 ≈38.6,见 analytic_best),
# 否则一个近视策略会"赌一把":用最大增益冲到窗口边缘,每步红利最高 ~1.0,
# 几十步就能收 4~5,而罚分只有 5 —— 于是它会主动打爆窗口(实测 G6 全灭)。
# 这不是把奖励调好看,而是让「打爆窗口」在模型里真的等于失败。
ETA_CHOICES = (0.01, 0.05, 0.20)
ACTION_NAMES = ("stop", "eta=0.01", "eta=0.05", "eta=0.20")


class MuFrcEnv(gym.Env):
    """μ-FRC 0-D 控制环境(单回合 = 一次"推到哪一步停"的决策)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        mu0: float = 0.0,
        me_mi: float = ME_MI_DT,
        cost_coef: float = 0.02,
        cost_model: str = "work",
        n_max: int = 400,
        reward_scale: bool = True,
    ) -> None:
        super().__init__()
        if cost_model not in ("work", "locking"):
            raise ValueError(f"cost_model ∈ {{'work','locking'}}:{cost_model!r}")
        self.mu0 = float(mu0)
        self.me_mi = float(me_mi)
        self.cost_coef = float(cost_coef)
        self.cost_model = cost_model
        self.n_max = int(n_max)
        self.reward_scale = bool(reward_scale)

        # 观测:[μ, 余量比(margin/me_mi,截到 1e6 以外不截), 步进度, 已累计代价]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1e3, 1.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        self._mu = self.mu0
        self._n = 0
        self._cost = 0.0
        self._closed = False

    # ── 观测 ───────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        margin_ratio = min(window_margin(self._mu) / self.me_mi, 1e3)
        return np.array(
            [self._mu, margin_ratio, self._n / self.n_max, min(self._cost, 10.0)],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, Any]:
        lf = locking_factor(self._mu)
        return {
            "mu": self._mu,
            "n": self._n,
            "margin": window_margin(self._mu),
            "locking": lf,
            "closed": self._closed,
            "cum_cost": self._cost,
        }

    # ── Gymnasium API ─────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._mu = self.mu0
        self._n = 0
        self._cost = 0.0
        self._closed = False
        return self._obs(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"动作越界:{action!r}")
        if self._closed:
            raise RuntimeError("回合已结束(窗口关闭),须先 reset()")

        terminated = False
        truncated = False
        reward = 0.0

        if action == 0:                     # 停:不再收红利,干净结束
            terminated = True
            return self._obs(), 0.0, terminated, truncated, self._info()

        eta = ETA_CHOICES[action - 1]
        mu_new = mu_step(self._mu, eta)
        lf_after = locking_factor(mu_new)

        # ① 红利:这一步运行在 τ_E 标度因子 LF(μ) 上(FC5b) —— 每步照收
        gain = lf_after
        # ② 代价:抹平功(选择项;'locking' 让功率随锁定因子一起涨)
        cost = self.cost_coef * eta * (lf_after if self.cost_model == "locking" else 1.0)
        reward = gain - cost

        self._mu, self._n, self._cost = mu_new, self._n + 1, self._cost + cost

        # 归一化**先做**,罚分在归一化单位上扣 —— 这样"一步红利"与"罚分"才可比,
        # 也才能和 analytic_best 的口径一致(单位不一致的罚分 = 偷偷改了物理)。
        if self.reward_scale:
            reward /= LF_REF
        if window_closed(self._mu, self.me_mi):   # ③ 窗口关闭 = 失败终止
            self._closed = True
            terminated = True
            reward -= PENALTY_CLOSED
        elif self._n >= self.n_max:
            truncated = True
        return self._obs(), float(reward), terminated, truncated, self._info()


# ── 参考解:穷举(η, 停步)────────────────────────────────────────────────
def analytic_best(
    mu0: float = 0.0,
    me_mi: float = ME_MI_DT,
    cost_coef: float = 0.02,
    cost_model: str = "work",
    n_max: int = 400,
    reward_scale: bool = True,
) -> dict[str, Any]:
    """穷举「固定 η 推进 n 步再停」这一类策略的最优解(策略基准,不是物理结论).

    返回最优的 (eta, n, return)。窗口关闭的策略按环境的罚分口径计入(负回报)。
    """
    best: dict[str, Any] = {"eta": None, "n": -1, "return": -float("inf")}
    for eta in ETA_CHOICES:
        mu = mu0
        total = 0.0
        for n in range(1, n_max + 1):
            mu = mu_step(mu, eta)
            lf_after = locking_factor(mu)
            cost = cost_coef * eta * (lf_after if cost_model == "locking" else 1.0)
            r = lf_after - cost
            if reward_scale:
                r /= LF_REF
            total += r
            if window_closed(mu, me_mi):        # 打到窗口 → 罚分并结束(这一步的收益照收,因为已经发生)
                total -= PENALTY_CLOSED
                break
            if total > best["return"]:
                best = {"eta": eta, "n": n, "return": total}
    return best
