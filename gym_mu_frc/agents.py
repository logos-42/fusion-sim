"""策略与训练:随机基线 + 表格 Q-learning(纯 numpy,不加依赖).

为什么要随机基线:任何"策略学会了"的说法都得先有对照(与 Hibs-Physics/实验纪律一致:
单 seed 结论不作数、必须 ≥2 路对照)。这里给出两路:
  ① RandomPolicy  —— 均匀随机动作(上界参考:环境本身好不好玩)
  ② TabularQ      —— ε-greedy 表格 Q-learning(状态 = (μ 分箱, 步进度分箱))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class RandomPolicy:
    """均匀随机动作基线."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))

    def __repr__(self) -> str:  # pragma: no cover
        return f"RandomPolicy(n_actions={self.n_actions})"


@dataclass
class TabularQ:
    """表格 Q-learning:状态 (μ, 步进度) 分箱,动作 = 停 / 三个增益档."""

    n_actions: int
    mu_bins: int = 30
    n_bins: int = 25
    n_max: int = 400
    lr: float = 0.25
    gamma: float = 1.0          # 回合制、无折扣:目标就是总回报
    eps_start: float = 0.30
    eps_end: float = 0.02
    eps_tau_frac: float = 0.15     # ε 衰减时间常数 = 训练回合数的这个比例
    seed: int = 0

    q: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.q = np.zeros((self.mu_bins, self.n_bins, self.n_actions), dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    # ── 状态离散化 ─────────────────────────────────────────────────────────
    def encode(self, obs: np.ndarray) -> tuple[int, int]:
        mu, _margin_ratio, progress, _cost = obs
        i = min(int(mu * self.mu_bins), self.mu_bins - 1)
        j = min(int(progress * self.n_bins), self.n_bins - 1)
        return i, j

    # ── 动作 ───────────────────────────────────────────────────────────────
    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        i, j = self.encode(obs)
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[i, j]))

    def update(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        i, j = self.encode(obs)
        i2, j2 = self.encode(next_obs)
        target = reward if terminated else reward + self.gamma * float(np.max(self.q[i2, j2]))
        self.q[i, j, action] += self.lr * (target - self.q[i, j, action])


def train_tabular_q(env, episodes: int = 400, seed: int = 0, agent: TabularQ | None = None):
    """训练一个表格 Q-learning 智能体,返回 (agent, 每回合回报序列)."""
    if agent is None:
        agent = TabularQ(n_actions=env.action_space.n, n_max=env.n_max, seed=seed)
    returns: list[float] = []
    lens: list[int] = []
    tau = max(1.0, episodes * agent.eps_tau_frac)
    for ep in range(episodes):
        # ε 快速指数衰减:线性衰减时,探索噪声整轮都在"随机按停",
        # 平均回合长度只有十几步 → 智能体**从来没见过**长轨迹的长程回报,
        # 于是 G7(逼近穷举参考的 38)不可能达成 —— 这是探索设计问题,不是物理问题。
        eps = agent.eps_end + (agent.eps_start - agent.eps_end) * float(np.exp(-ep / tau))
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        total = 0.0
        steps = 0
        while True:
            a = agent.act(obs, eps=eps)
            nxt, r, terminated, truncated, _info = env.step(a)
            agent.update(obs, a, r, nxt, terminated or truncated)
            total += r
            steps += 1
            obs = nxt
            if terminated or truncated:
                break
        returns.append(total)
        lens.append(steps)
    agent.train_ep_len_mean = float(np.mean(lens)) if lens else 0.0
    return agent, returns


def evaluate(policy, env, episodes: int = 20, seed: int = 0) -> dict:
    """贪心/确定性地跑若干回合,返回回报分布 + 停步 + 终态 μ(含窗口是否守住)."""
    rets, stops, mus, lens, closed = [], [], [], [], 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        total, steps = 0.0, 0
        while True:
            a = policy.act(obs) if not isinstance(policy, TabularQ) else policy.act(obs, eps=0.0)
            obs, r, terminated, truncated, info = env.step(a)
            total += r
            steps += 1
            if terminated or truncated:
                break
        rets.append(total)
        stops.append((steps, a == 0))     # (步数, 是否以"主动停"结束)
        lens.append(steps)
        mus.append(info["mu"])
        closed += int(info["closed"])
    return {
        "return_mean": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "stop_step_mean": float(np.mean([s0 for s0, _ in stops])),
        "stopped_by_action": int(sum(1 for _, by_stop in stops if by_stop)),
        "mu_final_mean": float(np.mean(mus)),
        "len_mean": float(np.mean(lens)),
        "closed_episodes": closed,
        "episodes": episodes,
    }


# ── 线性 Q(比表格 Q 更适合这个长视野问题)──────────────────────────────────
class LinearQ:
    """线性 Q(s,a) = w[a]·φ(s),特征里显式给出"离窗口还有多远"(log 标度)。

    为什么需要它:表格 Q 在 μ×进度 上分箱(30×25)时,η=0.05 的步长(≈0.025)比一个箱还窄,
    相邻状态落进同一个箱 —— 值函数**在结构上表达不了**"再走 160 步"这件事,
    于是学出来的策略只走 20–50 步就停(实测:G7 只有参考的 0.75%)。
    这不是物理问题,是函数逼近的表达力问题。
    """

    N_FEATURES = 5

    def __init__(self, n_actions: int, lr: float = 0.05, gamma: float = 1.0,
                 eps_start: float = 0.30, eps_end: float = 0.02, eps_tau_frac: float = 0.12,
                 seed: int = 0) -> None:
        self.n_actions = n_actions
        self.lr, self.gamma = lr, gamma
        self.eps_start, self.eps_end, self.eps_tau_frac = eps_start, eps_end, eps_tau_frac
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((n_actions, self.N_FEATURES), dtype=np.float64)

    @staticmethod
    def features(obs: np.ndarray) -> np.ndarray:
        mu, margin_ratio, progress, _cost = (float(x) for x in obs)
        log_margin = float(np.clip(np.log10(max(margin_ratio, 1e-12)), -5.0, 5.0)) / 5.0
        return np.array([1.0, mu, log_margin, progress, progress * progress], dtype=np.float64)

    def q(self, obs: np.ndarray) -> np.ndarray:
        return self.w @ self.features(obs)

    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q(obs)))

    def update(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None:
        target = reward if terminated else reward + self.gamma * float(np.max(self.q(next_obs)))
        err = target - float(self.q(obs)[action])
        self.w[action] += self.lr * err * self.features(obs)


def train_linear_q(env, episodes: int = 400, seed: int = 0, agent: LinearQ | None = None):
    """训练线性 Q(与 train_tabular_q 同接口)。"""
    if agent is None:
        agent = LinearQ(n_actions=env.action_space.n, seed=seed)
    returns: list[float] = []
    lens: list[int] = []
    tau = max(1.0, episodes * agent.eps_tau_frac)
    for ep in range(episodes):
        eps = agent.eps_end + (agent.eps_start - agent.eps_end) * float(np.exp(-ep / tau))
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        total, steps = 0.0, 0
        while True:
            a = agent.act(obs, eps=eps)
            nxt, r, terminated, truncated, _info = env.step(a)
            agent.update(obs, a, r, nxt, terminated or truncated)
            total += r
            steps += 1
            obs = nxt
            if terminated or truncated:
                break
        returns.append(total)
        lens.append(steps)
    agent.train_ep_len_mean = float(np.mean(lens)) if lens else 0.0
    return agent, returns
