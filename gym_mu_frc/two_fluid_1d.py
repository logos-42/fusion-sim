r"""1-D 冷两流体 / 双反向流（counter-streaming）——**μ 可关闭**的最小可信核。

hushfusion 公开判决漏斗 **G0** 的执行层（S0/S1）：把「标准两流 / MHD 会给出什么」算到
低于诊断分辨率，并反推诊断需要多高的分辨率。

模型（静电、1-D、周期）：

    ∂n_s/∂t + ∂(n_s v_s)/∂x = 0
    ∂v_s/∂t + v_s ∂v_s/∂x = −(q_s / (m_s(1 − μ_s))) ∂φ/∂x
    ∂²φ/∂x² = −ρ/ε0 ,  ρ = Σ_s q_s n_s      （固定中性化背景）

单位：ε0 = q = m = 1，总密度 n_tot = 1 ⟹ 电子束对（各 n/2，速度 ±v0）在 μ = 0 时
ω_pe = 1，v0 = 1。**闭式（推出来的，不是背的）**：

    ε(ω,k) = 1 − Σ_j ω_j²/(ω − k v_j)² = 0
    ⟹ (ω² − k²v0²)² = ω_p²(ω² + k²v0²)          （对称双束，ω_p² = Σω_j² = ω_pe²）
    ⟹ γ² = [−(2a + ω_p²) + √(8aω_p² + ω_p⁴)]/2 ,  a = k²v0²
    ⟹ **γ_max = ω_p/(2√2)**（a = 3ω_p²/8），**k* v0 = √(3/8)·ω_p ≈ 0.6124 ω_p**
    不稳定带：k v0 < ω_p

μ 的注入：m_s → m_s(1 − μ_s)（惯性项与等离子体频率**同时**变），依据 ProjectionPhysics
的 TD 系列 / FC11 线。**硬纪律**：μ 必须可一键关闭，μ = 0 时退化为标准两流体。

**签名可见性定理（本模块的关键结论）**：把**所有**物种的质量同倍重标 m → m(1−μ)
等价于时间重标定 ⟹ 任何「同源速率的比值」（γ/ω_p、γ/Ω_ci …）**零签名**（数值已证到
机器精度）；只有相对**外部时钟**（RMF 驱动频率、回路频率、壁钟约束时间）的**绝对速率**
才带 1/√(1−μ)；只有**物种间不匹配**（μ_e ≠ μ_i）才会在跨尺度比值里露头。

**三条独立路线（2026-09-29 实测，互为交叉验证）**：
1. 手推闭式 γ_max = ω_p/(2√2)、k* v0 = √(3/8)ω_p ⟹ 数值扫 k 命中 **0.000000%**；
2. 介电函数多项式根：与 4×4 线性化**同谱**（差 e^{∓iωt} 约定 ⟹ 相差一个 i 因子）；
3. 1-D 时域谱方法（不稳定本征矢作初值）：γ 与解析差 **−0.00%**，R² = 1.00000，dt 减半不动。

诚实边界：冷静电模型没有 Debye 屏蔽 / 动理学阻尼；非线性饱和不在本模块（S4 才上动理学）；
两条谱的一致性只在同一 k 上验过。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ───────────────────────── S0：解析谱 ─────────────────────────

def cold_two_stream_roots(k: float, wp: list[float], v: list[float]) -> np.ndarray:
    """冷静电介电函数 ε(ω,k)=0 的全部根（复）。

    多项式（降幂）：Π_l (ω − k v_l)² − Σ_j ω_j² Π_{l≠j} (ω − k v_l)² = 0
    """
    from numpy.polynomial import polynomial as P

    lead = np.array([1.0 + 0j])
    for vv in v:
        f = np.array([1.0 + 0j, -k * vv])          # (ω − k v)
        lead = P.polymul(lead, P.polymul(f, f))
    out = lead.astype(complex)
    deg = len(out)
    for j, w in enumerate(wp):
        t = np.array([w ** 2 + 0j])
        for l, v2 in enumerate(v):
            if l == j:
                continue
            f = np.array([1.0 + 0j, -k * v2])
            t = P.polymul(t, P.polymul(f, f))
        out[deg - len(t):] -= t                    # 降幂对齐：低阶项贴尾
    return np.roots(out)


def closed_form_gamma(wp_total: float) -> float:
    """对称冷双束：γ_max = ω_p/(2√2)。"""
    return wp_total / (2.0 * math.sqrt(2.0))


def closed_form_kstar(wp_total: float, v0: float) -> float:
    """k* v0 = √(3/8)·ω_p。"""
    return math.sqrt(3.0 / 8.0) * wp_total / v0


def max_growth_rate(wp_each: list[float], v: list[float],
                    k_lo: float = 0.02, k_hi: float = 1.4, n_k: int = 2800) -> tuple[float, float]:
    """扫 k 取最大增长率，并做三点抛物线细化（否则会被 k 网格钉住）：(k*, γ_max)。"""
    ks = np.linspace(k_lo, k_hi, n_k)
    gs = []
    for k in ks:
        r = cold_two_stream_roots(k, wp_each, v)
        gs.append(float(np.max(r.imag)) if r.size else -1.0)
    gs = np.asarray(gs)
    i = int(np.argmax(gs))
    k_best, g_best = float(ks[i]), float(gs[i])
    if 0 < i < n_k - 1:                      # 三点抛物线细化
        y0, y1, y2 = gs[i - 1], gs[i], gs[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0.0:
            dk = 0.5 * (y0 - y2) / denom
            if abs(dk) <= 1.0:
                k_best += dk * (ks[1] - ks[0])
                g_best = y1 - 0.25 * (y0 - y2) * dk
    return k_best, g_best


def resolution_table(mus=(1e-4, 1e-3, 1e-2, 1e-1), per_shot: float = 1e-3) -> list[dict]:
    """判决量：μ 签名的相对效应量与所需诊断精度。

    绝对速率比 γ(μ)/γ(0) = 1/√(1−μ) ⟹ 相对效应量 ≈ μ/2（小 μ）。
    「3σ 检测」要求单发相对精度 < 效应量/3；N 发平均放宽 √N：N = (per_shot/(效应量/3))²。
    注意：该效应量**只在相对外部时钟的绝对速率上出现**；同源速率之比带**零**签名。
    """
    rows = []
    for mu in mus:
        ratio = 1.0 / math.sqrt(1.0 - mu)
        rel = ratio - 1.0
        p_req = rel / 3.0
        rows.append({
            "mu": mu,
            "γ(μ)/γ(0) = 1/√(1−μ)": ratio,
            "相对效应量 δγ/γ": rel,
            "小 μ 近似 μ/2": mu / 2.0,
            "3σ 单发所需相对精度": p_req,
            "若单发精度 1e−3，需要的发数": (per_shot / p_req) ** 2 if p_req > 0 else float("inf"),
        })
    return rows


def ion_stream_constraint(n: float, Te_eV: float, mi_amu: float, L: float) -> dict:
    """反向流的稳定性约束：两档，一档**算出来**，一档**教科书口径**（标注清楚）。

    档 A（本模块可算，冷静电两流判据）：不稳定带 k v0 < ω_p，峰值 γ_max = ω_p/(2√2)。
    档 B（教科书口径，本模块**未复算**）：热电子中性化下，离子反向流的阈值落在离子声速
    c_s = √(T_e/m_i) 上（相对漂移 ≳ c_s 即失稳）⟹ 工程约束 = **马赫数 < 1**。
    RMF 驱动可能故意拿不稳定性当加热源 —— 那也必须进账，不是忽略。
    """
    e, eps0, amu = 1.602176634e-19, 8.8541878128e-12, 1.66053906660e-27
    mi = mi_amu * amu
    cs = math.sqrt(Te_eV * e / mi)                 # 档 B
    wpi = math.sqrt(n * e ** 2 / (eps0 * mi))
    k_min = 2.0 * math.pi / L
    v_cold = wpi / k_min                           # 档 A
    return {
        "n [m^-3]": n, "T_e [eV]": Te_eV, "m_i [amu]": mi_amu, "L [m]": L,
        "档A ω_pi [rad/s]": wpi,
        "档A k_min [1/m]": k_min,
        "档A 冷两流阈值 v0 ≳ ω_pi/k_min [m/s]": v_cold,
        "档A 冷两流峰值 γ_max = ω_pi/(2√2) [1/s]": wpi / (2.0 * math.sqrt(2.0)),
        "档B c_s = √(T_e/m_i) [m/s]": cs,
        "档B 是否比档A 更严": bool(cs < v_cold),
        "档B 工程约束": "反向流 v0 < c_s ⟹ Mach < 1（教科书口径，本模块未复算）",
    }


# ───────────────────────── S1：1-D 时域两流体 ─────────────────────────

@dataclass
class Species:
    n0: float          # 平均密度
    q: float           # 电荷
    m: float           # 质量
    v0: float          # 初始漂移速度（±v0 即双反向流）
    mu: float = 0.0    # 质量消除参数（本物种）；0 = 关闭


@dataclass
class TwoFluid1D:
    """静电 1-D 周期两流体（谱方法 + RK4）。μ 可一键关闭（mu = 0）。"""

    species: list[Species]
    L: float = 29.02
    N: int = 512
    eps0: float = 1.0
    seed: float = 1e-4
    seed_mode: int = 4
    seed_kind: str = "eigen"      # "eigen"（推荐，纯指数增长）| "cos"（只扰密度：先出拍频）
    x: np.ndarray = field(init=False)
    kx: np.ndarray = field(init=False)
    n: list = field(init=False)
    v: list = field(init=False)

    def __post_init__(self):
        self.x = np.arange(self.N) * (self.L / self.N)
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        self.n, self.v = [], []
        if self.seed_kind == "eigen":
            u = self.unstable_eigenvector()
            for s, us_n, us_v in zip(self.species, u[0::2], u[1::2]):
                xm = self.k_seed * self.x
                self.n.append(s.n0 + self.seed * np.real(us_n * np.exp(1j * xm)))
                self.v.append(np.full(self.N, s.v0) + self.seed * np.real(us_v * np.exp(1j * xm)))
        else:
            for s in self.species:
                self.n.append(s.n0 * (1.0 + self.seed * np.cos(self.seed_mode * 2 * np.pi * self.x / self.L)))
                self.v.append(np.full(self.N, s.v0))

    @property
    def k_seed(self) -> float:
        return self.seed_mode * 2.0 * np.pi / self.L

    # ---- 线性化：X' = M X，X = (δn̂_1, δv̂_1, δn̂_2, δv̂_2, …) ----
    def linear_matrix(self, k: float | None = None) -> np.ndarray:
        k = self.k_seed if k is None else k
        M = np.zeros((2 * len(self.species), 2 * len(self.species)), dtype=complex)
        for a, sa in enumerate(self.species):
            M[2 * a, 2 * a] = -1j * k * sa.v0
            M[2 * a, 2 * a + 1] = -1j * k * sa.n0
            M[2 * a + 1, 2 * a + 1] = -1j * k * sa.v0
            for b, sb in enumerate(self.species):
                # Gauss: ∇²φ = −ρ/ε0 ⟹ −k²φ̂ = −ρ̂/ε0 ⟹ φ̂ = ρ̂/(ε0k²)（**正号**）
                # ⟹ ∂δv_a/∂t += −i (q_a/(m_a(1−μ_a)))/(ε0 k) · q_b δn_b
                M[2 * a + 1, 2 * b] += (-1j / (self.eps0 * k)) * (sa.q / (sa.m * (1.0 - sa.mu))) * sb.q
        return M

    def unstable_eigenvector(self) -> np.ndarray:
        """最大 Re(λ) 的本征矢（作初值：纯指数增长，无拍频瞬态）。"""
        w, V = np.linalg.eig(self.linear_matrix())
        return V[:, int(np.argmax(w.real))]

    def growth_analytic(self) -> tuple[float, float]:
        """线性化的解析 (γ, ω)。与 cold_two_stream_roots 同谱（差 i 因子约定）。"""
        w, _ = np.linalg.eig(self.linear_matrix())
        i = int(np.argmax(w.real))
        return float(w[i].real), float(w[i].imag)

    # ---- 谱求导（2/3 去混叠）----
    def _d_dx(self, f: np.ndarray) -> np.ndarray:
        fh = np.fft.fft(f)
        cut = int(self.N / 3)
        fh[cut:self.N - cut] = 0.0
        return np.real(np.fft.ifft(1j * self.kx * fh))

    def _phi_for(self, n_list) -> np.ndarray:
        rho = np.zeros(self.N)
        for s, n in zip(self.species, n_list):
            rho += s.q * n
        rh = np.fft.fft(rho)
        phi_h = np.zeros(self.N, dtype=complex)
        mask = self.kx != 0.0
        phi_h[mask] = rh[mask] / (self.eps0 * self.kx[mask] ** 2)
        return np.real(np.fft.ifft(phi_h))

    def _rhs(self, n_list, v_list):
        dphi = self._d_dx(self._phi_for(n_list))
        dn, dv = [], []
        for s, n, v in zip(self.species, n_list, v_list):
            dn.append(-self._d_dx(n * v))
            dv.append(-v * self._d_dx(v) - (s.q / (s.m * (1.0 - s.mu))) * dphi)
        return dn, dv

    def step(self, dt: float) -> None:
        n0, v0 = self.n, self.v
        k1n, k1v = self._rhs(n0, v0)
        a = ([n + 0.5 * dt * x for n, x in zip(n0, k1n)], [v + 0.5 * dt * x for v, x in zip(v0, k1v)])
        k2n, k2v = self._rhs(*a)
        b = ([n + 0.5 * dt * x for n, x in zip(n0, k2n)], [v + 0.5 * dt * x for v, x in zip(v0, k2v)])
        k3n, k3v = self._rhs(*b)
        c = ([n + dt * x for n, x in zip(n0, k3n)], [v + dt * x for v, x in zip(v0, k3v)])
        k4n, k4v = self._rhs(*c)
        self.n = [n + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                  for n, k1, k2, k3, k4 in zip(n0, k1n, k2n, k3n, k4n)]
        self.v = [v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                  for v, k1, k2, k3, k4 in zip(v0, k1v, k2v, k3v, k4v)]

    def mode_amp(self, mode: int) -> float:
        rho = np.zeros(self.N)
        for s, n in zip(self.species, self.n):
            rho += s.q * n
        return float(abs(np.fft.fft(rho)[mode]))

    def run(self, t_end: float, dt: float, record_every: int = 4,
            fit_lo: float | None = None, fit_hi: float = 0.05) -> dict:
        """跑时域；线性区拟合窗口按**绝对振幅**划（默认 [20×seed, 0.05]）。

        不能按「末态振幅的百分比」划 —— 末态已进非线性（振幅 O(10)），那样会把饱和段
        算进斜率（实测偏 −14%）。
        """
        steps = int(t_end / dt)
        ts, amps = [], []
        for i in range(steps):
            self.step(dt)
            if i % record_every == 0:
                ts.append((i + 1) * dt)
                amps.append(self.mode_amp(self.seed_mode))
        ts, amps = np.array(ts), np.array(amps)
        a_max = float(amps.max())
        lo = 20.0 * self.seed if fit_lo is None else fit_lo
        m = (amps > lo) & (amps < fit_hi)
        gamma, r2, n_win = float("nan"), float("nan"), int(m.sum())
        if n_win >= 5:
            A = np.vstack([ts[m], np.ones(n_win)]).T
            sol, res, *_ = np.linalg.lstsq(A, np.log(amps[m]), rcond=None)
            ss_tot = float(np.sum((np.log(amps[m]) - np.log(amps[m]).mean()) ** 2))
            gamma = float(sol[0])
            r2 = float(1.0 - res[0] / ss_tot) if ss_tot > 0 else float("nan")
        return {"gamma_measured": gamma, "fit_r2": r2, "amp_final": a_max, "fit_n": n_win,
                "fit_amp_lo": lo, "fit_amp_hi": fit_hi, "t": ts, "amp": amps}


def two_stream_default(mu: float = 0.0, L: float = 29.02, N: int = 512) -> TwoFluid1D:
    """默认工况：两束电子各 n/2、速度 ±1，中性化固定背景；μ 同时作用在两束上。"""
    return TwoFluid1D([Species(0.5, -1.0, 1.0, +1.0, mu),
                       Species(0.5, -1.0, 1.0, -1.0, mu)], L=L, N=N)


def analytic_gamma_at_mode(sim: TwoFluid1D) -> tuple[float, float, float]:
    """种子模上的（解析 γ, k, ω_p），ω_p 按全部物种重标后计。"""
    wp_tot = math.sqrt(sum(s.n0 * s.q ** 2 / (sim.eps0 * s.m * (1.0 - s.mu)) for s in sim.species))
    wp_each = math.sqrt(wp_tot ** 2 / len(sim.species))
    r = cold_two_stream_roots(sim.k_seed, [wp_each] * len(sim.species), [s.v0 for s in sim.species])
    return (float(np.max(r.imag)) if r.size else float("nan")), sim.k_seed, wp_tot
