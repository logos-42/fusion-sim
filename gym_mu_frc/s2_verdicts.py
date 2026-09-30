r"""S2 —— 双流系统的判决量（R_ci 零假设底 / 阈值闭合 / 排序不可交换 / 盲样）。

接 S0/S1（`gym_mu_frc.two_fluid_1d`）。S2 的目标不是"再算几个工况"，而是把**判决量本身**
写死：在标准物理（μ = 0）下，R_ci 会被哪些系统性效应顶成非零；要被 3σ 分辨的最小 μ 是多少；
以及有没有第二条独立通道（排序）。

约定
----
R_ci **定义**（本模块的判决量）：绝对速率相对标准两流预测的相对超出量

    R_ci := γ_实测/γ_标准 − 1

μ = 0（标准 MHD）⟹ 真值 R_ci = 0；μ ≠ 0 ⟹ 真值 = 1/√(1−μ) − 1 ≈ μ/2。
本模块给出的是**零假设底**：由有限窗口 / 格式 / k 量化 / 噪声 / 非线性起始造成的
等效 R_ci（记作"等效 μ" = 2·R_ci，便于与 μ 阶梯直接比大小）。

诚实边界
--------
- 噪声试验把噪声加在**诊断信号**（模式振幅时间序列）上，不重跑物理；重跑物理的版本留下一步。
- 压缩/驱动是**离散算子**（不是连续流），所以"排序不可交换"检的是算子序列，不是时间积分器。
- 双流阈值用的是冷静电判据（k v0 < ω_p）；磁场只通过平衡关系（n ∝ B²）进入，**没有**动理学磁化效应。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gym_mu_frc.two_fluid_1d import Species, TwoFluid1D, closed_form_gamma, two_stream_default

MU = 1.806_139  # 不用；留占位以免误用


# ───────────────────── V1：R_ci 零假设底（系统性预算） ─────────────────────

def _gamma_of(sim: TwoFluid1D, **kw) -> float:
    return sim.run(**kw)["gamma_measured"]


def rc_null_budget(ref_mu: float = 0.0, t_end: float = 20.0, record_every: int = 4) -> dict:
    """把"标准物理下 R_ci 能假成多大"逐项量出来。

    每项都换算成**等效 μ = 2·(δγ/γ)**，方便与判决阶梯（1e−4/1e−3/1e−2）直接比。
    """
    base = two_stream_default(mu=ref_mu)
    g_ref = base.growth_analytic()[0]
    rows = []

    def rec(name, g):
        if g is None or not np.isfinite(g) or g <= 0:
            rows.append({"项": name, "γ": g, "δγ/γ": float("nan"), "等效 μ": float("inf")})
            return
        rel = abs(g / g_ref - 1.0)
        rows.append({"项": name, "γ": float(g), "δγ/γ": float(rel), "等效 μ": 2.0 * rel})

    # (a) 拟合窗口上沿（吃到非线性段会假高）
    for hi in (0.01, 0.05, 0.1, 0.2):
        rec(f"窗口上沿 fit_hi={hi}", _gamma_of(two_stream_default(mu=ref_mu),
                                              t_end=t_end, dt=0.01, record_every=record_every, fit_hi=hi))
    # (b) 窗口下沿
    for lo_mult in (5.0, 20.0, 100.0):
        s = two_stream_default(mu=ref_mu)
        rec(f"窗口下沿 fit_lo={lo_mult:g}×seed",
            s.run(t_end=t_end, dt=0.01, record_every=record_every, fit_lo=lo_mult * s.seed, fit_hi=0.05)["gamma_measured"])
    # (c) 时间步 & 网格
    for dt in (0.02, 0.01, 0.005):
        rec(f"dt={dt}", _gamma_of(two_stream_default(mu=ref_mu), t_end=t_end, dt=dt, record_every=record_every, fit_hi=0.05))
    for N in (256, 512):
        rec(f"N={N}", _gamma_of(two_stream_default(mu=ref_mu, N=N), t_end=t_end, dt=0.01, record_every=record_every, fit_hi=0.05))
        rec(f"N={N}（含 k 变化）", None)
    # (d) 积分时长（末态不同）
    for te in (15.0, 20.0, 25.0):
        rec(f"t_end={te}", _gamma_of(two_stream_default(mu=ref_mu), t_end=te, dt=0.01, record_every=record_every, fit_hi=0.05))

    vals = [r["等效 μ"] for r in rows if np.isfinite(r["等效 μ"])]
    rss = math.sqrt(sum(v * v for v in vals)) if vals else float("inf")
    return {"γ_ref（解析）": g_ref, "行": rows,
            "最大单项等效 μ": max(vals) if vals else float("inf"),
            "合成（RSS）等效 μ": rss,
            "能被 3σ 分辨的最小 μ（=3×RSS）": 3.0 * rss}


def k_quantization_budget(sim: TwoFluid1D | None = None) -> dict:
    """k 量化：周期盒把 k 钉在离散值上，与"设计 k"差 Δk ⟹ 等效 μ。

    γ(k) 在峰值附近平坦、离峰后下降；用数值 dγ/dk 把 Δk 换成 γ 的变化。
    """
    s = sim or two_stream_default()
    k0 = s.k_seed
    h = 1e-4
    from gym_mu_frc.two_fluid_1d import cold_two_stream_roots
    wp = [math.sqrt(0.5)] * 2

    def g(k):
        r = cold_two_stream_roots(k, wp, [1.0, -1.0])
        return float(np.max(r.imag))
    dgdk = (g(k0 + h) - g(k0 - h)) / (2 * h)
    g0 = g(k0)
    dk = 2.0 * math.pi / s.L / 2.0                     # 半格（最坏的一半）
    rel = abs(dgdk * dk) / g0
    return {"k0": k0, "dγ/dk": dgdk, "Δk（半格）": dk,
            "δγ/γ": rel, "等效 μ": 2.0 * rel,
            "诚实说明": "若诊断不知道真实的 k（例如只按「设计模」给 k），这一项就是不可消除的底"}


def k_quantization_budget_at_peak(m: int = 3, v0: float = 1.0) -> dict:
    """把工作点搬到**色散峰** k*（dγ/dk = 0）后，k 量化底的一阶项消失。

    做法：取盒长使某个整数模正好落在 k* = √(3/8)·ω_p/v0 上（L = 2πm/k*）。
    实测（见 S2 门输出）：这一搬把等效 μ 从 6.3e−1 降到 1e−3 量级 —— **这是设计规则，不是调参**。
    """
    from gym_mu_frc.two_fluid_1d import cold_two_stream_roots
    wp = [math.sqrt(0.5)] * 2
    k_star = math.sqrt(3.0 / 8.0) * 1.0 / v0

    def g(k):
        r = cold_two_stream_roots(k, wp, [v0, -v0])
        return float(np.max(r.imag))
    h = 1e-3
    g0 = g(k_star)
    d1 = (g(k_star + h) - g(k_star - h)) / (2 * h)
    d2 = (g(k_star + h) - 2 * g0 + g(k_star - h)) / h ** 2
    L = 2.0 * math.pi * m / k_star
    dk = math.pi / L                                    # 半格
    rel1 = abs(d1 * dk) / g0
    rel2 = abs(0.5 * d2 * dk ** 2) / g0
    return {"k*（闭式）": k_star, "取 m": m, "L": L, "Δk（半格）": dk,
            "dγ/dk（峰上应 ≈0）": d1, "d²γ/dk²": d2,
            "一阶等效 μ": 2.0 * rel1, "二阶等效 μ": 2.0 * rel2,
            "结论": "峰上工作点：k 量化底由一阶降为二阶"}


def peak_floor_scan(ms=(3, 6, 12, 24), v0: float = 1.0) -> dict:
    """峰上工作点的 k 量化底 vs 盒长：验证二阶项 ∝ 1/L²（即 ∝ 1/m²）。

    结论（实测）：把工作点搬到色散峰只消掉**一阶**项；剩下的二阶项仍是主系统误差，
    必须靠**盒长**（或独立测 k）解决 —— μ=1e−3 档需要 m ≳ 30 个波长。
    """
    from gym_mu_frc.two_fluid_1d import cold_two_stream_roots
    wp = [math.sqrt(0.5)] * 2
    k_star = math.sqrt(3.0 / 8.0) * 1.0 / v0

    def g(k):
        r = cold_two_stream_roots(k, wp, [v0, -v0])
        return float(np.max(r.imag))
    h = 1e-3
    g0 = g(k_star)
    d2 = (g(k_star + h) - 2 * g0 + g(k_star - h)) / h ** 2
    rows = []
    for m in ms:
        L = 2.0 * math.pi * m / k_star
        dk = math.pi / L
        rel2 = abs(0.5 * d2 * dk ** 2) / g0
        rows.append({"m（波长数）": m, "L": L, "Δk（半格）": dk, "二阶等效 μ": 2.0 * rel2})
    # 1/L² 律：log-log 斜率应 ≈ −2
    xs = np.log([r["m（波长数）"] for r in rows])
    ys = np.log([r["二阶等效 μ"] for r in rows])
    slope = float(np.polyfit(xs, ys, 1)[0])
    need_m = float(np.sqrt(2.0 * abs(0.5 * d2 * (math.pi * k_star / (2 * math.pi)) ** 2) / g0 / 5e-4))
    return {"行": rows, "log-log 斜率（应 ≈ −2）": slope,
            "μ=1e−3 档所需波长数 m（等效 μ ≤ 5e−4）": need_m}


def required_sigma_for(mu_target: float, sigma_ref: float = 1e-2,
                       equiv_mu_at_sigma_ref: float = 1.369e-2) -> dict:
    """反推仪器规格：σ_signal 需要多小，才能把 μ_target 做到 3σ。

    噪声底严格 ∝ σ（实测 σ=1e−2/1e−3/1e−4 的等效 μ 恰好 ×10 递减）⟹ 线性外推。
    """
    sigma_needed = sigma_ref * mu_target / equiv_mu_at_sigma_ref
    return {"μ 目标": mu_target, "需要的 σ_signal（3σ）": sigma_needed,
            "等效于单发相对精度": sigma_needed,
            "口径": f"以 σ={sigma_ref:g} 时 3σ 等效 μ={equiv_mu_at_sigma_ref:g} 为基准线性外推"}


# ───────────────────── V2：双流阈值曲线（μ, B, n 闭合） ─────────────────────

def field_required(n: float, Ti_eV: float, Te_eV: float, mu: float, v_flow: float = 0.0) -> dict:
    """平衡所需磁场（压力平衡 + 流动项）与 μ 的关系。

    B²/(2μ0) = n k_B (T_i + T_e) + n m_i (1−μ) v_flow²
    ⟹ B_req ∝ √(1−μ)（流动项那一支）；纯热压支不随 μ 变。
    本仓框架的预言：约束替代支按 √(1−μ) 降。
    """
    mu0, kB, e, amu = 4e-7 * math.pi, 1.380649e-23, 1.602176634e-19, 1.66053906660e-27
    mi = 2.0 * amu
    p = n * kB * (Ti_eV + Te_eV) * e / e            # n k_B T（eV→J 已含 e）
    p_th = n * (Ti_eV + Te_eV) * e
    p_flow = n * mi * (1.0 - mu) * v_flow ** 2
    B = math.sqrt(2.0 * mu0 * (p_th + p_flow))
    return {"n [m^-3]": n, "mu": mu, "v_flow [m/s]": v_flow,
            "p_th [Pa]": p_th, "p_flow [Pa]": p_flow,
            "B_req [T]": B,
            "B_req/B_req(μ=0)": B / math.sqrt(2.0 * mu0 * (p_th + n * mi * v_flow ** 2))}


def two_stream_threshold_scan(mus=(0.0, 1e-3, 1e-2, 1e-1), ns=(1e19, 1e20, 1e21),
                              Te_eV: float = 100.0, L: float = 1.0,
                              B_ceiling_T: float = 1.099) -> list[dict]:
    """(μ, n) 网格上的双流稳定性 + 磁场需求 —— μ 的两面。

    不稳带（冷静电）：k v0 < ω_p。取 k = 2π/L（最长波长）与 v0 = 由 RMF 设定的漂移。
    这里把 v0 取成"该 n、B 下热压对应的声速"以给出量级判断（并在返回里标出口径）。
    """
    e, eps0, amu = 1.602176634e-19, 8.8541878128e-12, 1.66053906660e-27
    mi = 2.0 * amu
    rows = []
    for mu in mus:
        for n in ns:
            wp = math.sqrt(n * e ** 2 / (eps0 * mi * (1.0 - mu)))     # 离子：∝1/√(1−μ)
            k = 2.0 * math.pi / L
            cs = math.sqrt(Te_eV * e / mi)
            thr = wp / k                                              # 不稳条件 k v0 < ω_p
            B = field_required(n, Te_eV, Te_eV, mu)["B_req [T]"]
            rows.append({
                "mu": mu, "n [m^-3]": n,
                "ω_pi [rad/s]": wp, "k [1/m]": k,
                "不稳阈值 v0 < ω_pi/k [m/s]": thr,
                "声速 c_s [m/s]": cs,
                "用 c_s 作 v0 是否不稳": bool(cs < thr),
                "B_req [T]": B,
                "超 B 天花板(1.099 T)": bool(B > B_ceiling_T),
            })
    return rows


# ───────────────────── V3：排序不可交换（TD15–TD17 的仿真版） ─────────────────────

@dataclass
class Driven1D(TwoFluid1D):
    """带交变外场驱动（RMF 类比）的两流体；可施加离散压缩算子。"""

    E_drive: float = 0.0
    omega_d: float = 1.0
    t_now: float = field(default=0.0, init=False)

    def _rhs(self, n_list, v_list):
        dn, dv = super()._rhs(n_list, v_list)
        if self.E_drive != 0.0:
            E = self.E_drive * math.cos(self.omega_d * self.t_now)
            dv = [d - (s.q / (s.m * (1.0 - s.mu))) * E for s, d in zip(self.species, dv)]
        return dn, dv

    def step(self, dt: float) -> None:
        super().step(dt)
        self.t_now += dt


def compress(sim: Driven1D, eps: float) -> None:
    """离散压缩算子 C(ε)：盒长 ∝(1−ε) ⟹ n → n/(1−ε)，v → v(1−ε)（位形空间体积守恒口径）。"""
    sim.n = [n * (1.0 + eps) for n in sim.n]
    sim.v = [v * (1.0 - eps) for v in sim.v]


def _functional(sim: Driven1D, mode: int = 4) -> float:
    """末态泛函：模式振幅（ρ̂ 的第 mode 模）+ 动能，归一用。"""
    amp = sim.mode_amp(mode)
    ke = sum(float(np.mean(n * v ** 2)) for n, v in zip(sim.n, sim.v))
    return amp + ke


def ordering_test(mu: float, dt: float = 0.02, t_drive: float = 4.0, eps: float = 0.05,
                  E_drive: float = 0.02, omega_d: float = 2.0, seed: float = 1e-4) -> dict:
    """D 后 C vs C 后 D：同一组算子、同一总时长，只换顺序。"""
    def mk():
        return Driven1D([Species(0.5, -1.0, 1.0, +1.0, mu), Species(0.5, -1.0, 1.0, -1.0, mu)],
                        L=29.02, N=512, seed=seed, E_drive=E_drive, omega_d=omega_d)

    a = mk()                                    # D → C
    steps = int(t_drive / dt)
    for _ in range(steps):
        a.step(dt)
    compress(a, eps)
    fa = _functional(a)

    b = mk()                                    # C → D
    compress(b, eps)
    for _ in range(steps):
        b.step(dt)
    fb = _functional(b)
    scale = max(abs(fa), abs(fb), 1e-300)
    return {"mu": mu, "F(D→C)": fa, "F(C→D)": fb,
            "相对顺序差 |Δ|/scale": abs(fa - fb) / scale, "eps": eps, "E_drive": E_drive}


def ordering_num_floor(dt_list=(0.02, 0.01), N_list=(256, 512), mu: float = 0.0) -> dict:
    """排序差的**数值底**：同一物理，只换 dt / N，看 Δ 抖动多少。

    没有这一项，"排序不可交换"当通道用就是空话（无法判断 μ 依赖部分是否过了噪音）。
    """
    rows = []
    for dt in dt_list:
        for N in N_list:
            def mk():
                return Driven1D([Species(0.5, -1.0, 1.0, +1.0, mu), Species(0.5, -1.0, 1.0, -1.0, mu)],
                                L=29.02, N=N, seed=1e-4, E_drive=0.02, omega_d=2.0)
            a = mk()
            for _ in range(int(4.0 / dt)):
                a.step(dt)
            compress(a, 0.05)
            fa = _functional(a)
            b = mk(); compress(b, 0.05)
            for _ in range(int(4.0 / dt)):
                b.step(dt)
            fb = _functional(b)
            rows.append({"dt": dt, "N": N, "Δ": abs(fa - fb) / max(abs(fa), abs(fb)),
                         "F": fa})
    ds = [r["Δ"] for r in rows]
    return {"行": rows, "Δ 均值": float(np.mean(ds)), "Δ 数值底（极差）": float(max(ds) - min(ds))}


# ───────────────────── V4：盲样（裁决规则 + 随机基线） ─────────────────────

def noise_montecarlo(mu: float, sigma_rel: float = 1e-2, n_trials: int = 2000,
                     dt: float = 0.01, t_end: float = 20.0, seed: int = 0,
                     fit_hi: float = 0.05) -> dict:
    """噪声加在**诊断信号**（模式振幅时间序列）上，重跑拟合流程。

    返回 γ̂ 的相对标准差与 3σ 等效 μ（= 2·3σ_γ/γ）。
    """
    rng = np.random.default_rng(seed)
    sim = two_stream_default(mu=mu)
    r = sim.run(t_end=t_end, dt=dt, record_every=4, fit_hi=fit_hi)
    t = np.asarray(r["t"])
    a = np.asarray(r["amp"])
    m = (a > 20.0 * sim.seed) & (a < fit_hi)
    ts, loga = t[m], np.log(a[m])
    n_win = ts.size
    gammas = np.empty(n_trials)
    for i in range(n_trials):
        noise = 1.0 + sigma_rel * rng.standard_normal(n_win)
        y = loga + np.log(np.clip(noise, 1e-6, None))
        A = np.vstack([ts, np.ones(n_win)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        gammas[i] = sol[0]
    g_mean, g_std = float(gammas.mean()), float(gammas.std(ddof=1))
    g_ideal = sim.growth_analytic()[0]
    return {"mu": mu, "σ_rel(信号)": sigma_rel, "n_trials": n_trials,
            "拟合窗口点数": n_win,
            "γ̂ 均值": g_mean, "γ̂ 标准差": g_std,
            "γ̂ 相对标准差": g_std / g_mean,
            "3σ 等效 μ": 2.0 * 3.0 * g_std / g_mean,
            "理想 γ": g_ideal, "γ̂−理想": g_mean - g_ideal}


def null_threshold(protocol: str = "fixed", sigma_rel: float = 1e-2, n_trials: int = 800,
                   t_end: float = 15.0, seed: int = 1) -> dict:
    """在**指定协议**下用 μ=0 集合标定 5% 假阳性阈值（等效 μ 口径）。"""
    rng = np.random.default_rng(seed)
    fit_hi = 0.05 if protocol == "fixed" else 0.2
    sim0 = two_stream_default(mu=0.0)
    ref = sim0.run(t_end=t_end, dt=0.01, record_every=4, fit_hi=fit_hi)
    t, a0 = np.asarray(ref["t"]), np.asarray(ref["amp"])
    m = (a0 > 20.0 * sim0.seed) & (a0 < fit_hi)
    ts, loga0 = t[m], np.log(a0[m])
    A = np.vstack([ts, np.ones(ts.size)]).T

    def est(y):
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(sol[0])
    g_ref = est(loga0)
    rc0 = np.empty(n_trials)
    for i in range(n_trials):
        y = loga0 + np.log(np.clip(1.0 + sigma_rel * rng.standard_normal(ts.size), 1e-6, None))
        rc0[i] = 2.0 * (est(y) / g_ref - 1.0)
    return {"protocol": protocol, "阈值（95% 分位, 等效 μ）": float(np.quantile(rc0, 0.95)),
            "σ_signal": sigma_rel, "n_win": int(ts.size), "γ_ref(拟合)": g_ref}


def blind_eval(protocol: str, thr: float, mu_classes=(0.0, 1e-3, 1e-2),
               sigma_rel: float = 1e-2, n_trials: int = 600, t_end: float = 15.0,
               seed: int = 1) -> dict:
    """用给定阈值评估某协议：假阳性率（μ=0 类）与检出率（各 μ 类）。

    关键是**交叉评估**：阈值由 fixed 协议标定，拿去判 sloppy 协议的数据 ⟹ 协议漂移造成的假阳性。
    """
    rng = np.random.default_rng(seed + 7)
    fit_hi = 0.05 if protocol == "fixed" else 0.2
    sim0 = two_stream_default(mu=0.0)
    ref = sim0.run(t_end=t_end, dt=0.01, record_every=4, fit_hi=fit_hi)
    t, a0 = np.asarray(ref["t"]), np.asarray(ref["amp"])
    m = (a0 > 20.0 * sim0.seed) & (a0 < fit_hi)
    ts = t[m]
    A = np.vstack([ts, np.ones(ts.size)]).T

    def est(y):
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(sol[0])
    g_ref_fixed = null_threshold("fixed", sigma_rel=sigma_rel, n_trials=200, t_end=t_end, seed=seed)["γ_ref(拟合)"]

    out = {"protocol": protocol, "阈值（来自 fixed）": thr, "类": {}}
    for mu in mu_classes:
        s_ = two_stream_default(mu=mu)
        r = s_.run(t_end=t_end, dt=0.01, record_every=4, fit_hi=fit_hi)
        aa = np.asarray(r["amp"])[m]
        stat = np.empty(n_trials)
        for i in range(n_trials):
            y = np.log(aa) + np.log(np.clip(1.0 + sigma_rel * rng.standard_normal(ts.size), 1e-6, None))
            stat[i] = 2.0 * (est(y) / g_ref_fixed - 1.0)
        out["类"][mu] = {"真值 R_ci": 1.0 / math.sqrt(1.0 - mu) - 1.0,
                         "检出率": float(np.mean(stat > thr)),
                         "统计量中位数": float(np.median(stat))}
    out["假阳性率(μ=0 类)"] = out["类"][0.0]["检出率"] if 0.0 in out["类"] else None
    out["随机基线检出率(任意 μ≠0)"] = 0.5
    return out
