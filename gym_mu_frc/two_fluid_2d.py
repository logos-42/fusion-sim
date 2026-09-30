r"""2-D 静电冷两流体（S3 的物理内核）——**沿用 1-D 的接口与纪律**。

模型（静电、2-D 周期盒、物种 s ∈ {i, e} 或双束）：

    ∂n_s/∂t + ∇·(n_s u_s) = 0
    ∂u_s/∂t + (u_s·∇)u_s = (q_s / (m_s(1 − μ_s))) E ,  E = −∇φ
    ∇²φ = −ρ/ε0 ,  ρ = Σ_s q_s n_s

单位与 1-D 一致（ε0 = q = m = 1，n_tot = 1，v0 = 1 ⟹ ω_pe = 1）。

**μ 注入**：与 1-D 同处：m_s → m_s(1−μ_s)，**惯性项与 ω_p 同时变**，可一键关闭（μ = 0）。
**为什么必须升级到 2-D**：S2 给出设计规则「工作点放色散峰 + 盒长 m ≳ 34 个波长」；
2-D 才能给出 (a) 斜模 (k_x, k_y ≠ 0) 的零点偏移、(b) 横向/丝化通道、(c) 2-D 诊断的空间平均
对 R_ci 的影响 —— 这三样在 1-D 里**不存在**，是本题必须补的自由度。

**S3 的第一道门（T2D-0，回归门）**：初值只含 ky = 0 模态时（非线性不会造出 ky ≠ 0 分量），
2-D 求解器必须**逐点重现** 1-D 求解器的同一结果。这一条把「2-D 写错了」与「2-D 有新物理」
彻底分开 —— 没有它，任何 2-D 差异都无法归因。

诚实边界：冷静电，无 Debye 屏蔽 / 无动理学；去混叠用 2/3 规则（与 1-D 同阶）；
本模块只到「线性增长 + 早期非线性」，饱和与热化不在内。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gym_mu_frc.two_fluid_1d import Species, TwoFluid1D, analytic_gamma_at_mode


@dataclass
class TwoFluid2D(TwoFluid1D):
    """静电 2-D 周期两流体（谱方法 + RK4）。字段布局：[x, y]。"""

    Ny: int = 0                     # 0 ⟹ 取 N
    seed_modes: tuple = (4, 0)      # (ix, iy) 种子模；iy = 0 ⟹ 纯 ky = 0（回归门用）
    x2: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    ky: np.ndarray = field(init=False)
    KX: np.ndarray = field(init=False)
    KY: np.ndarray = field(init=False)
    K2: np.ndarray = field(init=False)
    _mask: np.ndarray = field(init=False)
    n: list = field(init=False)
    vx: list = field(init=False)
    vy: list = field(init=False)

    def __post_init__(self):
        if self.seed_kind != "eigen":
            raise NotImplementedError("2-D 只实现 eigen 种子（cos 种子会出拍频，见 S1 陷阱）")
        if self.Ny == 0:
            self.Ny = self.N
        self.x2 = np.arange(self.N) * (self.L / self.N)
        self.y = np.arange(self.Ny) * (self.L / self.Ny)
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(self.Ny, d=self.L / self.Ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2[0, 0] = 1.0                                   # 均值模：φ 无关（中性）
        # 2/3 去混叠：|i| > N/3 的模清零（两个方向同时）
        ix = np.abs(np.fft.fftfreq(self.N, d=1.0 / self.N))
        iy = np.abs(np.fft.fftfreq(self.Ny, d=1.0 / self.Ny))
        self._mask = ((ix <= self.N / 3.0)[:, None] & (iy <= self.Ny / 3.0)[None, :])

        # 初值：种子模的本征矢沿 x，y 方向均匀（iy = 0）
        u = self.unstable_eigenvector()                        # 用 1-D 线性化（ky = 0 子空间）
        ix0, iy0 = self.seed_modes
        phase = np.exp((1j * ix0 * 2.0 * np.pi / self.L) * self.x2)[:, None] \
            * np.exp((1j * iy0 * 2.0 * np.pi / self.L) * self.y)[None, :]
        self.n, self.vx, self.vy = [], [], []
        for s, us_n, us_v in zip(self.species, u[0::2], u[1::2]):
            self.n.append(s.n0 + self.seed * np.real(us_n * phase))
            self.vx.append(s.v0 + self.seed * np.real(us_v * phase))
            self.vy.append(np.zeros((self.N, self.Ny)))

    # ---- 谱求导（2/3 去混叠）----
    def _grad(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fh = np.fft.fft2(f)
        fh = np.where(self._mask, fh, 0.0)
        return (np.real(np.fft.ifft2(1j * self.KX * fh)),
                np.real(np.fft.ifft2(1j * self.KY * fh)))

    def _phi_for2(self, n_list) -> np.ndarray:
        rho = np.zeros((self.N, self.Ny))
        for s, n in zip(self.species, n_list):
            rho = rho + s.q * n
        return np.real(np.fft.ifft2(np.fft.fft2(rho) / (self.eps0 * self.K2)))

    def _rhs2(self, n_list, vx_list, vy_list):
        phi = self._phi_for2(n_list)
        ex, ey = self._grad(phi)
        ex, ey = -ex, -ey
        dn, dvx, dvy = [], [], []
        for s, n, vx, vy in zip(self.species, n_list, vx_list, vy_list):
            dn.append(-self._div(n, vx, vy))
            dvx.append(-self._adv(vx, vy, vx) + (s.q / (s.m * (1.0 - s.mu))) * ex)
            dvy.append(-self._adv(vx, vy, vy) + (s.q / (s.m * (1.0 - s.mu))) * ey)
        return dn, dvx, dvy

    def _div(self, n, vx, vy) -> np.ndarray:
        """∇·(n u) = ∂x(n vx) + ∂y(n vy)，一次 FFT 两次 ifft。"""
        f = np.fft.fft2(n * vx)
        g = np.fft.fft2(n * vy)
        f = np.where(self._mask, f, 0.0)
        g = np.where(self._mask, g, 0.0)
        d = np.fft.ifft2(1j * self.KX * f) + np.fft.ifft2(1j * self.KY * g)
        return np.real(d)

    def _adv(self, vx, vy, f) -> np.ndarray:
        """(u·∇)f = vx ∂x f + vy ∂y f。

        **必须先求导再乘**：谱方法里 ∂x(vx·f) = vx ∂x f + f ∂x vx，用乘积的谱导数
        会把多出来的 f ∂x vx 也算进去（对 _adv(vx,vy,vx) 就是 vx ∂x vx 被算两次）。
        密度方程的 ∇·(n u) 是**散度**，用乘积形式才对 —— 两个算子的正确形式不同。
        """
        fx, fy = self._grad(f)
        return vx * fx + vy * fy

    def step(self, dt: float) -> None:
        n0, vx0, vy0 = self.n, self.vx, self.vy
        k1 = self._rhs2(n0, vx0, vy0)
        a = ([x + 0.5 * dt * k for x, k in zip(n0, k1[0])],
             [x + 0.5 * dt * k for x, k in zip(vx0, k1[1])],
             [x + 0.5 * dt * k for x, k in zip(vy0, k1[2])])
        k2 = self._rhs2(*a)
        b = ([x + 0.5 * dt * k for x, k in zip(n0, k2[0])],
             [x + 0.5 * dt * k for x, k in zip(vx0, k2[1])],
             [x + 0.5 * dt * k for x, k in zip(vy0, k2[2])])
        k3 = self._rhs2(*b)
        c = ([x + dt * k for x, k in zip(n0, k3[0])],
             [x + dt * k for x, k in zip(vx0, k3[1])],
             [x + dt * k for x, k in zip(vy0, k3[2])])
        k4 = self._rhs2(*c)
        w = lambda p, k1_, k2_, k3_, k4_: p + dt / 6.0 * (k1_ + 2 * k2_ + 2 * k3_ + k4_)  # noqa: E731
        self.n = [w(x, *k) for x, k in zip(n0, zip(k1[0], k2[0], k3[0], k4[0]))]
        self.vx = [w(x, *k) for x, k in zip(vx0, zip(k1[1], k2[1], k3[1], k4[1]))]
        self.vy = [w(x, *k) for x, k in zip(vy0, zip(k1[2], k2[2], k3[2], k4[2]))]

    def rho(self) -> np.ndarray:
        r = np.zeros((self.N, self.Ny))
        for s, n in zip(self.species, self.n):
            r = r + s.q * n
        return r

    def mode_amp(self, mode: int = 0, mode_y: int = 0) -> float:
        return float(abs(np.fft.fft2(self.rho())[mode, mode_y]))

    def ky_contamination(self) -> float:
        """ky ≠ 0 的总能量 / 总能量 —— 回归门里必须 ≈ 0。"""
        rh = np.fft.fft2(self.rho())
        tot = float(np.sum(np.abs(rh) ** 2))
        if tot == 0.0:
            return 0.0
        zero_y = np.zeros_like(rh)
        zero_y[:, 0] = rh[:, 0]
        zero_y[:, self.Ny // 2] = rh[:, self.Ny // 2] if self.Ny % 2 == 0 else 0.0
        return float(np.sum(np.abs(rh - zero_y) ** 2) / tot)

    def run2d(self, t_end: float, dt: float, record_every: int = 4,
              fit_hi: float = 0.05) -> dict:
        """时域：拟合窗口按**绝对振幅**划（与 1-D 同纪律）。"""
        steps = int(t_end / dt)
        ts, amps = [], []
        for i in range(steps):
            self.step(dt)
            if i % record_every == 0:
                ts.append((i + 1) * dt)
                amps.append(self.mode_amp(*self.seed_modes))
        ts, amps = np.array(ts), np.array(amps)
        lo = 20.0 * self.seed
        m = (amps > lo) & (amps < fit_hi)
        gamma, r2 = float("nan"), float("nan")
        if int(m.sum()) >= 5:
            A = np.vstack([ts[m], np.ones(int(m.sum()))]).T
            sol, res, *_ = np.linalg.lstsq(A, np.log(amps[m]), rcond=None)
            ss = float(np.sum((np.log(amps[m]) - np.log(amps[m]).mean()) ** 2))
            gamma = float(sol[0])
            r2 = float(1.0 - res[0] / ss) if ss > 0 else float("nan")
        return {"gamma_measured": gamma, "fit_r2": r2, "amp_final": float(amps.max()),
                "fit_n": int(m.sum()), "t": ts, "amp": amps}


def two_stream_default_2d(mu: float = 0.0, L: float = 29.02, N: int = 256, Ny: int = 0,
                          seed_modes: tuple = (4, 0)) -> TwoFluid2D:
    return TwoFluid2D([Species(0.5, -1.0, 1.0, +1.0, mu),
                       Species(0.5, -1.0, 1.0, -1.0, mu)],
                      L=L, N=N, Ny=Ny, seed_modes=seed_modes)
