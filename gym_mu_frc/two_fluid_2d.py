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
    seed_modes: tuple = (4, 0)      # (ix, iy) 探测模（也是默认种子模）；iy = 0 ⟹ 纯 ky = 0
    seed_list: tuple | None = None  # 多种子（非共线必须）：((ix,iy), ...)；None ⟹ 只用 seed_modes
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

        # 初值：种子模的本征矢沿各自 k_x，y 方向按 iy 给定。
        # **非共线多模必须用 seed_list**：单模共线（如只有 (4,0)）时，非线性卷积不会造出
        # ky ≠ 0 分量（实测污染恒为 0.0e+00）⟹ 单模 2-D 实验是空的，横向能量转移测不到。
        modes = self.seed_list if self.seed_list is not None else (self.seed_modes,)
        nn = [np.full((self.N, self.Ny), s.n0) for s in self.species]
        vxx = [np.full((self.N, self.Ny), s.v0) for s in self.species]
        vyy = [np.zeros((self.N, self.Ny)) for s in self.species]
        for (ix0, iy0) in modes:
            k_here = ix0 * 2.0 * np.pi / self.L
            u = self._eig_at(k_here)                       # 该模自己的 k_x 上的本征矢
            phase = np.exp((1j * ix0 * 2.0 * np.pi / self.L) * self.x2)[:, None] \
                * np.exp((1j * iy0 * 2.0 * np.pi / self.L) * self.y)[None, :]
            for i, (_, us_n, us_v) in enumerate(zip(self.species, u[0::2], u[1::2])):
                nn[i] = nn[i] + self.seed * np.real(us_n * phase)
                vxx[i] = vxx[i] + self.seed * np.real(us_v * phase)
        self.n, self.vx, self.vy = nn, vxx, vyy

    # ---- 谱求导（2/3 去混叠）----
    def _eig_at(self, k: float) -> np.ndarray:
        """给定 k 上的不稳定本征矢（父类只支持在 self.k_seed 上取，多种子需要任意 k）。"""
        w, V = np.linalg.eig(self.linear_matrix(k))
        return V[:, int(np.argmax(w.real))]

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
        """模幅，**按 Ny 归一**，使 y 均匀时与 1-D 的 `|fft(rho)[mode]|` 逐位相等。

        为什么必须除 Ny：`fft2` 对 y 均匀场给出的系数是 `fft` 的 Ny 倍（卷积可分离），
        不除的话 S2 那套按 1-D 约定标定的拟合窗口阈值 [20·seed, 0.05] 会整体落空
        （实测症状：窗口点 = 0，拟合返回 nan）—— 归一化不一致会被误读成"没有增长"。
        """
        return float(abs(np.fft.fft2(self.rho())[mode, mode_y])) / self.Ny

    def new_mode_fraction(self, seeded) -> float:
        """**种子之外**的能量占比 = 1 − Σ_{种子模}|ρ̂|² / Σ_{全部非零模}|ρ̂|²。

        为什么不能只看"横向占比"：若种子本身就是 ky≠0 的模（非共线实验必然如此），
        横向占比从 t=0 就 ≈ 1，**把种子自己算成转移**了 —— 那不是转移的证据。
        要证明能量流进了新模，必须把种子集合排除在外：本函数 t=0 应为 0，随非线性增长。
        """
        rh = np.fft.fft2(self.rho()).copy()
        rh[0, 0] = 0.0
        tot = float(np.sum(np.abs(rh) ** 2))
        if tot == 0.0:
            return 0.0
        seed_e = 0.0
        for (ix, iy) in seeded:
            # **必须把共轭伙伴一起减掉**：实场的 ρ̂ 满足 ρ̂[-i,-j] = conj(ρ̂[i,j])，
            # 种一个模实际占两个（±k）。只减一个的话 t=0 就会给出 0.5 的假基线。
            for (a, b) in ((ix, iy), ((-ix) % self.N, (-iy) % self.Ny)):
                seed_e += float(abs(rh[a, b]) ** 2)
        return float(1.0 - seed_e / tot)

    def top_modes(self, k: int = 8) -> list[tuple]:
        """按能量排序的前 k 个模（诊断用：看非线性把能量送进了哪些模）。"""
        rh = np.fft.fft2(self.rho()).copy()
        rh[0, 0] = 0.0
        p = np.abs(rh) ** 2
        idx = np.dstack(np.unravel_index(np.argsort(p, axis=None)[::-1][:k], p.shape))[0]
        return [(int(i), int(j), float(p[i, j])) for i, j in idx]

    def transverse_fraction(self) -> float:
        """横向能量占比 = Σ_{k_y ≠ 0} |ρ̂|² / Σ_{全部非零模} |ρ̂|²（去掉均值模）。

        这是修订后 S3 的**主观测量**：单模共线初值下它恒为 0（非线性不造 ky≠0 分量），
        只有**非共线多种子**才会让它长起来 —— 横向能量转移/丝化的出现与否，看这一条。
        """
        rh = np.fft.fft2(self.rho())
        rh = rh.copy()
        rh[0, 0] = 0.0                                  # 去均值模
        tot = float(np.sum(np.abs(rh) ** 2))
        if tot == 0.0:
            return 0.0
        keep = np.zeros_like(rh)
        keep[:, 0] = rh[:, 0]                           # 只留 ky = 0 的模
        if self.Ny % 2 == 0:
            keep[:, self.Ny // 2] = rh[:, self.Ny // 2]  # Nyquist 也属"共线"
        return float(1.0 - np.sum(np.abs(keep) ** 2) / tot)

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
                          seed_modes: tuple = (4, 0), seed_list: tuple | None = None) -> TwoFluid2D:
    return TwoFluid2D([Species(0.5, -1.0, 1.0, +1.0, mu),
                       Species(0.5, -1.0, 1.0, -1.0, mu)],
                      L=L, N=N, Ny=Ny, seed_modes=seed_modes, seed_list=seed_list)
