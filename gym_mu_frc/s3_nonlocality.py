#!/usr/bin/env python3
"""S3 判决实验：局部操作能不能改变「离散拓扑量」？（自写 2-D 谱求解器）

背景（上游结论，ProjectionPhysics `theory-mu-topology.md`，77c794f）：
"改变净连接数必须非局部" ⟹ μ 是**离散配置标签**，不是连续旋钮。
本步执行该支的**判决实验**：在自写的 2-D 通量演化里区分三类操作 ——

  局部（不改拓扑）：局部速度扰动 / 光滑压缩（ψ 乘正的平滑因子）/ 局部加热
  全局但**平滑**（仍不改拓扑）：大尺度汇聚流（把两岛推到一起）
  拓扑改变通道：局部电阻率（重联式）

模型（**模型替身 stand-in**，口径说清）：
    ∂ψ/∂t + {φ, ψ} = η(x,y)·∇²ψ − ν₄·∇⁴ψ          {a,b} = ∂ₓa ∂_y b − ∂_y a ∂ₓb
  * 理想极限（η=0）：ψ 被平流 ⟹ 等值线拓扑（岛分合）**严格不变**
    （这是 2-D 理想平流的定理 ⟹ 本步的"局部/全局平滑不改拓扑"是**数值体检 + 离散化体检**，
      不是发现；真正的产出是 ②③④）
  * 拓扑改变只走电阻通道（重联式）

实测产出：
  ① 离散性：N_islands 只在电阻事件里**整数跳变**（2 → 1），不是连续漂移
  ② 触发一次拓扑事件所需的**欧姆耗散能量**（阈值代价，代码单位 + 量纲口径）
  ③ 配置表 v1：N_islands ⟹ (e, gross, μ)（用上游指数和/连接数账口径）
  ④ 证伪判定：若理想极限下任何操作改了 N_islands ⟹ 上游那一支被证伪（如实报）

第一版教训（已修，更正过两次）：
  · 真因：**∇⁴ 的符号写反** —— `lap(·, K4)` 返回的是 −∇⁴，再乘 −ν₄ 就变成 +ν₄∇⁴（**反扩散**）
    ⟹ 连零流基线都发散成 NaN，岛计数假跌成 1。现在算子拆成 laplace(∇²) / bilaplace(∇⁴) 消歧义。
  · 我一度把发散归因于"φ 取 sin·cos 是双曲驻点流 ⟹ 指数拉伸" —— 那个诊断**是错的**（已改口）。
  · 另加：每步有限性守卫（发散返回哨兵 −1，绝不把 NaN 读成"合并"）+ rfft2（快 3×）。
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "s3_nonlocality"
TWO_PI = 2.0 * math.pi


# ───────────────────────── 谱工具（实 FFT） ─────────────────────────

class Grid:
    def __init__(self, n: int = 128, lx: float = TWO_PI, ly: float = TWO_PI):
        self.n, self.lx, self.ly = n, lx, ly
        x = np.arange(n) * lx / n
        y = np.arange(n) * ly / n
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        kx = 2 * math.pi * np.fft.fftfreq(n, d=lx / n)
        ky = 2 * math.pi * np.fft.rfftfreq(n, d=ly / n)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K4 = self.K2 ** 2
        kxr = np.abs(self.KX) / (math.pi * n / lx)
        kyr = np.abs(self.KY) / (math.pi * n / ly)
        self.dealias = (kxr < 2 / 3) & (kyr < 2 / 3)

    def fwd(self, f):
        return np.fft.rfft2(f)

    def inv(self, F):
        return np.fft.irfft2(F, s=(self.n, self.n))

    def dx(self, f):
        return self.inv(1j * self.KX * self.fwd(f))

    def dy(self, f):
        return self.inv(1j * self.KY * self.fwd(f))

    def laplace(self, f):
        """∇²f（谱：−K²）—— 注意与下一行区分，别把 ∇⁴ 的符号搞反（第一版就栽在这儿）。"""
        return self.inv(-self.K2 * self.fwd(f))

    def bilaplace(self, f):
        """∇⁴f（谱：+K⁴）。"""
        return self.inv(self.K4 * self.fwd(f))

    def trunc(self, f):
        return self.inv(self.dealias * self.fwd(f))


def jacobian(g: Grid, dphi_dx, dphi_dy, psi):
    """{φ, ψ} = ∂ₓφ ∂_yψ − ∂_yφ ∂ₓψ；φ 的导数**预存**（静态场 ⟹ 省一半变换）。"""
    return g.trunc(dphi_dx * g.dy(psi) - dphi_dy * g.dx(psi))


# ───────────────────────── 构型与流 ─────────────────────────

def blobs(g: Grid, sep: float = 2.0, w: float = 0.40) -> np.ndarray:
    """两个同号高斯通量岛（岛心沿 y 分开 sep）—— 分合与否 = 离散拓扑量。"""
    def bump(cy):
        d2 = (((g.X + math.pi) % TWO_PI - math.pi) ** 2 +
              ((g.Y - cy + math.pi) % TWO_PI - math.pi) ** 2)
        return np.exp(-d2 / (2 * w ** 2))
    return bump(math.pi - sep / 2) + bump(math.pi + sep / 2)


def convergent_flow(g: Grid, amp: float = 0.5) -> np.ndarray:
    """大尺度汇聚流（把两岛沿 y 往中间推）：φ = amp·sin(y)/k —— 线性剪切式，无指数拉伸。"""
    return amp * np.sin(g.Y - math.pi)


def local_flow(g: Grid, amp: float = 0.3, w: float = 0.6) -> np.ndarray:
    """局部旋转驱动（高斯包络内的小尺度涡）。"""
    env = np.exp(-(((g.X + math.pi) % TWO_PI - math.pi) ** 2 +
                   ((g.Y - math.pi + math.pi) % TWO_PI - math.pi) ** 2) / (2 * w ** 2))
    return amp * env * np.cos(g.X) * np.sin(g.Y)


# ───────────────────────── 离散拓扑量 ─────────────────────────

def _idx(g: Grid, t: float, axis: str) -> int:
    return int(round((t % TWO_PI) / (g.ly if axis == "y" else g.lx) * g.n)) % g.n


def bridge_ratio(g: Grid, psi: np.ndarray, sep: float = 2.0, x0: float = 0.0) -> float:
    """辅助连续量：中线（x = x0）上鞍部 / 两峰较小者。

    **第一版仪器教训**：当时鞍部写成 `psi[:, j].min()`（扫遍整个 x）⟹ 取到的是背景 0
    ⟹ 比值恒为 1.0、计数恒为 2（门永远绿）—— 那是坏仪器。现在只在**两岛心之间的那条线段**上取。
    """
    if not np.all(np.isfinite(psi)):
        return float("nan")
    i0 = _idx(g, x0, "x")
    j0, j1 = _idx(g, math.pi - sep / 2, "y"), _idx(g, math.pi + sep / 2, "y")
    # 沿较短弧列出两峰之间的 y 索引
    n = g.n
    fwd = [(j0 + k) % n for k in range((j1 - j0) % n + 1)]
    bwd = [(j0 - k) % n for k in range((j0 - j1) % n + 1)]
    js = fwd if len(fwd) <= len(bwd) else bwd
    peak = min(max(psi[i0, j] for j in [j0]), max(psi[i0, j] for j in [j1]))
    bridge = min(psi[i0, j] for j in js)
    if peak <= 0:
        return float("nan")
    return float(bridge / peak)


def island_count(g: Grid, psi: np.ndarray, sep: float = 2.0, thr: float = 0.05) -> int:
    """**主判据**：块粗化后数局部极大（真·离散计数，不受"被抹开"干扰）。

    哨兵：数值发散 ⟹ −1（绝不把 NaN 读成"合并"）。
    """
    return island_count_generic(g, psi, block=8, rel=0.05)


def island_count_generic(g: Grid, psi: np.ndarray, block: int = 8, rel: float = 0.05) -> int:
    """块粗化 + 局部极大计数（同一块集合内取"≥ 全邻居"者）。"""
    if not np.all(np.isfinite(psi)):
        return -1
    n, b = g.n, block
    c = psi.reshape(n // b, b, n // b, b).mean(axis=(1, 3))
    mx = np.ones_like(c, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            mx &= c >= np.roll(np.roll(c, di, 0), dj, 1)
    rng = c.max() - c.min()
    if rng <= 0:
        return 0
    return int(np.sum(mx & (c > c.min() + rel * rng)))


def connected_components(mask: np.ndarray, periodic: bool = True) -> int:
    """4-连通分量数（并查集）。**必须带周期连通**：岛心落在 x=0 上时掩码在网格里被切成两半，
    不带周期会把它数成 2 个（本步第二个仪器 bug，见 selftest 的"跨边界"用例）。
    """
    n, m = mask.shape
    parent = list(range(n * m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(n):
        for j in range(m):
            if not mask[i, j]:
                continue
            k = i * m + j
            if i + 1 < n and mask[i + 1, j]:
                union(k, (i + 1) * m + j)
            if j + 1 < m and mask[i, j + 1]:
                union(k, i * m + j + 1)
    if periodic:
        for i in range(n):                      # 上下边贴合
            if mask[i, 0] and mask[i, m - 1]:
                union(i * m, i * m + m - 1)
        for j in range(m):                      # 左右边贴合
            if mask[0, j] and mask[n - 1, j]:
                union(j, (n - 1) * m + j)
    roots = {find(i * m + j) for i in range(n) for j in range(m) if mask[i, j]}
    return len(roots)


def island_components(g: Grid, psi: np.ndarray, thr: float) -> int:
    """**真拓扑不变量**：固定阈值下超水平集 {ψ > thr} 的连通分量数（带周期连通）。

    理想平流保等值线拓扑 ⟹ 只要没有临界点越过 thr，这个计数不变；
    电阻事件让鞍部升高越过 thr ⟹ 分量合并 ⟹ 计数整数跳变。发散返回 −1。
    """
    if not np.all(np.isfinite(psi)):
        return -1
    return connected_components(psi > thr, periodic=True)


def selftest_instrument(g: Grid) -> dict:
    """**仪器自检**（合成数据，防"门永远绿"）：分量计数必须能区分分离/重合，且抗抹开。

    第一版仪器坏在有"扫遍全场取最小值"的鞍部取样 ⟹ 计数恒为 2、门永远绿；
    第二版用"局部极大计数"，但理想平流**能拉出新的局部极大** ⟹ 不是拓扑不变量。
    现在主判据 = **固定阈值连通分量数**（真不变量），局部极大计数留作交叉核对。
    """
    far = blobs(g, sep=2.4, w=0.40)
    near = blobs(g, sep=0.25, w=0.40)
    smeared = far.copy()
    for _ in range(1):
        smeared = 0.5 * smeared + 0.125 * (np.roll(smeared, 1, 0) + np.roll(smeared, -1, 0)
                                          + np.roll(smeared, 1, 1) + np.roll(smeared, -1, 1))
    thr = 0.5 * float(np.min([far[:, _idx(g, math.pi - 1.2, "y")].max(),
                              far[:, _idx(g, math.pi + 1.2, "y")].max()]))
    res = {"阈值": round(thr, 4),
           "分离(sep=2.4)": island_components(g, far, thr),
           "重合(sep=0.25)": island_components(g, near, thr),
           "抹开 1 次(sep=2.4)": island_components(g, smeared, thr),
           "交叉核对（局部极大计数）分离/重合": (island_count_generic(g, far),
                                                island_count_generic(g, near)),
           "桥接比 分离/重合": (round(bridge_ratio(g, far, 2.4), 4),
                           round(bridge_ratio(g, near, 0.25), 4))}
    # 跨周期边界：岛心在 x=0 上（本构型的实际情况）—— 不带周期连通会数成 2
    cross = blobs(g, sep=2.4)
    res["跨边界（岛心在 x=0）"] = island_components(g, cross, thr)
    m = np.zeros((5, 5), bool)
    m[0, 0] = m[0, 2] = m[3, 3] = m[3, 4] = m[4, 3] = True
    res["并查集单测（应为 3）"] = connected_components(m, periodic=False)
    res["自检通过"] = (res["分离(sep=2.4)"] == 2 and res["重合(sep=0.25)"] == 1
                     and res["抹开 1 次(sep=2.4)"] == 2
                     and res["跨边界（岛心在 x=0）"] == 2
                     and res["并查集单测（应为 3）"] == 3)
    return res


# ───────────────────────── 演化 ─────────────────────────

class Flux2D:
    def __init__(self, g: Grid, psi0, dphi_dx, dphi_dy, eta=None, nu4: float = 1e-5):
        self.g, self.psi = g, psi0.copy()
        self.dphi_dx, self.dphi_dy = dphi_dx, dphi_dy
        self.eta = np.zeros_like(psi0) if eta is None else eta
        self.nu4 = nu4
        self.ohmic = 0.0
        self.diverged = False

    def rhs(self, psi):
        adv = jacobian(self.g, self.dphi_dx, self.dphi_dy, psi)   # {φ, ψ}
        diff = self.eta * self.g.laplace(psi)                     # +η∇²ψ（扩散/重联）
        hyper = -self.nu4 * self.g.bilaplace(psi)                 # −ν₄∇⁴ψ（超黏，耗散）
        return -adv + diff + hyper

    def step(self, dt_nom: float, max_change: float = 0.03) -> float:
        """RK4 + 自适应步长（合并生成薄电流片，固定 dt 会炸）+ 每步谱截断防小尺度堆积。"""
        k1 = self.rhs(self.psi)
        scale = float(np.max(np.abs(k1)))
        dt = dt_nom if scale == 0 else min(dt_nom, max_change / scale)
        k2 = self.rhs(self.psi + 0.5 * dt * k1)
        k3 = self.rhs(self.psi + 0.5 * dt * k2)
        k4 = self.rhs(self.psi + dt * k3)
        self.psi = self.psi + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.psi = self.g.trunc(self.psi)
        J = self.g.laplace(self.psi)
        self.ohmic += float(np.sum(self.eta * J ** 2) * (self.g.lx * self.g.ly / self.g.n ** 2) * dt)
        if not np.all(np.isfinite(self.psi)):
            self.diverged = True
        return dt


def initial_threshold(g: Grid, psi0: np.ndarray, sep: float) -> float:
    """固定阈值：两岛心处峰值的较小者的一半（岛区间的水平集阈值）。"""
    i0 = _idx(g, 0.0, "x")
    peaks = [psi0[i0, _idx(g, math.pi - sep / 2, "y")].max(),
             psi0[i0, _idx(g, math.pi + sep / 2, "y")].max()]
    return 0.5 * float(min(peaks))


def snapshot(g: Grid, psi: np.ndarray, thr: float, sep: float) -> dict:
    rng = float(np.max(psi) - np.min(psi)) if np.all(np.isfinite(psi)) else float("nan")
    return {"N_分量": island_components(g, psi, thr),
            "N_局部极大（交叉核对）": island_count_generic(g, psi),
            "桥接比": round(bridge_ratio(g, psi, sep), 6) if np.all(np.isfinite(psi)) else float("nan"),
            "峰值/阈值": round(float(np.max(psi)) / thr, 4) if np.isfinite(rng) and thr > 0 else float("nan")}


def run(g: Grid, psi0, phi, eta=None, t_end=8.0, dt=0.004, nu4=1e-5,
        record_every=25, sep=2.0, thr=None):
    dpx, dpy = g.dx(phi), g.dy(phi)
    sim = Flux2D(g, psi0, dpx, dpy, eta=eta, nu4=nu4)
    thr = initial_threshold(g, psi0, sep) if thr is None else thr
    hist = [{"t": 0.0, **snapshot(g, sim.psi, thr, sep)}]
    t, i, dt_min = 0.0, 0, dt
    while t < t_end - 1e-12:
        d = sim.step(dt)
        dt_min = min(dt_min, d)
        t += d
        i += 1
        if sim.diverged:
            hist.append({"t": round(t, 4), "N_分量": -1, "N_局部极大（交叉核对）": -1,
                         "桥接比": float("nan"), "峰值/阈值": float("nan"), "发散": True})
            break
        if i % record_every == 0 or t >= t_end - 1e-12:
            hist.append({"t": round(t, 4), **snapshot(g, sim.psi, thr, sep)})
    sim.dt_min, sim.nsteps, sim.thr = dt_min, i, thr
    return sim, hist


# ───────────────────────── 三类操作 ─────────────────────────

def ops_test(g: Grid, sep: float = 2.0, t_end: float = 8.0) -> dict:
    psi0 = blobs(g, sep=sep)
    thr = initial_threshold(g, psi0, sep)
    n0 = island_components(g, psi0, thr)
    rows = []

    def add(name, phi, psi_in=None):
        sim, hist = run(g, psi_in if psi_in is not None else psi0, phi, t_end=t_end, sep=sep, thr=thr)
        rows.append({"操作": name, "N_分量 初": n0, "N_分量 末": hist[-1]["N_分量"],
                     "N_局部极大 末": hist[-1]["N_局部极大（交叉核对）"],
                     "桥接比 初": round(bridge_ratio(g, psi0, sep), 4),
                     "桥接比 末": hist[-1]["桥接比"],
                     "峰值/阈值 末": hist[-1]["峰值/阈值"],
                     "发散": sim.diverged, "欧姆耗散": sim.ohmic,
                     "步数": sim.nsteps, "最小有效 dt": sim.dt_min})

    zero = np.zeros_like(psi0)
    add("基线（无流）", zero)
    add("L1 局部旋转驱动（改 φ）", local_flow(g))
    add("L2 光滑压缩（ψ × 正因子）", zero,
        psi_in=psi0 * (1.0 + 0.25 * np.exp(-(((g.Y - math.pi) + math.pi) % TWO_PI - math.pi) ** 2 / (2 * 1.0 ** 2))))
    add("L3 局部加热（φ 局部时窗）", 0.5 * local_flow(g, w=0.9))
    add("G0 大尺度剪切流（平滑，仍理想）", convergent_flow(g))
    return {"N 初": n0, "阈值": round(thr, 4), "行": rows,
            "违例（理想操作改了 N 分量）": sum(1 for r in rows if r["N_分量 末"] not in (n0, -1)),
            "发散行数": sum(1 for r in rows if r["发散"])}


def reconnection_event(g: Grid, sep: float = 2.0, eta0: float = 5e-2,
                       t_end: float = 15.0) -> dict:
    """拓扑改变通道：**无流**、只在两岛之间开电阻率（窄带）⟹ 岛合并（N_分量: 2 → 1）。

    口径：这里的电阻扩散项 η∇²ψ 就是 2-D 里的重联通道（改变等值线拓扑的项）。
    """
    psi0 = blobs(g, sep=sep)
    thr = initial_threshold(g, psi0, sep)
    eta = eta0 * np.exp(-(((g.Y - math.pi + math.pi) % TWO_PI - math.pi) ** 2) / (2 * 0.6 ** 2))
    sim, hist = run(g, psi0, np.zeros_like(psi0), eta=eta, t_end=t_end, sep=sep, thr=thr)
    t_event, E_event, prev = None, None, hist[0]["N_分量"]
    for r in hist[1:]:
        if r["N_分量"] != prev and r["N_分量"] > 0:
            t_event, E_event = r["t"], r["欧姆耗散"]
            break
        prev = r["N_分量"]
    return {"N 初": hist[0]["N_分量"], "N 末": hist[-1]["N_分量"], "发散": sim.diverged,
            "阈值": round(thr, 4),
            "时刻序列": [r["t"] for r in hist[:10]], "N 序列": [r["N_分量"] for r in hist[:12]],
            "桥接比序列": [r["桥接比"] for r in hist[:10]],
            "峰值/阈值序列": [r["峰值/阈值"] for r in hist[:10]],
            "首次跳变时刻": t_event, "跳变时累积欧姆耗散（代码单位）": E_event,
            "总欧姆耗散（代码单位）": sim.ohmic,
            "步数": sim.nsteps, "最小有效 dt": sim.dt_min}


def config_table() -> dict:
    """配置表 v1：离散拓扑量 ⟹ 上游 (gross, net, μ) 账。**模型替身口径**。"""
    rows = []
    for name, gross, net in (("两岛（分离）", 2, 2), ("一岛（已合并）", 1, 1)):
        rows.append({"配置": name, "gross": gross, "net": net, "μ": 1.0 - net / gross,
                     "e（指数和口径）": gross})
    for name, gross, net in (("两岛反向（一正一负）", 2, 0),):
        rows.append({"配置": name, "gross": gross, "net": net, "μ": 1.0 - net / gross,
                     "e（指数和口径）": gross})
    return {"行": rows,
            "口径": "模型替身：岛数当连接单位；示 μ 随配置**离散跳变**；真装置映射待定",
            "诚实": "带符号那一行（一正一负 ⟹ μ = 1）是**预测的下一档**，本步尚未实现该构型"}


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    print("=== S3 判决实验：局部/全局平滑 vs 离散拓扑量 ===")
    g = Grid(128)
    out = {}

    st = selftest_instrument(g)
    out["S3-0_仪器自检"] = st
    print(f"S3-0 仪器自检：分离 {st['分离(sep=2.4)']}、重合 {st['重合(sep=0.25)']}、"
          f"抹开一次 {st['抹开 1 次(sep=2.4)']}、跨边界 {st['跨边界（岛心在 x=0）']}、"
          f"并查集单测 {st['并查集单测（应为 3）']} ⟹ 通过 = {st['自检通过']}")

    o = ops_test(g)
    out["S3-1_理想操作"] = o
    print(f"S3-1 理想极限（η=0）：N 分量 初 = {o['N 初']}（阈值 {o['阈值']}）")
    for r in o["行"]:
        print(f"      {r['操作']:28s} N: {r['N_分量 初']} → {r['N_分量 末']}"
              f"  桥接比 末 {r['桥接比 末']}  峰值/阈值 {r['峰值/阈值 末']}"
              + ("  ⚠发散" if r["发散"] else ""))

    ev = reconnection_event(g)
    out["S3-2_重联事件"] = ev
    print(f"S3-2 重联（无流 + 窄带电阻率）：N 分量 {ev['N 初']} → {ev['N 末']}；"
          f"首次跳变 t = {ev['首次跳变时刻']}，跳变时累积欧姆耗散 = {ev['跳变时累积欧姆耗散（代码单位）']}")

    tbl = config_table()
    out["S3-3_配置表v1"] = tbl
    print("S3-3 配置表 v1：" + "；".join(f"{r['配置']} ⟹ μ={r['μ']}" for r in tbl["行"]))
    print(f"S3-3b 仪器：主判据=固定阈值连通分量数；交叉核对=局部极大计数")

    dim = {"口径": ("代码单位 → 装置：以 B=12.2T、a=0.5m、R=2.111m 的场能密度作单位换算；"
                    "**量级口径，非推导**；上游解析估计区间 1e6–1e9 J"),
           "场能密度 [J/m³]": 12.2 ** 2 / (2 * 4e-7 * math.pi),
           "上游估计区间 [J]": [1e6, 1e9]}
    out["S3-4_阈值代价口径"] = dim
    print(f"S3-4 阈值代价：代码单位 {ev['跳变时累积欧姆耗散（代码单位）']}；装置映射见 report")

    todo = [
        "带符号构型（一正一负两岛 ⟹ μ = 1）尚未实现：那是 μ 跳变的**非零**档",
        "3-D 才是真链接数（本步 2-D 岛合并是**模型替身**）",
        "阈值代价的量纲映射需要一句可辩护的归一化（当前是量级口径）",
        "电阻率大小扫描（找合并的 η 阈值）未做",
    ]
    out["TODO"] = todo
    for t in todo:
        print("TODO: " + t)

    (ART / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    (ART / "summary.txt").write_text("\n".join([
        "S3 判决实验：局部/全局平滑操作 vs 离散拓扑量（2-D 模型替身）",
        f"仪器自检：{st['自检通过']}（分离 {st['分离(sep=2.4)']} / 重合 {st['重合(sep=0.25)']} / "
        f"抹开 {st['抹开 1 次(sep=2.4)']}）",
        f"理想极限：N 分量 初 {o['N 初']} → 各操作末态，违例 = {o['违例（理想操作改了 N 分量）']}，"
        f"发散行 {o['发散行数']}",
        f"重联事件（局部电阻率带）：N {ev['N 初']} → {ev['N 末']}，首次跳变 t = {ev['首次跳变时刻']}，"
        f"累积欧姆耗散 {ev['跳变时累积欧姆耗散（代码单位）']}",
        f"配置表 v1：{len(tbl['行'])} 行（模型替身口径；含预测档 μ=1）",
    ]) + "\n")
    print(f"产物 → {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
