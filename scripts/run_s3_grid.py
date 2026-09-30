#!/usr/bin/env python3
"""S3 段 2：2-D 零假设基准表（G0 的执行层，2-D 版）。

分辨率由 S2 的 k 量化设计规则定：m ≳ 34 个波长、工作点放色散峰。
    L = 348.9（= 34·λ*），种子模 ix = 34 ⟹ k_seed = 34·2π/L = 0.6124 = k*（峰上，dγ/dk = 0）

网格（修订版）：μ ∈ {0, 1e−4, 1e−3, 1e−2, 1e−1} × 种子几何 ∈ {共线, 非共线, 宽谱}，
外加收敛复核（dt 减半 ×2 点、横向粗化 ×1 点）。

**两个诊断信号同时测**（这是 2-D 相对 1-D 的实质增量）：
  ① 真模幅 A_mode(t)：FFT 单模系数 → 实振幅（除以 N/2）
  ② 弦积分幅 A_chord(t)：先沿 y 平均（只留 ky = 0），再取 x 方向的模系数
     —— ② 正是「视线/弦积分诊断」测到的东西；两者拟合出的 γ 之差 = 2-D 诊断偏差。
     **弦积分与真值同（偏差 ≈ 0）只在共线几何成立** —— 那就是 1-D 的情形。

判决量：R_ci := γ_meas(μ)/γ_meas(μ=0) − 1（同装置同 dt ⟹ RK4 系统偏差相消），
真值 1/√(1−μ) − 1；等效 μ = 2·R_ci。ΔR_ci(几何) = R_ci(该几何) − R_ci(共线) ⟹ 2-D 系统偏差。

用法：python scripts/run_s3_grid.py [--procs 16] [--nsteps 5000] [--out artifacts/s3_grid]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gym_mu_frc.two_fluid_1d import cold_two_stream_roots          # noqa: E402
from gym_mu_frc.two_fluid_2d import two_stream_default_2d          # noqa: E402

L = 348.9           # m = 34 个波长（S2 的 k 量化设计规则）
IX = 34             # k_seed = 0.6124 = k*（色散峰）
IY_OBL = 17         # 斜模：k_y/k_x ≈ 0.5
DT = 0.01
SEED = 1e-4

GEOMS = {
    "共线": ((IX, 0),),
    "非共线": ((IX, IY_OBL), (IX, -IY_OBL)),
    "宽谱": ((IX, 3), (IX, 9), (IX, 17), (IX, -3), (IX, -9), (IX, -17)),
}
MUS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)


def gamma_linear(kx: float, mu: float = 0.0) -> float:
    """解析：ω_pe(μ) = 1/√(1−μ)（m → m(1−μ)）⟹ γ(μ) = γ(0)/√(1−μ)。"""
    wp_each = math.sqrt(0.5 / (1.0 - mu))
    r = cold_two_stream_roots(kx, [wp_each] * 2, [1.0, -1.0])
    return float(np.max(r.imag)) if r.size else 0.0


def fit_gamma(t: np.ndarray, a: np.ndarray, lo: float, hi: float) -> dict:
    """在显式振幅窗口 [lo, hi] 上拟合 ln A = γt + c，并做**自带有效性自检**：
    窗口前后两半各自拟合，两者之差就是"窗口是否落在纯指数段"的证据。
    窗口不合法时（非线性/数值污染）前后半会给出明显不同的 γ —— 直接写进表里，不靠事后追查。
    """
    m = (a > lo) & (a < hi)
    if int(m.sum()) < 10:
        return {"gamma": float("nan"), "r2": float("nan"), "n": int(m.sum()),
                "gamma_lo_half": float("nan"), "gamma_hi_half": float("nan"), "half_gap": float("nan")}

    def _fit(mask):
        A = np.vstack([t[mask], np.ones(int(mask.sum()))]).T
        sol, res, *_ = np.linalg.lstsq(A, np.log(a[mask]), rcond=None)
        ss = float(np.sum((np.log(a[mask]) - np.log(a[mask]).mean()) ** 2))
        return float(sol[0]), (float(1.0 - res[0] / ss) if ss > 0 else float("nan"))

    g, r2 = _fit(m)
    mm = (a > lo) & (a < np.sqrt(lo * hi))          # 下半（振幅靠前）
    mh = (a > np.sqrt(lo * hi)) & (a < hi)          # 上半
    gl = _fit(mm)[0] if int(mm.sum()) >= 5 else float("nan")
    gh = _fit(mh)[0] if int(mh.sum()) >= 5 else float("nan")
    gap = abs(gh - gl) / abs(g) if np.isfinite(gl) and np.isfinite(gh) else float("nan")
    return {"gamma": g, "r2": r2, "n": int(m.sum()),
            "gamma_lo_half": gl, "gamma_hi_half": gh, "half_gap": gap}


def one_run(args) -> dict:
    mu, gname, nsteps, n, ny, dt, tag, fit_lo, fit_hi = args
    t0 = time.perf_counter()
    seeds = GEOMS[gname]
    sim = two_stream_default_2d(mu=mu, L=L, N=n, Ny=ny, seed_modes=seeds[0], seed_list=seeds)
    probe = seeds[0]
    ts, a_mode, a_chord, hi_frac, harm = [], [], [], [], []
    for i in range(1, nsteps + 1):
        sim.step(dt)
        if i % 20 == 0:
            rho = sim.rho()
            pert = rho - rho.mean()
            rh = np.fft.fft2(pert)
            p = np.abs(rh) ** 2
            tot = float(p.sum())
            idx = np.arange(n)
            idx = np.minimum(idx, n - idx)
            mhi = idx > n // 4
            hi_frac.append(float(p[mhi, :].sum() / tot) if tot > 0 else 0.0)
            harm.append(float(np.sqrt(p[2 * probe[0] % n, 0] / p[probe[0], probe[1]]))
                        if p[probe[0], probe[1]] > 0 else 0.0)
            cs = pert.mean(axis=1)
            ts.append(i * dt)
            a_mode.append(2.0 * sim.mode_amp(*probe) / n)
            a_chord.append(2.0 * abs(np.fft.fft(cs)[probe[0]]) / n)
    ts = np.array(ts)
    a_mode = np.array(a_mode)
    a_chord = np.array(a_chord)
    fm = fit_gamma(ts, a_mode, fit_lo, fit_hi)
    fc = fit_gamma(ts, a_chord, fit_lo, fit_hi)
    finite = bool(np.all(np.isfinite(a_mode)))
    out = {
        "tag": tag, "mu": mu, "geom": gname, "nsteps": nsteps, "N": n, "Ny": ny, "dt": dt,
        "probe": list(probe), "n_seed_modes": len(seeds), "fit_window": [fit_lo, fit_hi],
        "finite": finite,
        "gamma_mode": fm, "gamma_chord": fc,
        "gamma_analytic_mu0": gamma_linear(probe[0] * 2 * np.pi / L, 0.0),
        "gamma_analytic_this_mu": gamma_linear(probe[0] * 2 * np.pi / L, mu),
        "high_k_frac_at_top": float(hi_frac[-1]) if hi_frac else float("nan"),
        "harmonic_ratio_at_top": float(harm[-1]) if harm else float("nan"),
        "new_mode_frac_end": sim.new_mode_fraction(seeds),
        "transverse_frac_end": sim.transverse_fraction(),
        "amp_mode_end": float(a_mode[-1]),
        "top_modes": sim.top_modes(6),
        "wall_s": time.perf_counter() - t0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--nsteps", type=int, default=5000)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--mus", default=",".join(f"{m:g}" for m in MUS),
                    help="逗号分隔的 μ 子集（冒烟用）")
    ap.add_argument("--geoms", default=",".join(GEOMS),
                    help="逗号分隔的几何子集（冒烟用）")
    ap.add_argument("--fit_lo", type=float, default=20.0 * SEED,
                    help="拟合窗口下沿（实振幅）；默认 20·seed")
    ap.add_argument("--fit_hi", type=float, default=0.05,
                    help="拟合窗口上沿（实振幅）；必须由「振幅分段 γ 曲线」实测确认在纯指数段内")
    ap.add_argument("--out", default="artifacts/s3_grid")
    args = ap.parse_args()
    mus = [float(x) for x in args.mus.split(",") if x.strip()]
    geoms = [g for g in args.geoms.split(",") if g.strip()]
    for g in geoms:
        if g not in GEOMS:
            print(f"未知几何 {g}；可选 {list(GEOMS)}")
            return 2
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    fit_lo, fit_hi = args.fit_lo, args.fit_hi
    jobs = []
    for mu in mus:
        for gname in geoms:
            jobs.append((mu, gname, args.nsteps, args.N, args.N, DT, f"{gname}_mu{mu:g}",
                         fit_lo, fit_hi))
    # 收敛复核：dt 减半（2 点）+ 横向粗化（1 点）
    jobs.append((0.0, "共线", args.nsteps, args.N, args.N, DT / 2, "收敛_dt半_共线_mu0", fit_lo, fit_hi))
    jobs.append((1e-2, "非共线", args.nsteps, args.N, args.N, DT / 2, "收敛_dt半_非共线_mu0.01", fit_lo, fit_hi))
    jobs.append((1e-2, "非共线", args.nsteps, args.N, args.N // 2, DT, "收敛_Ny半_非共线_mu0.01", fit_lo, fit_hi))

    print(f"任务 {len(jobs)} 个，{args.procs} 进程；每点 {args.nsteps} 步，N={args.N}², dt={DT}；"
          f"拟合窗口 [{fit_lo:g}, {fit_hi:g}]", flush=True)
    t0 = time.perf_counter()
    done = []
    with mp.Pool(args.procs) as pool:
        for r in pool.imap_unordered(one_run, jobs):
            done.append(r)
            (outdir / f"{r['tag']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str))
            gm = r["gamma_mode"]
            print(f"  [完成 {len(done)}/{len(jobs)}] {r['tag']:26s} "
                  f"γ_mode={gm['gamma']:.6f}(n={gm['n']},半差={gm['half_gap']:.1e}) "
                  f"γ_chord={r['gamma_chord']['gamma']:.6f} 高频={r['high_k_frac_at_top']:.1e} "
                  f"谐波={r['harmonic_ratio_at_top']:.1e} 有限={r['finite']} {r['wall_s']:.0f}s", flush=True)

    wall = time.perf_counter() - t0
    # 汇总：以 μ=0 共线为基准算 R_ci，再按几何看 2-D 偏差
    base = {}
    for r in done:
        if r["geom"] == "共线" and r["dt"] == DT and r["Ny"] == args.N:
            base[r["mu"]] = r
    print("\n=== 基准表（R_ci = γ_meas(μ)/γ_meas(μ=0) − 1；等效 μ = 2·R_ci）===", flush=True)
    hdr = f"{'几何':<8}{'μ':>8}{'γ_mode':>12}{'γ_chord':>12}{'R_ci(mode)':>12}{'等效μ':>10}{'真值μ':>9}{'ΔR_ci(2D)':>12}"
    print(hdr, flush=True)
    rows = []
    g0 = base.get(0.0, {}).get("gamma_mode", {}).get("gamma", float("nan"))
    for r in sorted(done, key=lambda x: (x["geom"], x["mu"])):
        if r["dt"] != DT or r["Ny"] != args.N:
            continue
        gm = r["gamma_mode"]["gamma"]
        gc = r["gamma_chord"]["gamma"]
        rci = gm / g0 - 1.0
        ref = base.get(r["mu"], {}).get("gamma_mode", {}).get("gamma", float("nan"))
        rci_col = ref / g0 - 1.0 if np.isfinite(ref) else float("nan")
        drci = rci - rci_col
        rows.append({"tag": r["tag"], "geom": r["geom"], "mu": r["mu"], "gamma_mode": gm,
                     "gamma_chord": gc, "R_ci_mode": rci, "equiv_mu": 2.0 * rci,
                     "true_mu": r["mu"], "dR_ci_2d": drci,
                     "new_mode_frac": r["new_mode_frac_end"]})
        print(f"{r['geom']:<8}{r['mu']:>8g}{gm:>12.6f}{gc:>12.6f}{rci:>12.3e}{2*rci:>10.3e}"
              f"{1.0/math.sqrt(1.0-r['mu'])-1.0:>9.3e}{drci:>12.3e}", flush=True)

    print("\n=== 收敛复核 ===", flush=True)
    for r in done:
        if r["dt"] != DT or r["Ny"] != args.N:
            print(f"  {r['tag']:26s} γ_mode={r['gamma_mode']['gamma']:.6f} "
                  f"(n={r['gamma_mode']['n']}) 新模={r['new_mode_frac_end']:.2e}", flush=True)

    (outdir / "summary.json").write_text(json.dumps(
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_s": wall, "L": L, "IX": IX,
         "nsteps": args.nsteps, "N": args.N, "dt": DT, "rows": rows,
         "raw": [{k: v for k, v in r.items() if k != "top_modes"} for r in done]},
        ensure_ascii=False, indent=2, default=str))
    print(f"\nRESULT: {len(done)}/{len(jobs)} 完成，总机时 {wall:.0f}s（{wall/60:.1f} 分钟）"
          f"⟹ {args.out}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
