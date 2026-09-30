#!/usr/bin/env python3
"""2-D 谱求解器的吞吐探针：单进程 vs 多进程农场，并给出参数扫描的时间预算。

为什么用**进程**而不是线程：numpy 的 FFT 会释放 GIL，但 BLAS/FFT 是内存带宽受限的，
多进程能拿到的加速远小于核数 —— 这个脚本就是用来量"到底能拿到几倍"的，不靠估算。

单元成本 = 每步 4 次 FFT2(Poisson) + 4 次 FFT2(梯度)（RK4 一步的最小量）。
真实 2-D 两流体另有 4–6 个场与非线性项 ⟹ 把它当成**下界**看。

用法：
    python scripts/probe_2d_throughput.py            # 本机或远端通用
    python scripts/probe_2d_throughput.py --quick    # 只跑 N=256
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np


def run(N: int = 256, n_steps: int = 20, seed: int = 0) -> float:
    """返回每步秒数（含 3 步预热，不计入）。"""
    rng = np.random.default_rng(seed)
    k = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    rho = rng.standard_normal((N, N))

    def poisson(r):
        return np.real(np.fft.ifft2(np.fft.fft2(r) / K2))

    def grad(f):
        fh = np.fft.fft2(f)
        return (np.real(np.fft.ifft2(1j * KX * fh)), np.real(np.fft.ifft2(1j * KY * fh)))

    for _ in range(3):
        phi = poisson(rho)
        grad(phi)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        for _ in range(4):
            phi = poisson(rho)
            gx, gy = grad(phi)
            rho = rho - 0.01 * (gx + gy)
    return (time.perf_counter() - t0) / n_steps


def _worker(a):
    return run(*a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ncpu", type=int, default=mp.cpu_count())
    args = ap.parse_args()
    ncpu = args.ncpu
    print(f"核数 {ncpu}；单元成本 = 每步 4×FFT2(Poisson) + 4×FFT2(梯度)")

    sizes = (256,) if args.quick else (256, 512, 1024)
    print("\n=== 单进程 ===")
    single = {}
    for N in sizes:
        dt = run(N)
        single[N] = 1.0 / dt                       # 步/s
        print(f"  N={N:5d}  {dt*1e3:8.2f} ms/步   2000 步单次运行 {dt*2000:8.1f} s")

    print("\n=== 多进程农场（每进程 20 步；加速 = 总吞吐/单进程吞吐）===")
    Ks = [k for k in (1, 4, 8, 16, 32, ncpu) if k <= ncpu]
    Ks = sorted(set(Ks))
    best = {}
    for N in sizes:
        print(f"  -- N={N} (单进程 {single[N]:.1f} 步/s) --")
        for K in Ks:
            n_steps = 20 if N <= 512 else 6
            t0 = time.perf_counter()
            with mp.Pool(min(K, K)) as p:
                res = p.map(_worker, [(N, n_steps, i) for i in range(K)])
            wall = time.perf_counter() - t0
            agg = K * n_steps / wall                # 总吞吐 步/s
            sp = agg / single[N]                    # 加速比（同口径：每秒步数之比）
            best[N] = max(best.get(N, 0.0), agg)
            print(f"    K={K:2d}  wall {wall:7.2f}s  单进程 {np.mean(res)*1e3:8.2f} ms/步  "
                  f"总吞吐 {agg:7.1f} 步/s  加速 {sp:5.2f}x")

    print("\n=== 参数扫描预算（按各 N 的最佳总吞吐，27 点 × 2000 步）===")
    for N in sizes:
        tot_steps = 27 * 2000
        print(f"  N={N:5d}  最佳总吞吐 {best[N]:7.1f} 步/s  ⟹ 27 点扫描 {tot_steps/best[N]/60:6.1f} 分钟   "
              f"单点 {2000/single[N]/60:5.2f} 分钟")
    print("\n注：真实 2-D 两流体场数更多 ⟹ 上表是**下界**；GPU（cupy）未测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
