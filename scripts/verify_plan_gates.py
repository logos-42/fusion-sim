#!/usr/bin/env python3
"""S0/S1 仿真门禁 —— 双流系统（hushfusion G0 执行层）。

门设计原则（诚实门）：
  ① 每条断言量**被验对象本身**（增长率先从时域序列量出来，再与解析比）；
  ② 阴性对照**注入本身也要合格**（μ=0 必须逐位退化为标准两流）；
  ③ 负结果/待办**如实登记**，不写成"通过"（S2 的四项判决量标 TODO）。

跑法：make plan   （或 .venv-sim/bin/python scripts/verify_plan_gates.py）
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gym_mu_frc.two_fluid_1d import (  # noqa: E402
    Species, TwoFluid1D, cold_two_stream_roots, closed_form_gamma, closed_form_kstar,
    ion_stream_constraint, max_growth_rate, resolution_table, two_stream_default,
)

RESULTS: list[dict] = []
HUSH = pathlib.Path("/Users/apple/Downloads/hushfusion")


def check(name: str, ok: bool, detail=None) -> bool:
    RESULTS.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail is not None else ""))
    return bool(ok)


def main() -> int:
    t0 = time.time()
    print("=== S0/S1 门禁：1-D 冷两流（双反向流）+ μ 可关闭 ===")
    out = {}

    # ── T0：两条独立路线的谱必须同一（差 e^{∓iωt} 约定）──
    sim = two_stream_default()
    k_seed = sim.k_seed
    w4 = np.linalg.eigvals(sim.linear_matrix())
    rd = cold_two_stream_roots(k_seed, [math.sqrt(0.5), math.sqrt(0.5)], [1.0, -1.0])
    d_max = max(float(np.min(np.abs(w4 - x))) for x in 1j * rd)   # 1j·rd = 4×4 约定
    check("T0: 4×4 线性化谱 == 介电函数根（同谱，差 i 因子约定）", d_max < 1e-9,
          {"最大逐根距离": d_max, "k": k_seed})
    out["T0"] = {"max_root_distance": d_max}

    # ── T1：闭式基准（手推）与数值扫 k 一致 ──
    k_star, g_max = max_growth_rate([math.sqrt(0.5)] * 2, [1.0, -1.0])
    k_cf, g_cf = closed_form_kstar(1.0, 1.0), closed_form_gamma(1.0)
    e_k, e_g = abs(k_star - k_cf) / k_cf, abs(g_max - g_cf) / g_cf
    check("T1: 闭式 γ_max=ω_p/(2√2)、k*v0=√(3/8)ω_p 与数值扫 k 一致",
          e_k < 1e-6 and e_g < 1e-9, {"k* 相对误差": e_k, "γ 相对误差": e_g,
                                      "k*": k_star, "γ_max": g_max})
    out["T1"] = {"k_star": k_star, "gamma_max": g_max, "rel_err_k": e_k, "rel_err_gamma": e_g}

    # ── T2：时域 vs 解析（本征矢初值），且 dt 减半必须不动 ──
    g_an, _ = sim.growth_analytic()
    runs = {}
    for dt in (0.02, 0.01):
        s2 = two_stream_default()
        r = s2.run(t_end=30.0, dt=dt, record_every=4, fit_hi=0.05)
        runs[dt] = r
    rel = abs(runs[0.01]["gamma_measured"] - g_an) / g_an
    dt_drift = abs(runs[0.01]["gamma_measured"] - runs[0.02]["gamma_measured"]) / g_an
    check("T2: 时域谱方法复现解析增长率（含 dt 收敛）",
          rel < 5e-3 and dt_drift < 1e-4 and runs[0.01]["fit_r2"] > 0.999,
          {"解析 γ": g_an, "时域 γ(dt=0.01)": runs[0.01]["gamma_measured"],
           "相对误差": rel, "dt 漂移": dt_drift, "R²": runs[0.01]["fit_r2"],
           "拟合点数": runs[0.01]["fit_n"]})
    out["T2"] = {"anal": g_an, "num": runs[0.01]["gamma_measured"], "rel": rel,
                 "dt_drift": dt_drift, "r2": runs[0.01]["fit_r2"]}

    # ── T3：μ 标度定理（全物种同倍重标 ⟹ 比值不变；绝对速率 ∝ 1/√(1−μ)）──
    L0 = two_stream_default().L
    rows = []
    for mu in (0.0, 1e-2, 1e-1):
        sm = TwoFluid1D([Species(0.5, -1.0, 1.0, +1.0, mu), Species(0.5, -1.0, 1.0, -1.0, mu)],
                        L=L0 * math.sqrt(1.0 - mu), N=512)     # 固定 k v0/ω_p ⟹ 调 L
        g_a, _ = sm.growth_analytic()
        wp = 1.0 / math.sqrt(1.0 - mu)
        r = sm.run(t_end=30.0, dt=0.01, record_every=4, fit_hi=0.05)
        rows.append({"mu": mu, "γ_解析": g_a, "γ_时域": r["gamma_measured"],
                     "γ/ω_p": r["gamma_measured"] / wp,
                     "γ·√(1−μ)": r["gamma_measured"] * math.sqrt(1.0 - mu),
                     "时域 vs 解析相对误差": abs(r["gamma_measured"] - g_a) / g_a})
    spread = max(r["γ/ω_p"] for r in rows) - min(r["γ/ω_p"] for r in rows)
    check("T3: μ 标度 —— γ/ω_p 与 μ 无关（签名只在绝对速率上）",
          spread < 1e-3 and max(r["时域 vs 解析相对误差"] for r in rows) < 5e-3,
          {"γ/ω_p 三段散布": spread, "行": rows})
    out["T3"] = {"rows": rows, "spread": spread}

    # ── T4：μ 可关闭（阴性对照注入本身合格）──
    a = two_stream_default()
    b = two_stream_default()
    for sp in b.species:
        sp.mu = 0.0
    ga, _ = a.growth_analytic()
    gb, _ = b.growth_analytic()
    check("T4: μ=0 时逐位退化为标准两流（可关闭性）", ga == gb and a.linear_matrix().tobytes() == b.linear_matrix().tobytes(),
          {"γ_a": ga, "γ_b": gb})
    out["T4"] = {"gamma_mu0": ga}

    # ── T5：分辨率要求表（G0 的核心交付物）──
    res = resolution_table()
    check("T5: 分辨率要求表（μ 阶梯 ≤1e−2 时要求的精度）",
          all(r["3σ 单发所需相对精度"] > 0 for r in res) and abs(res[0]["相对效应量 δγ/γ"] - 5e-5) < 1e-6,
          {"μ=1e−4 相对效应量": res[0]["相对效应量 δγ/γ"],
           "μ=1e−4 3σ 单发精度": res[0]["3σ 单发所需相对精度"],
           "μ=1e−4 所需发数(单发 1e−3)": res[0]["若单发精度 1e−3，需要的发数"]})
    out["T5"] = {"rows": res}

    # ── T6：双流的工程约束（器件尺度实数）──
    isc = ion_stream_constraint(1e20, 100.0, 2.0, 1.0)
    check("T6: 双流稳定性约束 —— 离子声速档比冷两流档更严（D 等离子体实数）",
          isc["档B 是否比档A 更严"] is True,
          {"c_s [m/s]": isc["档B c_s = √(T_e/m_i) [m/s]"],
           "冷两流阈值 [m/s]": isc["档A 冷两流阈值 v0 ≳ ω_pi/k_min [m/s]"],
           "结论": isc["档B 工程约束"]})
    out["T6"] = isc

    # ── T7：与公开判决漏斗的 μ 阶梯对齐（跨仓一致性；仓库不在则跳过）──
    prog = HUSH / "progress.html"
    if prog.exists():
        html = prog.read_text(encoding="utf-8", errors="ignore")
        need = ["判决漏斗", "生死门", "15 人·月", "1.29%", "1e−4", "1e−3", "1e−2", "R_ci"]
        missing = [n for n in need if n not in html]
        check("T7: 与 hushfusion 公开漏斗同号（μ 阶梯 / G1 生死门 / 判决期）", not missing,
              {"缺失": missing} if missing else {"命中": len(need), "档": "1e−4 → 1e−3 → 1e−2"})
        out["T7"] = {"missing": missing}
    else:
        print(f"SKIP T7: 未找到 {prog}（跨仓一致性检查跳过）")
        out["T7"] = {"skipped": True}

    # ── T8：S2 待办如实登记（**不是通过项**）──
    todo = ["R_ci 零假设基准表（全工作区 vs 诊断分辨率）",
            "诊断分辨率要求（绝对时钟口径，非比值）",
            "双流阈值曲线 (μ, B, n) 闭合",
            "排序不可交换可观测性（先 RMF 后压缩 vs 反序，TD15–TD17）",
            "盲样试验（假阳性 ≤ 5%）"]
    print("TODO S2（下一批，未做）: " + "；".join(todo))
    out["TODO_S2"] = todo

    ok = all(r["pass"] for r in RESULTS)
    print(f"\nRESULT: {'ALL PASS' if ok else 'FAIL'}  ({len(RESULTS)} 门, {time.time() - t0:.1f}s)")
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "plan_s0_s1.json").write_text(json.dumps(
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "gates": RESULTS, "out": out},
        ensure_ascii=False, indent=2, default=str))
    print(f"报告 → artifacts/plan_s0_s1.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
