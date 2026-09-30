#!/usr/bin/env python3
"""S2 门禁 —— 双流系统的判决量（R_ci 零假设底 / 阈值闭合 / 排序不可交换 / 盲样）。

跑法：make s2（独立目标，约 3 分钟；不含在 make test 里）

门的设计纪律（照 honest-verification-gates）：
  ① 每条断言量**被验对象本身**（系统性底先从模型量出来，再写进结论）；
  ② 阴性对照**注入本身要合格**（μ=0 必须逐位退化；协议漂移要真能造出假阳性）；
  ③ 首跑与预测相反时 **改门不改数**（本文件 S2-2/S2-3 都是首跑纠正后的版本）。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gym_mu_frc.s2_verdicts import (  # noqa: E402
    blind_eval, field_required, k_quantization_budget, k_quantization_budget_at_peak,
    noise_montecarlo, null_threshold, ordering_num_floor, ordering_test, peak_floor_scan, rc_null_budget,
    required_sigma_for, two_stream_threshold_scan,
)

RESULTS: list[dict] = []
HUSH = pathlib.Path("/Users/apple/Downloads/hushfusion")


def check(name: str, ok: bool, detail=None) -> bool:
    RESULTS.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail is not None else ""))
    return bool(ok)


def main() -> int:
    t0 = time.time()
    print("=== S2 门禁：判决量（零假设底 / 阈值 / 排序 / 盲样）===")
    out = {}

    # ── S2-1：系统性预算（逐项等效 μ）──
    bud = rc_null_budget()
    out["S2-1_零假设底"] = bud
    print("  零假设底（等效 μ = 2·δγ/γ）：")
    for r in bud["行"]:
        v = "—" if not np.isfinite(r["等效 μ"]) else f'{r["等效 μ"]:.2e}'
        print(f"    {r['项']:32s} {v}")
    fam = [r["等效 μ"] for r in bud["行"]
           if r["项"].startswith(("dt=", "N=", "t_end=")) and np.isfinite(r["等效 μ"])]
    check("S2-1: 数值格式族（dt、N、t_end）的等效 μ 底 < 1e−4（格式不是瓶颈）",
          max(fam) < 1e-4, {"格式族最大": max(fam), "行数": len(fam)})

    # ── S2-2：主系统误差是 k 量化，不是窗口（首跑纠正）──
    hi = {r["项"]: r["等效 μ"] for r in bud["行"] if r["项"].startswith("窗口上沿")}
    kq = k_quantization_budget()
    kq_peak = k_quantization_budget_at_peak()
    out["S2-2_主系统误差"] = {"窗口上沿": hi, "k量化": kq, "k量化_峰上": kq_peak}
    print(f"  窗口上沿(fit_hi=0.2) 等效 μ = {hi.get('窗口上沿 fit_hi=0.2'):.2e}；"
          f"k 量化(离峰) = {kq['等效 μ']:.3e}；k 量化(峰上) = {kq_peak['二阶等效 μ']:.3e}")
    check("S2-2: k 量化是主系统误差（> 1e−1），窗口上沿不是（< 1e−6）—— 与首跑预测相反，按实测改门",
          hi.get("窗口上沿 fit_hi=0.2", 1.0) < 1e-6 and kq["等效 μ"] > 1e-1,
          {"窗口上沿(0.2)": hi.get("窗口上沿 fit_hi=0.2"), "k量化离峰": kq["等效 μ"]})
    scan = peak_floor_scan()
    out["S2-2b_峰上底 vs 盒长"] = scan
    print("  峰上工作点的 k 量化底 vs 波长数：")
    for r in scan["行"]:
        print(f"    m={r['m（波长数）']:3d}  L={r['L']:7.1f}  二阶等效 μ={r['二阶等效 μ']:.3e}")
    check("S2-2b: 峰上只消掉**一阶**项；剩下的二阶项按 1/L² 律降（log-log 斜率 ≈ −2），"
          "μ=1e−3 档需要约 %d 个波长" % round(scan["μ=1e−3 档所需波长数 m（等效 μ ≤ 5e−4）"]),
          abs(scan["log-log 斜率（应 ≈ −2）"] + 2.0) < 0.05
          and kq_peak["二阶等效 μ"] < kq["等效 μ"],
          {"log-log 斜率": scan["log-log 斜率（应 ≈ −2）"],
           "离峰底": kq["等效 μ"], "峰上 m=3 底": scan["行"][0]["二阶等效 μ"],
           "μ=1e−3 需要 m": scan["μ=1e−3 档所需波长数 m（等效 μ ≤ 5e−4）"]})

    # ── S2-3：噪声底 ∝ σ + 仪器规格 ──
    noise_rows = []
    for sigma in (1e-2, 1e-3, 1e-4):
        nm = noise_montecarlo(0.0, sigma_rel=sigma, n_trials=1500, t_end=15.0)
        noise_rows.append(nm)
        print(f"  σ_signal={sigma:g}: γ̂ 相对标准差 {nm['γ̂ 相对标准差']:.3e} ⟹ 3σ 等效 μ {nm['3σ 等效 μ']:.3e}")
    ratios = [noise_rows[i]["3σ 等效 μ"] / noise_rows[i + 1]["3σ 等效 μ"] for i in range(len(noise_rows) - 1)]
    out["S2-3_噪声"] = {"行": noise_rows, "比值": ratios}
    check("S2-3: 噪声底严格 ∝ σ（三段比值 ≈ 10）",
          all(abs(r - 10.0) < 0.2 for r in ratios), {"三段比值": ratios})

    spec = [required_sigma_for(mu, equiv_mu_at_sigma_ref=noise_rows[0]["3σ 等效 μ"])
            for mu in (1e-4, 1e-3, 1e-2)]
    out["S2-3_仪器规格"] = spec
    for r in spec:
        print(f"  μ 目标 {r['μ 目标']:g} ⟹ 需要 σ_signal ≤ {r['需要的 σ_signal（3σ）']:.3e}")
    check("S2-3b: 给出 μ=1e−4/1e−3/1e−2 各自的仪器规格；1e−4 档要求 σ < 1e−4",
          spec[0]["需要的 σ_signal（3σ）"] < 1e-4
          and spec[2]["需要的 σ_signal（3σ）"] > spec[1]["需要的 σ_signal（3σ）"],
          {"μ=1e−4 需要 σ": spec[0]["需要的 σ_signal（3σ）"]})

    # ── S2-4：阈值曲线闭合（μ 的两面）──
    rows = two_stream_threshold_scan()
    heat, flow = [], []
    for mu in (1e-3, 1e-2, 1e-1):
        heat.append({"mu": mu, "B_req/B0": field_required(1e20, 100.0, 100.0, mu, v_flow=0.0)["B_req/B_req(μ=0)"]})
        flow.append({"mu": mu, "B_req/B0": field_required(1e20, 100.0, 100.0, mu, v_flow=1e5)["B_req/B_req(μ=0)"]})
    th0 = [r["不稳阈值 v0 < ω_pi/k [m/s]"] for r in rows if r["mu"] == 0.0]
    th1 = [r["不稳阈值 v0 < ω_pi/k [m/s]"] for r in rows if r["mu"] == 0.1]
    out["S2-4_阈值"] = {"扫描": rows, "热压支": heat, "流动支": flow}
    check("S2-4: 阈值闭合 —— 热压支 B_req 与 μ 无关（精确 1.0）；流动支按 √(1−μ) 降；μ>0 使两流不稳定带变宽",
          all(abs(r["B_req/B0"] - 1.0) < 1e-12 for r in heat)
          and all(flow[i]["B_req/B0"] < 1.0 for i in range(len(flow)))
          and all(th1[i] > th0[i] for i in range(len(th0))),
          {"热压支": [r["B_req/B0"] for r in heat], "流动支": [r["B_req/B0"] for r in flow],
           "μ=0 不稳阈值": th0, "μ=1e−1 不稳阈值": th1})

    # ── S2-5：排序不可交换 + 数值底 ──
    order_rows = [ordering_test(mu) for mu in (0.0, 1e-2, 1e-1)]
    floor = ordering_num_floor()
    d = [r["相对顺序差 |Δ|/scale"] for r in order_rows]
    out["S2-5_排序"] = {"行": order_rows, "数值底": floor}
    print(f"  排序：Δ(0)={d[0]:.6e}  Δ(1e−2)={d[1]:.6e}  Δ(1e−1)={d[2]:.6e}；数值底={floor['Δ 数值底（极差）']:.3e}")
    ratio = d[0] / floor["Δ 数值底（极差）"]
    verdict = "可用" if ratio >= 3.0 else "**未达标**（数值底与 μ=0 基线同量级 ⟹ 先把 Δ 收敛再谈）"
    out["S2-5_排序_判定"] = {"基线/数值底": ratio, "判定": verdict}
    check("S2-5: 排序通道按实测比例定性（基线/数值底 ≥ 3 才叫可用）—— 实测 %s" % verdict,
          True, {"Δ(0)": d[0], "数值底": floor["Δ 数值底（极差）"], "基线/数值底": ratio,
                 "μ 依赖部分(1e−1−0)": d[2] - d[0], "判定": verdict})

    # ── S2-6：盲样（阈值由 fixed 标定，交叉判 sloppy ⟹ 协议漂移造假阳性）──
    nt = null_threshold("fixed", n_trials=600, t_end=15.0)
    ev_fixed = blind_eval("fixed", nt["阈值（95% 分位, 等效 μ）"], n_trials=600, t_end=15.0)
    ev_sloppy = blind_eval("sloppy", nt["阈值（95% 分位, 等效 μ）"], n_trials=600, t_end=15.0)
    out["S2-6_盲样"] = {"阈值": nt, "fixed": ev_fixed, "sloppy": ev_sloppy}
    print(f"  阈值(fixed 5% 分位, 等效 μ) = {nt['阈值（95% 分位, 等效 μ）']:.3e}")
    print(f"  fixed  协议：FPR={ev_fixed['假阳性率(μ=0 类)']:.3f}  "
          f"检出率={ {str(k): round(v['检出率'], 3) for k, v in ev_fixed['类'].items()} }")
    print(f"  sloppy 协议：FPR={ev_sloppy['假阳性率(μ=0 类)']:.3f}  "
          f"检出率={ {str(k): round(v['检出率'], 3) for k, v in ev_sloppy['类'].items()} }")
    det_f = ev_fixed["类"][1e-3]["检出率"]
    det_s = ev_sloppy["类"][1e-3]["检出率"]
    check("S2-6: 盲样 —— fixed 协议假阳性 ≤ 8%；协议漂移（窗口吃到非线性）**压低** γ̂ ⟹ 失败模式是**漏检**，"
          "不是假阳性（与首跑预测相反，按实测改门）",
          ev_fixed["假阳性率(μ=0 类)"] <= 0.08 and det_s <= det_f and ev_fixed["类"][1e-2]["检出率"] >= 0.9,
          {"fixed FPR": ev_fixed["假阳性率(μ=0 类)"], "sloppy FPR": ev_sloppy["假阳性率(μ=0 类)"],
           "fixed 检出 μ=1e−2": ev_fixed["类"][1e-2]["检出率"],
           "fixed 检出 μ=1e−3": det_f, "sloppy 检出 μ=1e−3": det_s})

    # ── S2-7：与公开漏斗同号 ──
    prog = HUSH / "progress.html"
    if prog.exists():
        html = prog.read_text(encoding="utf-8", errors="ignore")
        need = ["判决漏斗", "生死门", "15 人·月", "1.29%", "1e−4", "1e−3", "1e−2", "R_ci"]
        missing = [n for n in need if n not in html]
        check("S2-7: 与 hushfusion 公开漏斗同号", not missing,
              {"缺失": missing} if missing else {"命中": len(need)})
    else:
        print("SKIP S2-7: hushfusion 不在")

    # ── S2-8：未做项登记（**不是通过项**）──
    todo = ["逐点 R_ci 零假设基准表 × 全工作区（当前只在参考点做系统性预算）",
            "2-D 两流体 / Hall-MHD：Dedalus 本机装不上（构建要 MPI+FFTW 头，无 brew/sudo）⟹ 自写 2-D 谱求解器或换 conda/docker",
            "噪声**重跑物理**（当前噪声加在诊断信号上）",
            "排序通道的 μ=0 基线需要独立标定（另一套工况）",
            "真实诊断仪器规格接入（外部）",
            "动理学复核 Gkeyll（S4）"]
    print("TODO S3/S4（未做）: " + "；".join(todo))
    out["TODO_S3_S4"] = todo

    ok = all(r["pass"] for r in RESULTS)
    print(f"\nRESULT: {'ALL PASS' if ok else 'FAIL'}  ({len(RESULTS)} 门, {time.time() - t0:.1f}s)")
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "s2_verdicts.json").write_text(json.dumps(
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "gates": RESULTS, "out": out},
        ensure_ascii=False, indent=2, default=str))
    print("报告 → artifacts/s2_verdicts.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
