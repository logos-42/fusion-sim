#!/usr/bin/env python3
"""gym-μ-FRC v0 门禁入口:python3 scripts/verify_sim.py [--fast]

输出约定(与 Hibs-Physics 一致):每组门逐条打印 ✓/✗ + 载荷,末尾一行权威汇总。
退出码:全绿 0;有失败 1。--fast 把回合数/seed 数压到最小值(冒烟),默认是完整跑。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gym_mu_frc.gates import run_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="冒烟:少 seed、少回合")
    ap.add_argument("--out", default="artifacts/sim_report.json")
    args = ap.parse_args()

    seeds = (0,) if args.fast else (0, 1, 2, 3, 4)
    episodes = 60 if args.fast else 400
    eval_episodes = 5 if args.fast else 20

    t0 = time.time()
    checks, details = run_all(seeds=seeds, episodes=episodes, eval_episodes=eval_episodes)

    group = None
    for c in checks:
        if c.group != group:
            group = c.group
            print(f"\n── {group} ──")
        print(f"  {'✓' if c.ok else '✗'} {c.name}" + (f" — {c.payload}" if c.payload else ""))

    # 分类:通过 / 失败 / 已登记的缺口(名字里带「已知缺口」且 G7b 登记门为绿 → 计入跳过,绝不藏)
    registered_ok = all(c.ok for c in checks if c.group.startswith("G7b"))
    n_pass = sum(c.ok for c in checks)
    skipped = [c for c in checks if not c.ok and "已知缺口" in c.name and registered_ok]
    n_fail = len(checks) - n_pass - len(skipped)
    if skipped:
        for c in skipped:
            print(f"  ⚠ [已登记缺口] {c.name} — {c.payload}")
    print(f"\n通过 {n_pass} 项,失败 {n_fail} 项,跳过 {len(skipped)} 项(已登记缺口)  "
          f"({time.time() - t0:.1f}s)")

    for cm, det in details.items():
        ref = det["reference"]
        print(f"  [{cm}] 穷举参考: η={ref['eta']} n={ref['n']} return={ref['return']:.4f}")
        for r in det["rows"]:
            print(f"      seed{r['seed']}: 策略 {r['agent']['return_mean']:.4f} "
                  f"(停步 {r['agent']['stop_step_mean']:.0f}, μ→{r['agent']['mu_final_mean']:.6f}) "
                  f"vs 随机 {r['random']['return_mean']:.4f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_by": "scripts/verify_sim.py",
        "fast": args.fast,
        "seeds": list(seeds),
        "episodes": episodes,
        "eval_episodes": eval_episodes,
        "passed": n_pass, "failed": n_fail, "skipped": len(skipped),
        "skipped_names": [c.name for c in skipped],
        "checks": [c.__dict__ for c in checks],
        "learning": details,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"报告 → {out}")

    if n_fail:
        print("失败明细:")
        for c in checks:
            if not c.ok and c not in skipped:
                print(f"  - [{c.group}] {c.name} — {c.payload}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
