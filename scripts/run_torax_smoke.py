#!/usr/bin/env python3
"""A 线冒烟:TORAX(可微 1-D 输运)+ Gym-TORAX(Gymnasium 控制环境)在本机的可运行性报告.

跑什么(不训练 RL:本机没装 SB3/RLlib,那是下一步):
  ① 版本/链路自检(jax · torax · gymtorax · gymnasium + 设备)
  ② 环境契约:gymnasium check_env
  ③ 同一条 ITER ramp-up 任务上两个**策略基线**对照各 3 seed:
       RandomAgent(随机动作)vs PIDAgent(经典反馈控制)
     记录:总回报 · 步数 · 关键等离子体量(tau_E / Q_fusion / beta_N / P_alpha_total)
  ④ 每步耗时(决定后面能不能在本机做优化/训练)

产物:artifacts/torax_smoke.json + 屏幕表格。诚实口径:这是**链路报告**,不是物理结论。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "torax_smoke.json"

SCALARS_OF_INTEREST = ("tau_E", "Q_fusion", "beta_N", "P_alpha_total", "n_e_volume_avg",
                       "T_i_volume_avg", "W_thermal_total", "H98", "q95")


def versions() -> dict:
    import importlib.metadata as md

    import gymnasium
    import jax
    import jaxlib

    out = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "torax": md.version("torax"),
        "gymtorax": md.version("gymtorax"),
        "gymnasium": gymnasium.__version__,
        "devices": [str(d) for d in jax.devices()],
    }
    try:
        out["platform"] = jax.devices()[0].platform
    except Exception:  # noqa: BLE001
        out["platform"] = "unknown"
    return out


def scalar_snapshot(obs) -> dict:
    sc = obs["scalars"]
    return {k: float(sc[k][0]) for k in SCALARS_OF_INTEREST if k in sc}


def run_episode(env, agent, seed: int, step_cap: int = 0) -> dict:
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    t0 = time.time()
    while True:
        action = agent.act(obs)
        obs, r, terminated, truncated, _info = env.step(action)
        total += float(r)
        steps += 1
        if terminated or truncated:
            break
        if step_cap and steps >= step_cap:      # 只用于计时冒烟,不算完整回合
            break
    return {
        "seed": seed,
        "return": total,
        "steps": steps,
        "seconds": round(time.time() - t0, 2),
        "sec_per_step": round((time.time() - t0) / max(steps, 1), 3),
        "final_scalars": scalar_snapshot(obs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=0, help=">0 时只跑这么多步(计时冒烟,不是完整回合)")
    args = ap.parse_args()

    import gymnasium as gym
    import gymtorax  # noqa: F401  (注册环境)
    from gymnasium.utils import env_checker
    from gymtorax.agents import IterHybridAgent, RandomAgent

    report: dict = {"versions": versions(), "episodes": {}, "notes": []}
    print("版本:", json.dumps(report["versions"], ensure_ascii=False))

    env = gym.make("gymtorax/IterHybrid-v0")

    try:
        env_checker.check_env(env, skip_render_check=True)
        report["env_check"] = "ok"
        print("环境契约 check_env: ok")
    except Exception as exc:  # noqa: BLE001
        report["env_check"] = f"{type(exc).__name__}: {exc}"
        print("环境契约 check_env: 失败 —", report["env_check"])

    # 两个基线:随机动作 vs 脚本化的 ITER hybrid 参考时序(IterHybridAgent)。
    # PIDAgent 也在包里,但它的构造要 (get_j_target, ramp_rate, kp, ki, kd) 五个外部参数,
    # 包内没有给例子 → 记成"待补",不当成"验过了"。
    for name, factory in (("random", lambda s: RandomAgent(env.action_space)),
                          ("scripted_iter_hybrid", lambda s: IterHybridAgent(env.action_space))):
        rows = []
        for seed in range(args.seeds):
            try:
                rows.append(run_episode(env, factory(seed), seed, step_cap=args.steps))
            except Exception as exc:  # noqa: BLE001
                rows.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
        ok = [r for r in rows if "error" not in r]
        report["notes"].append("PIDAgent 未跑:构造需要 get_j_target/ramp_rate/kp/ki/kd,包内无示例")
        report["episodes"][name] = {
            "rows": rows,
            "return_mean": st.mean([r["return"] for r in ok]) if ok else None,
            "steps_mean": st.mean([r["steps"] for r in ok]) if ok else None,
            "sec_per_step_mean": st.mean([r["sec_per_step"] for r in ok]) if ok else None,
            "step_cap": args.steps or None,
        }
        if ok:
            print(f"\n[{name}] {len(ok)}/{args.seeds} 回合成功 | 回报均值 "
                  f"{report['episodes'][name]['return_mean']:.3f} | 步数均值 "
                  f"{report['episodes'][name]['steps_mean']:.1f} | 每步 "
                  f"{report['episodes'][name]['sec_per_step_mean']:.3f}s")
            for r in ok:
                q = r["final_scalars"].get("Q_fusion")
                tau = r["final_scalars"].get("tau_E")
                print(f"    seed{r['seed']}: return={r['return']:.3f} steps={r['steps']} "
                      f"Q_fusion={q} tau_E={tau}")
        else:
            print(f"\n[{name}] 全部回合失败:", rows[0].get("error"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n报告 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
