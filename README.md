# fusion-sim — a fusion simulation workbench (v0)

**English** · [中文](./README.zh-CN.md)

Two tracks, one repo:

| Track | What | Entry point |
|---|---|---|
| **A · off-the-shelf AI stack** | TORAX (differentiable 1-D transport, JAX) + Gym-TORAX (Gymnasium control env): does the AI-ready fusion stack run *here*, and what does it look like | `scripts/run_torax_smoke.py`, report in `docs/RUN-A-TORAX.md` |
| **B · our own environment** | `gym-μ-FRC v0`: μ dynamics (TD1–TD21) + FRC window (FC11b/FC5b) + RMF cost model, as a Gymnasium control problem with RL baselines and gates | `gym_mu_frc/`, gates via `make test` |

## Why both

A answers "can the AI-friendly open-source fusion stack be installed and exercised locally" (a plumbing question).
B answers "can *our* physics (μ-modified inertia) be turned into a trainable, verifiable, comparable control problem"
(a differentiation question).

No open-source fusion code knows anything about μ. So B's validation rests on three things only:
① line-by-line agreement with the formalisation in Hibs-Physics (two independent implementations),
② the μ→0 regression (with μ switched off it must fall back to the classical FRC scalings),
③ multi-seed runs against a random baseline **and** an exhaustive search reference.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements-sim.txt     # B: numpy + gymnasium only
pip install -r requirements-torax.txt   # A: jax / flax / torax / gymtorax
```

Version pinning is mandatory on Intel macOS: `torax 1.4.3` needs `jaxlib>=0.10`, but the newest
installable `jaxlib` for x86_64 macOS is **0.4.38** — hence `jax/jaxlib==0.4.38 + torax==1.0.3 + gymtorax==1.0.0`
(see `docs/RUN-A-TORAX.md` §3). On Apple Silicon / Linux, use the latest versions.

## Run

```bash
make test          # B: full gate set (5 seeds, 400 training episodes, baselines, exhaustive reference)
make fast          # B: smoke (1 seed, 60 episodes)
make smoke-torax   # A: TORAX smoke + one Gym-TORAX environment pass
bash scripts/teeth.sh   # gate teeth: inject two real defects (μ update ignoring η / close step off by one) → must go red → restore
```

Outputs: one authoritative summary line, `artifacts/sim_report.json` (per-check name / red-green / payload),
and `artifacts/torax_smoke.json` for track A.

**Gate teeth (measured)**: making `mu_step` ignore η → **7 checks red**; shifting `close_step` by one →
**1 check red** (expected 165 / got 166); restore → green. If an injected defect makes a contract function
raise, group-level guarding records it as **one red check** instead of crashing the whole suite
(a crashed suite reads far too much like "nothing ran").

## Honest boundaries (quote these with any result)

- the μ update `μ ↦ μ + η(1−μ)` and the bridge `η = flattening progress` are **model choices**, not derived physics;
- the cost model (how flattening work is priced) is likewise a choice — the gates run two ablations (`work` / `locking`);
- the FRC side uses only two results already proved in Hibs-Physics (`FC11b`, `FC5b`) and introduces **no new mechanism**;
- the repo contains no experimental data and does no fitting: every statement is conditional on this model;
- it is **not** a plant-level simulation and does not replace neutronics / tritium / materials (a different scale);
- **v0's RL learner does not solve the control problem** — this is registered as an open gap, with the measured
  spread, in `docs/ALIGN-LEAN.md` §4.1 (and the gate suite counts it as an explicit skip, never silently passes it).
