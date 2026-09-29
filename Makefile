PY ?= .venv/bin/python
# 轻量运行(只装 numpy+gymnasium 时):make fast PY=.venv-sim/bin/python

.PHONY: test fast smoke-torax report clean help

help:
	@echo "make test        —— B 线全门禁(5 seeds / 400 回合)"
	@echo "make fast        —— B 线冒烟(1 seed / 60 回合)"
	@echo "make smoke-torax —— A 线:TORAX + Gym-TORAX 冒烟(需 requirements-torax.txt)"
	@echo "make report      —— 跑门禁并写 artifacts/sim_report.json"

test:
	$(PY) scripts/verify_sim.py

fast:
	$(PY) scripts/verify_sim.py --fast

report:
	$(PY) scripts/verify_sim.py --out artifacts/sim_report.json

smoke-torax:
	$(PY) scripts/run_torax_smoke.py

clean:
	rm -rf artifacts/sim_report.json __pycache__ gym_mu_frc/__pycache__ scripts/__pycache__
