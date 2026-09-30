PY ?= .venv/bin/python
# 轻量运行(只装 numpy+gymnasium 时):make fast PY=.venv-sim/bin/python
# 计划门(S0/S1 双流)只需 numpy ⟹ 用 .venv-sim 也行:make plan PY=.venv-sim/bin/python

.PHONY: test fast smoke-torax report plan s2 s3 clean help

help:
	@echo "make test        —— B 线全门禁(5 seeds / 400 回合) + S0/S1 计划门"
	@echo "make fast        —— B 线冒烟(1 seed / 60 回合;含计划门)"
	@echo "make plan        —— S0/S1 双流门(T0–T7 + S2 待办登记)⟹ artifacts/plan_s0_s1.json"
	@echo "make s2          —— S2 判决量门(零假设底/阈值/排序/盲样)⟹ artifacts/s2_verdicts.json"
	@echo "make s3          —— S3 判决实验(自写 2-D 求解器: 局部/全局 vs 离散拓扑量)⟹ artifacts/s3_nonlocality/"
	@echo "make smoke-torax —— A 线:TORAX + Gym-TORAX 冒烟(需 requirements-torax.txt)"
	@echo "make report      —— 跑门禁并写 artifacts/sim_report.json"

test:
	$(PY) scripts/verify_sim.py
	$(PY) scripts/verify_plan_gates.py

fast:
	$(PY) scripts/verify_sim.py --fast
	$(PY) scripts/verify_plan_gates.py

plan:
	$(PY) scripts/verify_plan_gates.py

s2:
	$(PY) scripts/verify_s2_gates.py

s3:
	$(PY) -m gym_mu_frc.s3_nonlocality && $(PY) scripts/verify_s3_gates.py

report:
	$(PY) scripts/verify_sim.py --out artifacts/sim_report.json

smoke-torax:
	$(PY) scripts/run_torax_smoke.py

clean:
	rm -rf artifacts/sim_report.json artifacts/plan_s0_s1.json __pycache__ gym_mu_frc/__pycache__ scripts/__pycache__
