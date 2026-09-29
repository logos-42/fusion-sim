#!/usr/bin/env bash
# B 线门齿:把 μ 更新改坏(当成 η≡1)与把关闭步改坏(+1),各自必须让门变红
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv-sim/bin/python}
cp gym_mu_frc/physics.py /tmp/tt_physics.py

echo "=== A) 注入:mu_step 忽略 η(变成 η≡1) ==="
python3 - <<'PY'
import pathlib
p = pathlib.Path('gym_mu_frc/physics.py'); s = p.read_text()
old = "    return mu + eta * (1.0 - mu)"
assert old in s
p.write_text(s.replace(old, "    return mu + 1.0 * (1.0 - mu)   # 注入", 1))
PY
$PY scripts/verify_sim.py --fast 2>&1 | grep -E "300 步后|441 网格|严格推进|TD21|μ₀=0 且|通过 [0-9]+ 项" | head -8
cp /tmp/tt_physics.py gym_mu_frc/physics.py

echo "=== B) 注入:close_step 多算一步 ==="
python3 - <<'PY'
import pathlib
p = pathlib.Path('gym_mu_frc/physics.py'); s = p.read_text()
old = "    return int(math.ceil(n - 1e-12))"
assert old in s
p.write_text(s.replace(old, "    return int(math.ceil(n - 1e-12)) + 1   # 注入", 1))
PY
$PY scripts/verify_sim.py --fast 2>&1 | grep -E "TD21|首次关闭|通过 [0-9]+ 项" | head -6
cp /tmp/tt_physics.py gym_mu_frc/physics.py

echo "=== 复原后总检 ==="
$PY scripts/verify_sim.py --fast 2>&1 | tail -2
