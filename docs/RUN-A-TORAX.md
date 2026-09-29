# A 线报告:TORAX + Gym-TORAX 在本机跑起来了

**日期** 2026-09-29 · **机器** Intel macOS 13.7.8(x86_64)· Python 3.12.8 · CPU-only(JAX 无 GPU)
**产物** `artifacts/torax_smoke.json` · **脚本** `scripts/run_torax_smoke.py`

## 1. 结论(先给结论)

1. **链路通**:TORAX(可微 1-D 输运, JAX)+ Gym-TORAX(Gymnasium 控制环境)在本机 CPU 上装得起来、
   跑得动,环境的 Gymnasium 契约检查通过。每步 **0.36 s**,一条 151 步的 ITER 混合运行 ≈ **54 s**。
2. **环境确实能分辨策略好坏**:同一任务、同一视野(151 步),
   脚本化的 ITER 混合时序把末态 **Q_fusion 做到 7.68**,随机动作只有 **1.50–2.76**。
3. **本机必须钉版本**(否则装不上,见 §3):`jax/jaxlib==0.4.38 + torax==1.0.3 + gymtorax==1.0.0`。
4. **本机没做的事**:没有训练 RL 策略(本机没装 SB3/RLlib);`PIDAgent` 没跑(它需要 5 个外部参数,
   包里没给示例)—— 这两条都是"待补",不是"验过了"。

## 2. 实跑数据(3 seeds × 两个策略基线)

| 策略 | 成功回合 | 总回报 | 步数 | 每步 | 末态 Q_fusion | 末态 τ_E |
|---|---|---|---|---|---|---|
| 随机动作 (RandomAgent) | 3/3 | 3.280 | 151 | 0.358 s | 1.50 / 2.63 / 2.76 | 2.54 / 3.65 / 3.26 |
| 脚本化 ITER 混合 (IterHybridAgent) | 3/3 | 3.409 | 151 | 0.361 s | **7.68**(三 seed 完全一致,脚本确定性) | 2.38 |

读法:两个策略的**总回报**接近(3.28 vs 3.41,因为环境的回报是场景跟随型),
但**末态 Q_fusion 差 3–5 倍** —— 说明这个任务有真实的可优化空间,不是"随便动也差不多"。
脚本 agent 三 seed 完全一致,是因为它是**预定时序**(没有随机性):这正好是它的用途 —— 当参考基线。

## 3. 本机的版本天花板(重要,别再踩)

`torax 1.4.3`(最新)要求 `jax>=0.10 / jaxlib>=0.10`;而 **x86_64 macOS 上可安装的最高 `jaxlib` 是 0.4.38**
(上游已停发 Intel Mac 的 wheel)。所以在这台机器上:

```
jax==0.4.38  jaxlib==0.4.38  torax==1.0.3  gymtorax==1.0.0      # ← 实测可用
```

`torax 1.0.3` 要求 `jaxlib>=0.4.32` ✓,`gymtorax 1.0.0` 要求 `torax==1.0.*` ✓,两者对得上。
Apple Silicon / Linux 上不受此限,直接装最新版即可(那也意味着:想用新特性/更大算例,走远程 GPU 服务器)。

## 4. 环境长什么样(为后面训练准备的事实)

`gym.make("gymtorax/IterHybrid-v0")` 已注册,开箱可用:

- **观测**:`Dict(profiles, scalars)`。`scalars` 里有 `tau_E`、`Q_fusion`、`beta_N`、`q95`、`H98`、
  `P_alpha_total`、`n_e_volume_avg`、`T_i_volume_avg`、`P_SOL_total` …(可直接当奖励/约束信号);
  `profiles` 是 26–27 点的一维剖面(T_e, T_i, n_e, q, psi …)。
- **动作**:`Dict(ECRH=(功率, 频率?, 相位?), Ip=(电流,), NBI=(功率, 能量, 角度))`,连续,带上限
  (NBI ≤ 33 MW,ECRH ≤ 20 MW,Ip ≤ 15 MA,斜坡率 0.2 MA/s)。
- **自带 agent**:`IterHybridAgent`(脚本时序,可当参考)· `RandomAgent`(随机基线)·
  `PIDAgent`(需自备 `get_j_target/ramp_rate/kp/ki/kd`)。

## 5. 下一步(按性价比排序)

1. **训一轮真策略**:装 `stable-baselines3`(或 `cleanrl`),用 PPO/SAC 在 `Ip`+`NBI` 上做
   "跟踪目标 Q_fusion + 不越 beta_N/q95 约束";3–5 seeds,随机基线 + 脚本基线两路对照(与实验纪律一致)。
2. **接我们自己的环境**:B 线 `gym-μ-FRC` 已经与 Gymnasium 同构 —— 两个环境可以共用同一套训练/评估脚本,
   这正是把"μ 修正"塞进 AI 闭环的入口。
3. **可微路线的升级**:TORAX 是 JAX 写的,可直接 `jax.grad` 穿过时间步做脉冲设计/控制器优化
   (这是它相比传统输运码最大的不同);本机 CPU 够做小规模,规模上去要 GPU 服务器。
4. **别在本机上追新特性**:版本天花板就在 `jaxlib 0.4.38`,要新东西就上服务器(见 §3)。

## 6. 诚实边界

- 这一轮**没有**训练任何策略,也没调任何物理参数;所有数字都是"原样跑基线"的读数。
- `Q_fusion` 等量是**仿真输出**,不是实验数据;TORAX 的输运模型本身含经验/代理模型(如 QLKNN)。
- 本机是 CPU + 老 jax:适合跑通链路与小算例,**不适合**把它当性能基准,更不适合外推规模。
