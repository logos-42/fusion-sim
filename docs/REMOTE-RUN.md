# 远程运行（2-D / 大扫描）

本仓的计算分层：**本机做门禁与判决量，远端做扫描与 2-D/3-D**。

## 1. 远端是什么（2026-09-30 实测）

    入口      ssh -p 6100 root@100.100.30.185        （容器：Ubuntu 20.04，PID1=tail -f /dev/null）
    GPU       2 × NVIDIA A100-PCIE-40GB（驱动 570.133.07），nvidia-smi 正常 ⟹ 直通可用
    CPU       48 核 Xeon Silver 4510 @ 3.3 GHz
    磁盘      /work 数据盘 15 TB（余 2.2 TB）—— **容器重启只有 /work 保留**
    环境      /work/liuyuanjie/envs/fusion-sim/bin/python  (3.11.16 + numpy 2.4.6 + gymnasium 1.3.0)
    代码      /work/liuyuanjie/fusion-sim/
    网络      阿里 pypi 镜像与 pypi.org 均 HTTP 200

共用机：/work 下有 5 个他人目录，GPU 上常有他人任务占用。**先看空闲再决定**：

    nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader

按 `remote-gpu-server-ops` 的规则：每队列显存预算 ≤ 4 GiB，所有队列之和 ≤ 空闲显存 60%。
不需要 GPU 的活儿（本仓目前的全部活儿）就走 CPU 农场，完全不影响他人。

## 2. 传代码的纪律

    cd /Users/apple/Downloads/fusion-sim
    tar --exclude='.venv*' --exclude='__pycache__' --exclude='.git' -czf /tmp/fusion-sim.tgz .
    md5 -q /tmp/fusion-sim.tgz                       # 记下本机哈希
    scp -P 6100 /tmp/fusion-sim.tgz root@100.100.30.185:/work/liuyuanjie/
    ssh -p 6100 root@100.100.30.185 'md5sum /work/liuyuanjie/fusion-sim.tgz'   # 两端必须一致

解包后**一定重传改动过的单文件**（整包覆盖会把远端产物一起刷掉）：

    scp -P 6100 ./scripts/verify_s2_gates.py root@...:/work/liuyuanjie/fusion-sim/scripts/verify_s2_gates.py
    ssh ... 'md5sum /work/liuyuanjie/fusion-sim/scripts/verify_s2_gates.py'    # 对代码真正读取的那个路径

## 3. 跨机复现性（已验证）

同一份代码两端跑，判决量数字**逐位一致**（12 位有效数字）：

    项                     本机 i7-7700HQ        远端 Xeon 4510
    S2-3 噪声三档           0.013696735583 …      0.013696735583 …
    S2-6 盲样阈值           0.00810748652         0.00810748652
    S2-2b 1/L² 斜率         −2.0                  −2.0
    S2-5 基线/数值底        2.054866736518        2.054866736518

⟹ 门禁不依赖机器数值特性，可以任选一端跑。

**两处 SKIP 是环境依赖，不是丢断言**（脚本会显式打印 SKIP 行）：
远端没有同级的 `hushfusion` 仓 ⟹ `S2-7`、`T7`（跨仓一致性）跳过。要想远端也跑这两门，
把 hushfusion 也传上去即可（或在 gate 脚本里把 `HUSH` 改成环境变量）。

## 4. 2-D 吞吐实测（`scripts/probe_2d_throughput.py`）

单元成本 = 每步 4×FFT2(Poisson) + 4×FFT2(梯度)（RK4 一步的最小量，真实两流体场数更多 ⟹ 下表是**下界**）。

单进程：

    N       本机 i7-7700HQ      远端 Xeon 4510     远端 2000 步单次
    256     102 ms/步           33.1 ms/步         66 s
    512     249 ms/步           153.4 ms/步        307 s
    1024    1891 ms/步          837.9 ms/步        1676 s

多进程农场（远端，加速 = 总吞吐/单进程吞吐）：

    N=256   单进程 30.2 步/s  K=4: 3.80x  K=8: 6.77x  K=16: **8.86x**  K=32: 8.58x  K=48: 8.37x
    N=512   单进程  6.5 步/s  K=4: 3.42x  K=8: 5.30x  K=16: 7.46x  K=32: 8.28x  K=48: **8.49x**
    N=1024  单进程  1.2 步/s  K=4: 3.01x  K=8: 4.46x  K=16: **5.84x** K=32: 4.80x  K=48: 6.69x

**48 核只换来 ~8.5 倍**（不是 48 倍）：FFT 是内存带宽受限的，核数上去后单进程变慢
（K=48 时单进程 142 ms/步 vs 单跑 33 ms/步）。**K=16 是甜点**，K≥32 只增占用不增吞吐。

扫描预算（27 点 × 2000 步）：

    N=256    单点 1.1 分钟    27 点扫描 **3.4 分钟**
    N=512    单点 5.1 分钟    27 点扫描 16.3 分钟
    N=1024   单点 27.9 分钟   27 点扫描 112.8 分钟

实物算量比上表大 3–5 倍（场数多）⟹ 现实预期：**N=256 的 27 点扫描 ~10–17 分钟**（本机过夜才做完的量）。
GPU（cupy）未测；单次运行要被 GPU 提速得 N ≥ 512，且必须先在空闲显存里留出预算。

## 5. 放哪跑（决策表）

    单元测试 / 判据推导 / 门禁 / 判决量            本机（几十秒～2 分钟，改一次跑一次）
    1-D 扫描、多 seed 重跑                         任一端（本机也行，远端快 ~2.5×）
    2-D 单工况、N=256 小扫描                       远端（分钟级）
    2-D N=512/1024 扫描、多 seed                    远端 + 多进程农场
    3-D、动理学（Gkeyll/WarpX）                    需要 GPU + 大显存；本仓目前不涉及
