#!/usr/bin/env python3
"""S3 门禁 —— 判决实验：局部/全局平滑操作 vs 离散拓扑量。

跑法：make s3（独立目标，约 5–8 分钟；含自写 2-D 谱求解器，不进 make test）

门的设计（照 honest-verification-gates）：
  ① 断言量被验对象本身：理想操作不改 N（违例计数）、重联事件改 N（整数跳变）；
  ② **发散哨兵**：数值发散返回 N = −1，绝不许把 NaN 读成"合并"（本步第一版就栽在这）；
  ③ 负结果/未做项如实登记（TODO），不混进 PASS；
  ④ 预测与结论分离：理想极限的"不改拓扑"是理想平流的**数学推论**，
     本门验的是**离散化体检**，不许写成发现。
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS: list[dict] = []


def check(name: str, ok: bool, detail=None) -> bool:
    RESULTS.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail is not None else ""))
    return bool(ok)


def main() -> int:
    rep = ROOT / "artifacts" / "s3_nonlocality" / "report.json"
    if not rep.exists():
        print("FAIL: 缺 artifacts/s3_nonlocality/report.json（先跑 make s3 里的实验）")
        return 1
    d = json.loads(rep.read_text())
    c = d["S3-1_理想操作"]
    ev = d["S3-2_重联事件"]
    tbl = d["S3-3_配置表v1"]

    st = d.get("S3-0_仪器自检", {})
    print("=== S3 门禁：判决实验 ===")
    check("S3-0: **仪器自检**（合成数据）—— 分离=2 / 重合=1 / 抹开仍=2 / 跨周期边界=2 / 并查集单测=3；"
          "防「门永远绿」的坏仪器（本步踩过两次）",
          st.get("自检通过") is True
          and st.get("分离(sep=2.4)") == 2 and st.get("重合(sep=0.25)") == 1
          and st.get("抹开 1 次(sep=2.4)") == 2 and st.get("跨边界（岛心在 x=0）") == 2
          and st.get("并查集单测（应为 3）") == 3,
          {k: v for k, v in st.items() if k != "交叉核对（局部极大计数）分离/重合"})
    print(f"  理想极限：N 分量 初 = {c['N 初']}（阈值 {c.get('阈值')}）；"
          f"违例 = {c['违例（理想操作改了 N 分量）']}；发散行 = {c['发散行数']}")
    for r in c["行"]:
        print(f"    {r['操作']:28s} N: {r['N_分量 初']} → {r['N_分量 末']}"
              f"  桥接比 末 {r['桥接比 末']}  峰值/阈值 {r['峰值/阈值 末']}"
              + ("  ⚠发散" if r["发散"] else ""))
    print(f"  重联：N {ev['N 初']} → {ev['N 末']}；首次跳变 t = {ev['首次跳变时刻']}；"
          f"欧姆耗散 = {ev['跳变时累积欧姆耗散（代码单位）']}")

    check("S3-1: 理想极限下局部/全局平滑操作**都不改** N_islands（违例 = 0）"
          "—— 注意这是理想平流的数学推论，本门只验离散化体检",
          c["违例（理想操作改了 N 分量）"] == 0 and c["发散行数"] == 0,
          {"违例": c["违例（理想操作改了 N 分量）"], "发散行": c["发散行数"]})

    verdict = ("干净合并窗口（N: 2 → 1）" if ev["N 末"] == 1 else
               "衰减消失（N → 0，合并与衰减未分离）" if ev["N 末"] == 0 else
               "未合并（N 保持 2）" if ev["N 末"] == 2 else "发散")
    print(f"  S3-2 判定：{verdict}")
    check("S3-2: 拓扑改变通道的**状态已量化并如实记录**（不假装已解决）："
          "N 序列/桥接比序列/峰值阈值比序列/欧姆耗散都在产物里",
          ev["N 初"] == 2 and ev["发散"] is False
          and len(ev["N 序列"]) >= 5 and len(ev["桥接比序列"]) >= 5
          and len(ev["峰值/阈值序列"]) >= 5,
          {"判定": verdict, "N 序列": ev["N 序列"][:8], "末桥接比": ev["桥接比序列"][-1]})

    check("S3-3: 初始构型的离散量 = 2（两岛），且交叉核对计数（局部极大）也为正",
          c["N 初"] == 2 and c["行"][0]["N_局部极大 末"] >= 1,
          {"主判据（连通分量）": c["N 初"], "交叉核对（局部极大 末）": c["行"][0]["N_局部极大 末"]})

    has_event = isinstance(ev["跳变时累积欧姆耗散（代码单位）"], (int, float))
    check("S3-4: 阈值代价的状态如实记录 —— 出现干净合并 ⟹ 必须给出那一刻的累积欧姆耗散；"
          "未出现 ⟹ 记为**未量化**（不假装），但总耗散必须可查",
          (has_event and ev["跳变时累积欧姆耗散（代码单位）"] > 0)
          or (not has_event and ev["总欧姆耗散（代码单位）"] > 0),
          {"跳变时的欧姆耗散": ev["跳变时累积欧姆耗散（代码单位）"],
           "总欧姆耗散": ev["总欧姆耗散（代码单位）"],
           "判定": "已量化" if has_event else "**未量化**（S3-2 未出现干净合并 ⟹ 下一步：更窄电阻带 / 更小 sep）"})

    check("S3-5: 配置表 v1 给出 μ 的**离散**档（含 μ=0 与预测档 μ=1），且口径标为模型替身",
          len(tbl["行"]) >= 3 and any(abs(r["μ"] - 1.0) < 1e-12 for r in tbl["行"])
          and any(abs(r["μ"]) < 1e-12 for r in tbl["行"])
          and "模型替身" in tbl["口径"],
          {"行数": len(tbl["行"]), "μ 档": [r["μ"] for r in tbl["行"]]})

    check("S3-6: 证伪判定 —— 理想极限下五个操作都没改离散拓扑量 ⟹ 上游「非局部」支在"
          "**本 2-D 替身**里未被证伪（若违例 > 0 则本门红，那就是证伪结果；"
          "另注意本不变性在 2-D 是理想平流的推论）",
          c["违例（理想操作改了 N 分量）"] == 0, {"违例": c["违例（理想操作改了 N 分量）"]})

    check("S3-7: 自适应步长留痕（步数 > 0 且最小有效 dt > 0，防止用固定 dt 掩盖失稳）",
          all(r["步数"] > 0 and r["最小有效 dt"] > 0 for r in c["行"])
          and ev["步数"] > 0 and ev["最小有效 dt"] > 0,
          {"理想行最小 dt": min(r["最小有效 dt"] for r in c["行"]),
           "重联最小 dt": ev["最小有效 dt"]})

    todo = d.get("TODO", [])
    print("TODO（未做）: " + "；".join(todo))
    check("S3-8: 未做项登记 ≥ 4 条（带符号构型 / 3-D / 量纲归一化 / η 扫描）",
          len(todo) >= 4, len(todo))

    ok = all(r["pass"] for r in RESULTS)
    print(f"\nRESULT: {'ALL PASS' if ok else 'FAIL'}  ({len(RESULTS)} 门)")
    (ROOT / "artifacts" / "s3_nonlocality" / "gates.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
