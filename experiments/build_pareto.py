"""build_pareto.py — assemble the quality-vs-working-set Pareto from Phase-2 runs.

Reads every experiments/<exp_id>/metrics.json (trained candidates), adds the analytical/anchored
points (reference U-Net, C4 tiled = reference quality), and emits:
  - experiments/pareto.md   (table, owner-facing)
  - experiments/pareto.png  (SI-SDR vs deployable working set; the Phase-2 gate visual)

The working-set axis is the **deployable** (streamed/tiled) on-chip activation — the same metric the
Phase-1 model validated — so trained quality lands on the exact axis the hardware pays for.
"""
from __future__ import annotations

import sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from pathlib import Path

HERE = Path(__file__).parent

# Anchored (not-trained-here) reference points.
#   Reference / C4: the deployed time-domain U-Net. Quality from PAPER_DATA §B (FPGA INT16 AV) on
#   LRS3; working set = Phase-1 numbers. C4 (tiled) has the SAME quality as the reference (tiling is
#   math-identical) but a far smaller deployable working set.
ANCHORS = [
    # label, si_sdr(dB), pesq, stoi, working_set_MB, note
    ("Reference U-Net (4-bitstream)", 5.46, 1.743, 0.7378, 4.10,
     "deployed baseline; does NOT fit single-config (215% BRAM)"),
    ("C4 Tiled U-Net (= ref quality)", 5.46, 1.743, 0.7378, 0.30,
     "tiling is math-identical to the reference; fits single-config"),
]


def load_runs():
    rows = []
    for mj in sorted(HERE.glob("p2-*/metrics.json")):
        if "quick" in mj.parent.name:
            continue
        d = json.loads(mj.read_text(encoding="utf-8"))
        # Use the BEST-SI-SDR epoch (that is the checkpoint we would deploy, best.pt), not the last.
        hist = d.get("history") or []
        cand = [h for h in hist if h.get("val", {}).get("si_sdr") is not None]
        bestv = max(cand, key=lambda h: h["val"]["si_sdr"])["val"] if cand else (d.get("final_val") or {})
        mtw = d["args"].get("max_train_windows")
        rows.append({
            "label": f"{d['model'].upper()} {d.get('model_name','')}".strip(),
            "si_sdr": bestv.get("si_sdr"), "pesq": bestv.get("pesq_wb"), "stoi": bestv.get("stoi"),
            "ws_mb": d.get("deployable_working_set_mb"), "params": d.get("params"),
            "note": f"best of {d['args'].get('epochs')}ep / {'full' if not mtw else str(mtw)+' win'}",
            "exp_id": d["exp_id"],
        })
    return rows


def main() -> int:
    runs = load_runs()
    # markdown
    lines = ["# Phase 2 — quality vs deployable working-set Pareto\n"]
    lines.append("SI-SDR / PESQ / STOI on LRS3 dev windows vs the **deployable** on-chip activation "
                 "working set (MB, audio path; Phase-1-consistent). Anchors are not retrained here.\n")
    lines.append("| candidate | SI-SDR (dB) | PESQ-WB | STOI | working set (MB) | params | note |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    def fmt(v, d=2): return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
    for a in ANCHORS:
        lines.append(f"| {a[0]} | {fmt(a[1])} | {fmt(a[2],3)} | {fmt(a[3],3)} | {a[4]:.2f} | — | {a[5]} |")
    for r in runs:
        lines.append(f"| {r['label']} ({r['exp_id']}) | {fmt(r['si_sdr'])} | {fmt(r['pesq'],3)} | "
                     f"{fmt(r['stoi'],3)} | {fmt(r['ws_mb'],3)} | {r['params']:,} | {r['note']} |")
    if not runs:
        lines.append("\n_(no trained runs found yet — run `python -m avse.train ...` first)_")
    (HERE / "pareto.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {HERE/'pareto.md'}")

    # plot (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for a in ANCHORS:
            ax.scatter(a[4], a[1], marker="s", s=80, color="#888")
            ax.annotate(a[0], (a[4], a[1]), fontsize=8, xytext=(5, 5), textcoords="offset points")
        for r in runs:
            if r["si_sdr"] is None or r["ws_mb"] is None:
                continue
            ax.scatter(r["ws_mb"], r["si_sdr"], marker="o", s=90, color="#1f77b4")
            ax.annotate(r["label"].split()[0], (r["ws_mb"], r["si_sdr"]), fontsize=9,
                        xytext=(5, -10), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("deployable on-chip working set (MB, audio path) — log scale")
        ax.set_ylabel("SI-SDR (dB)")
        ax.set_title("Phase 2 Pareto: enhancement quality vs single-config working set")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(HERE / "pareto.png", dpi=130)
        print(f"Wrote {HERE/'pareto.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
