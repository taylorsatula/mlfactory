"""Required CausalGraph MVP plots from preserved per-example outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import fit_depth_logistic, raw_depth_stats


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _curve(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    fit = fit_depth_logistic(records); xs = np.linspace(min(r["depth"] for r in records), max(r["depth"] for r in records), 200)
    if fit.get("beta1") is None: return xs, np.full_like(xs, .5)
    p = 1 / (1 + np.exp(-(fit["beta0"] + fit["beta1"] * xs)))
    return xs, p


def _plot_curves(ax: Any, items: list[tuple[str, list[dict[str, Any]]]]) -> None:
    for label, records in items:
        stats = raw_depth_stats(records); x, y = _curve(records)
        ax.plot(x, y, label=label)
        ax.errorbar([r["depth"] for r in stats], [r["accuracy"] for r in stats], yerr=[[r["accuracy"] - r["wilson_low"] for r in stats], [r["wilson_high"] - r["accuracy"] for r in stats]], fmt=".", alpha=.45)
    ax.set_ylim(-.03, 1.03); ax.set_xlabel("dependency depth"); ax.set_ylabel("accuracy"); ax.grid(alpha=.2); ax.legend()


def make_plots(run: Path) -> None:
    import matplotlib.pyplot as plt
    out = run / "plots"; out.mkdir(exist_ok=True)
    baseline = _read(run / "model_outputs" / "baseline_sealed.jsonl")
    target_path = run / "model_outputs" / "targeted_final_sealed.jsonl"; random_path = run / "model_outputs" / "random_final_sealed.jsonl"
    targeted = _read(target_path) if target_path.exists() else []
    random = _read(random_path) if random_path.exists() else []
    fig, ax = plt.subplots(figsize=(10, 6)); items = [("BASELINE", baseline)]
    if targeted: items.append(("TARGETED final", targeted))
    if random: items.append(("RANDOM final", random))
    _plot_curves(ax, items); fig.tight_layout(); fig.savefig(out / "competence_curves.png", dpi=150); plt.close(fig)
    if targeted and random:
        fig, ax = plt.subplots(figsize=(10, 6)); _plot_curves(ax, [("TARGETED", targeted), ("RANDOM", random)]); fig.tight_layout(); fig.savefig(out / "target_vs_control.png", dpi=150); plt.close(fig)
    # Frontier progression from saved development fits.
    fig, ax = plt.subplots(figsize=(9, 5))
    for branch, color in [("targeted", "tab:blue"), ("random", "tab:orange"), ("depth_matched", "tab:green")]:
        xs=[]; ys=[]
        for path in sorted((run / "frontier_fits").glob(f"{branch}_round_*.json")):
            fit=json.loads(path.read_text())["frontier"]; xs.append(int(path.stem.rsplit("_",1)[-1])); ys.append(fit.get("d50"))
        if ys: ax.plot(xs, ys, marker="o", label=branch.upper(), color=color)
    basefit=fit_depth_logistic(baseline); ax.axhline(basefit.get("d50", np.nan), linestyle="--", color="black", label="baseline sealed")
    ax.set_xlabel("adaptive round"); ax.set_ylabel("d50"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out / "frontier_progression.png", dpi=150); plt.close(fig)
    # Falloff bands.
    fig, ax = plt.subplots(figsize=(9, 5)); labels=[]; lows=[]; mids=[]; highs=[]
    for label, records in items:
        f=fit_depth_logistic(records); labels.append(label); lows.append(f.get("d80", np.nan)); mids.append(f.get("d50", np.nan)); highs.append(f.get("d20", np.nan))
    xpos=np.arange(len(labels)); ax.vlines(xpos, lows, highs); ax.scatter(xpos, mids, color="black", zorder=3); ax.set_xticks(xpos, labels, rotation=20); ax.set_ylabel("depth"); ax.set_title("falloff band [d80, d20]"); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(out / "falloff_band.png", dpi=150); plt.close(fig)
    # Acquisition movement.
    summary=json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}; acquisition=summary.get("acquisition", {})
    fig, ax = plt.subplots(figsize=(9, 5))
    for branch, values in acquisition.items(): ax.plot([v["round"] for v in values], [v.get("depth_median", np.nan) for v in values], marker="o", label=branch)
    ax.set_xlabel("adaptive round"); ax.set_ylabel("acquired depth median"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out / "acquisition_movement.png", dpi=150); plt.close(fig)
    # World transfer.
    fig, ax = plt.subplots(figsize=(10, 6))
    def split(rows): return ([r for r in rows if r["world"] in {"greenhouse","factory","security","household"}], [r for r in rows if r["world"] in {"warehouse","computer"}])
    for label, rows in [("BASELINE", baseline), ("TARGETED final", targeted)]:
        if not rows: continue
        train, held = split(rows); _plot_curves(ax, [(label + " train", train), (label + " held-out", held)])
    fig.tight_layout(); fig.savefig(out / "world_transfer.png", dpi=150); plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("run"); make_plots(Path(p.parse_args().run).resolve())


if __name__ == "__main__": main()
