#!/usr/bin/env python3
"""
Refit CombinedEquationSEDF per animal using the Morris -> Sobol reduced
term set produced by morris_sobol_workflow.py, and plot, for each animal:

    fig1  sigma_theta vs lambda_theta   (all PD sweeps, exp + k=5/8/10 model)
    fig2  sigma_z     vs lambda_z       (all FL sweeps, exp + k=5/8/10 model)

The Morris ranking includes base-quadratic iso terms (b_th E_th^2, b_z E_z^2,
b_thz <E_th><E_z>) that are part of DecomposedSEDFBase, not tweak terms;
those are filtered out so the top-k subsets contain symbolic cross terms only.
"""
from __future__ import annotations
import argparse, importlib.util, os, re
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd


def import_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def is_base_name(n: str) -> bool:
    """Identify base-quadratic / fiber-family terms parsed by sensitivity_with_base."""
    return n.startswith("b_") or n.startswith("k1")


def build_subsets(morris_csv: Path, ks=(5, 8, 10)) -> Dict[int, List[str]]:
    df = pd.read_csv(morris_csv)
    df = df.sort_values("mu_star", ascending=False)
    cross_only = [n for n in df["name"].tolist() if not is_base_name(n)]
    return {k: cross_only[:k] for k in ks}


# --------------------------------------------------------------------------
# Two-figure-per-animal stress-stretch plot (overlaid sweeps)
# --------------------------------------------------------------------------
def _level(curve_id) -> int:
    m = re.search(r"(\d+)", str(curve_id))
    return int(m.group(1)) if m else 0


def plot_per_direction(metrics_df: pd.DataFrame, preds: Dict,
                       subsets: Dict[int, List[str]], out_dir: Path,
                       mode_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "refit_figures_per_direction"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ks = sorted(subsets.keys())
    style_pool = ["-", "--", ":", "-."]
    k_styles = {k: style_pool[i % len(style_pool)] for i, k in enumerate(ks)}
    cmap = plt.get_cmap("tab10")

    for (age, animal), bundle in preds.items():
        if not bundle["by_k"]:
            continue
        Xth, yth, ids_th = bundle["Xth"], bundle["yth"], bundle["ids_th"]
        Xz,  yz,  ids_z  = bundle["Xz"],  bundle["yz"],  bundle["ids_z"]

        # ---- σθ vs λθ ----
        if Xth.size:
            pd_ids = sorted(set(ids_th.tolist()), key=_level)
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for ci, cid in enumerate(pd_ids):
                color = cmap(ci % 10)
                mask = (ids_th == cid)
                x = Xth[mask, 0]; order = np.argsort(x)
                xs = x[order]
                ax.scatter(xs, yth[mask][order], s=30,
                           facecolors="none", edgecolors=color,
                           linewidths=1.2,
                           label=f"{cid} exp", zorder=4)
                for k in ks:
                    pk = bundle["by_k"].get(k)
                    if pk is None or pk["sth"].size == 0:
                        continue
                    ax.plot(xs, pk["sth"][mask][order],
                            color=color, lw=1.6,
                            linestyle=k_styles[k], alpha=0.9,
                            label=f"{cid} k={k}" if ci == 0 else None,
                            zorder=3)
            ax.set_xlabel(r"$\lambda_\theta$")
            ax.set_ylabel(r"$\sigma_\theta$ [kPa]")
            ax.set_title(f"{age} / {animal} — σθ vs λθ ({mode_label})")
            ax.grid(alpha=0.3)
            # Compact dual legend: curves (color) + k (linestyle)
            curve_handles = [plt.Line2D([], [], marker="o",
                              markerfacecolor="none",
                              markeredgecolor=cmap(i % 10),
                              linestyle="-", color=cmap(i % 10),
                              label=cid) for i, cid in enumerate(pd_ids)]
            k_handles = [plt.Line2D([], [], color="0.2",
                          linestyle=k_styles[k], lw=1.6,
                          label=f"k={k}") for k in ks]
            leg1 = ax.legend(handles=curve_handles, loc="upper left",
                             fontsize=8, title="curve")
            ax.add_artist(leg1)
            ax.legend(handles=k_handles, loc="lower right", fontsize=8,
                      title="model")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{age}_{animal}_sigma_theta.png", dpi=140)
            plt.close(fig)

        # ---- σz vs λz ----
        if Xz.size:
            fl_ids = sorted(set(ids_z.tolist()), key=_level)
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for ci, cid in enumerate(fl_ids):
                color = cmap(ci % 10)
                mask = (ids_z == cid)
                x = Xz[mask, 1]; order = np.argsort(x)
                xs = x[order]
                ax.scatter(xs, yz[mask][order], s=30,
                           facecolors="none", edgecolors=color,
                           linewidths=1.2,
                           label=f"{cid} exp", zorder=4)
                for k in ks:
                    pk = bundle["by_k"].get(k)
                    if pk is None or pk["sz"].size == 0:
                        continue
                    ax.plot(xs, pk["sz"][mask][order],
                            color=color, lw=1.6,
                            linestyle=k_styles[k], alpha=0.9,
                            zorder=3)
            ax.set_xlabel(r"$\lambda_z$")
            ax.set_ylabel(r"$\sigma_z$ [kPa]")
            ax.set_title(f"{age} / {animal} — σz vs λz ({mode_label})")
            ax.grid(alpha=0.3)
            curve_handles = [plt.Line2D([], [], marker="o",
                              markerfacecolor="none",
                              markeredgecolor=cmap(i % 10),
                              linestyle="-", color=cmap(i % 10),
                              label=cid) for i, cid in enumerate(fl_ids)]
            k_handles = [plt.Line2D([], [], color="0.2",
                          linestyle=k_styles[k], lw=1.6,
                          label=f"k={k}") for k in ks]
            leg1 = ax.legend(handles=curve_handles, loc="upper left",
                             fontsize=8, title="curve")
            ax.add_artist(leg1)
            ax.legend(handles=k_handles, loc="lower right", fontsize=8,
                      title="model")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{age}_{animal}_sigma_z.png", dpi=140)
            plt.close(fig)

    print(f"      Wrote per-direction figures for {len(preds)} animals.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--workflow_dir", default="cross_age_morris_sobol_workflow",
                    help="Directory containing morris_screening.csv from morris_sobol_workflow.py")
    ap.add_argument("--out_dir", default="refit_morris_sobol",
                    help="Where to write per-animal RMSE + figures.")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 8, 10])
    ap.add_argument("--epochs_base", type=int, default=2000)
    ap.add_argument("--epochs_tweak", type=int, default=1000)
    ap.add_argument("--epochs_full", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load helper module with refit_combined ---
    cts = import_sibling("cts", root / "compile_terms_and_sensitivity.py")

    morris_csv = root / args.workflow_dir / "morris_screening.csv"
    if not morris_csv.exists():
        raise SystemExit(f"Missing {morris_csv}. Run morris_sobol_workflow.py first.")

    subsets = build_subsets(morris_csv, ks=tuple(args.ks))
    with open(out_dir / "subsets_used.txt", "w", encoding="utf-8") as f:
        f.write(f"Source: {morris_csv}\n")
        f.write("Filter: symbolic cross terms only (base iso/fiber terms removed)\n\n")
        for k, terms in subsets.items():
            f.write(f"k={k}: " + ", ".join(terms) + "\n")
    print("[1/3] Subsets built from Morris ranking (cross terms only):")
    for k, terms in subsets.items():
        print(f"      k={k}: {terms}")

    J = cts.import_symbolic_module(root)
    specimens = cts.discover_specimens(root)

    print(f"[2/3] Refitting CombinedEquationSEDF per animal "
          f"(epochs base={args.epochs_base}, tweak={args.epochs_tweak}) ...")
    metrics, preds = cts.refit_combined(
        J, specimens, subsets,
        epochs_base=args.epochs_base,
        epochs_tweak=args.epochs_tweak,
        epochs_full=args.epochs_full,
    )

    if metrics.empty:
        raise SystemExit("No animals refit successfully.")

    metrics.to_csv(out_dir / "refit_metrics.csv", index=False)
    summary = (metrics.groupby("k")
               .agg(n_animals=("animal", "nunique"),
                    mean_rmse_theta=("rmse_theta", "mean"),
                    median_rmse_theta=("rmse_theta", "median"),
                    mean_rmse_z=("rmse_z", "mean"),
                    median_rmse_z=("rmse_z", "median"),
                    mean_rmse_combined=("rmse_combined", "mean"),
                    median_rmse_combined=("rmse_combined", "median"))
               .reset_index())
    summary.to_csv(out_dir / "refit_summary.csv", index=False)
    with open(out_dir / "refit_summary.txt", "w", encoding="utf-8") as f:
        f.write("CombinedEquationSEDF refit on Morris-Sobol reduced subsets\n")
        f.write("=" * 64 + "\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n")
    print("[3/3] Plotting σθ vs λθ and σz vs λz per animal ...")
    plot_per_direction(metrics, preds, subsets, out_dir,
                       mode_label="Morris→Sobol")

    print(f"\nAll outputs in: {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
