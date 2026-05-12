"""
Unified juvenile SEDF over age — Stage B aggregator.

Reads per-animal results produced by Juvenile_SEDF_final.py with --variant f
and --term_set v2 (Stage A), then summarizes how each parameter of the fixed
13-term unified equation evolves across age.

Input layout (assumed):
    {root}/{N}weeks/{Animal}/{out_root}/{Animal}_variant_{vk}_weights.csv
    {root}/{N}weeks/{Animal}/{out_root}/{Animal}_variant_{vk}_base_params.csv

Outputs (written to --summary_dir, default {root}/unified_summary):
    unified_params_long.csv   — tidy (animal, age_weeks, param, index, value, kind)
    unified_age_curves.csv    — per-parameter smooth-fit coefficients vs age
    unified_age_curves.png    — small-multiples plot (raw + mean±SD + smooth fit)
    unified_equation.txt      — closed-form W_total with w_k(t) substituted

Usage:
    python Unified_age_summary.py
    python Unified_age_summary.py --root . --out_root runs_unified --variant f
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AGE_DIR_RE = re.compile(r"^(\d+)\s*weeks?$", re.IGNORECASE)


def discover_animals(root: str, out_root: str, variant: str
                     ) -> List[Tuple[int, str, str, str]]:
    """Return list of (age_weeks, animal, weights_csv, base_csv) tuples."""
    found = []
    for entry in sorted(os.listdir(root)):
        m = AGE_DIR_RE.match(entry)
        if not m:
            continue
        age_weeks = int(m.group(1))
        age_dir = os.path.join(root, entry)
        if not os.path.isdir(age_dir):
            continue
        for animal in sorted(os.listdir(age_dir)):
            animal_dir = os.path.join(age_dir, animal)
            if not os.path.isdir(animal_dir):
                continue
            run_dir = os.path.join(animal_dir, out_root)
            if not os.path.isdir(run_dir):
                continue
            # Prefer filenames matching the parent folder name; fall back to any
            # `*_variant_{variant}_weights.csv` since some animals' data live in
            # a differently-named subfolder (e.g. Oreon → data/Orion/, Sagittarius
            # → data/Original/), which causes Juvenile_SEDF_final.py to write
            # files prefixed with the data-subfolder name rather than the parent.
            w_csv_pref = os.path.join(run_dir, f"{animal}_variant_{variant}_weights.csv")
            b_csv_pref = os.path.join(run_dir, f"{animal}_variant_{variant}_base_params.csv")
            if os.path.isfile(w_csv_pref) and os.path.isfile(b_csv_pref):
                found.append((age_weeks, animal, w_csv_pref, b_csv_pref))
                continue
            w_glob = sorted(glob.glob(os.path.join(run_dir, f"*_variant_{variant}_weights.csv")))
            b_glob = sorted(glob.glob(os.path.join(run_dir, f"*_variant_{variant}_base_params.csv")))
            if len(w_glob) == 1 and len(b_glob) == 1:
                found.append((age_weeks, animal, w_glob[0], b_glob[0]))
                actual = os.path.basename(w_glob[0]).replace(f"_variant_{variant}_weights.csv", "")
                print(f"  [match] {age_weeks}w/{animal}: using files prefixed '{actual}'")
            else:
                missing = []
                if not w_glob:
                    missing.append(f"*_variant_{variant}_weights.csv")
                if not b_glob:
                    missing.append(f"*_variant_{variant}_base_params.csv")
                print(f"  [skip] {age_weeks}w/{animal}: missing {missing}")
    return found


def build_long_table(items: List[Tuple[int, str, str, str]]) -> pd.DataFrame:
    """Concatenate per-animal weights+base CSVs into a tidy long table."""
    rows = []
    for age, animal, w_csv, b_csv in items:
        wdf = pd.read_csv(w_csv)
        for _, r in wdf.iterrows():
            name = str(r["term_name"])
            rows.append(dict(
                animal=animal, age_weeks=age,
                param=f"w[{name}]", index="",
                value=float(r["w"]), kind="cross_weight",
            ))
        bdf = pd.read_csv(b_csv)
        for _, r in bdf.iterrows():
            rows.append(dict(
                animal=animal, age_weeks=age,
                param=str(r["param"]),
                index=("" if pd.isna(r["index"]) else str(r["index"])),
                value=float(r["value"]), kind="base_param",
            ))
    return pd.DataFrame(rows)


def fit_age_curve(ages: np.ndarray, vals: np.ndarray) -> Dict:
    """Compare constant / linear / quadratic fits by AIC; return best.

    AIC = n·ln(SSR/n) + 2k.  k counts coefficients only (not σ²) — fine for
    relative comparison among nested polynomial models.
    """
    n = ages.size
    if n < 2:
        return dict(model="constant", coeffs=[float(vals.mean()) if n else 0.0],
                    aic=float("nan"), r2=float("nan"), n=int(n))

    candidates = []
    max_deg = min(2, n - 1)  # need n > deg
    for deg in range(0, max_deg + 1):
        c = np.polyfit(ages, vals, deg)
        pred = np.polyval(c, ages)
        ssr = float(np.sum((vals - pred) ** 2))
        k = deg + 1
        aic = n * math.log(ssr / n + 1e-30) + 2 * k
        ss_tot = float(np.sum((vals - vals.mean()) ** 2))
        r2 = 1.0 - ssr / ss_tot if ss_tot > 0 else float("nan")
        candidates.append((aic, deg, c.tolist(), r2))
    aic, deg, coeffs, r2 = min(candidates, key=lambda x: x[0])
    name = {0: "constant", 1: "linear", 2: "quadratic"}[deg]
    return dict(model=name, coeffs=coeffs, aic=aic, r2=r2, n=int(n))


def bootstrap_band(ages: np.ndarray, vals: np.ndarray,
                   t_grid: np.ndarray, deg: int,
                   n_boot: int = 500, seed: int = 0,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2.5/97.5 quantile band of the polynomial fit on t_grid."""
    rng = np.random.default_rng(seed)
    n = ages.size
    if n < deg + 1 or n_boot <= 0:
        return np.full_like(t_grid, np.nan), np.full_like(t_grid, np.nan)
    preds = np.empty((n_boot, t_grid.size))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            c = np.polyfit(ages[idx], vals[idx], deg)
            preds[b] = np.polyval(c, t_grid)
        except (np.linalg.LinAlgError, ValueError):
            preds[b] = np.nan
    lo = np.nanpercentile(preds, 2.5, axis=0)
    hi = np.nanpercentile(preds, 97.5, axis=0)
    return lo, hi


def summarize_per_param(df_long: pd.DataFrame) -> pd.DataFrame:
    """Per-parameter age-curve fit and per-timepoint stats."""
    rows = []
    df_long = df_long.copy()
    df_long["param_full"] = df_long["param"] + df_long["index"].apply(
        lambda s: f"[{s}]" if s != "" else ""
    )
    for (param, kind), g in df_long.groupby(["param_full", "kind"]):
        ages = g["age_weeks"].to_numpy(dtype=float)
        vals = g["value"].to_numpy(dtype=float)
        fit = fit_age_curve(ages, vals)
        per_age = (g.groupby("age_weeks")["value"]
                     .agg(["mean", "std", "count"]).reset_index())
        rows.append(dict(
            param=param, kind=kind, n=int(len(g)),
            n_ages=int(per_age.shape[0]),
            best_model=fit["model"], r2_age=fit["r2"], aic_age=fit["aic"],
            coeffs=json.dumps(fit["coeffs"]),
            per_age_json=per_age.to_json(orient="records"),
        ))
    return pd.DataFrame(rows)


def plot_age_curves(df_long: pd.DataFrame, df_summary: pd.DataFrame,
                    out_path: str, kind_filter: str = "cross_weight",
                    n_boot: int = 500, seed: int = 0):
    """Small-multiples plot for one parameter kind (cross_weight by default)."""
    df_long = df_long.copy()
    df_long["param_full"] = df_long["param"] + df_long["index"].apply(
        lambda s: f"[{s}]" if s != "" else ""
    )
    sel = df_summary[df_summary["kind"] == kind_filter].copy()
    if sel.empty:
        print(f"  [plot_age_curves] no params of kind={kind_filter}; skip")
        return

    params = sel["param"].tolist()
    n = len(params)
    ncol = min(4, n)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                              constrained_layout=True, squeeze=False)

    age_min = df_long["age_weeks"].min()
    age_max = df_long["age_weeks"].max()
    t_grid = np.linspace(age_min, age_max, 100)

    for i, param in enumerate(params):
        ax = axes[i // ncol, i % ncol]
        sub = df_long[df_long["param_full"] == param]
        ages = sub["age_weeks"].to_numpy(dtype=float)
        vals = sub["value"].to_numpy(dtype=float)
        ax.scatter(ages, vals, s=18, alpha=0.55, color="#1f77b4",
                   label="per animal")
        per_age = sub.groupby("age_weeks")["value"].agg(["mean", "std"]).reset_index()
        ax.errorbar(per_age["age_weeks"], per_age["mean"],
                    yerr=per_age["std"].fillna(0.0),
                    fmt="o", color="#000000", capsize=3, ms=5,
                    label="mean ± SD")
        row = sel[sel["param"] == param].iloc[0]
        coeffs = json.loads(row["coeffs"])
        deg = len(coeffs) - 1
        if deg >= 0:
            ax.plot(t_grid, np.polyval(coeffs, t_grid),
                    color="#d62728", lw=1.6,
                    label=f"{row['best_model']} (R²={row['r2_age']:.2f})")
            if deg >= 1:
                lo, hi = bootstrap_band(ages, vals, t_grid, deg,
                                        n_boot=n_boot, seed=seed + i)
                ax.fill_between(t_grid, lo, hi, color="#d62728",
                                alpha=0.15, label="95% boot CI")
        ax.set_title(param, fontsize=9)
        ax.set_xlabel("age (weeks)")
        ax.set_ylabel("value")
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for j in range(n, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")

    fig.suptitle(f"Unified-equation parameters vs age  ({kind_filter})",
                 fontsize=13, weight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def write_unified_equation_txt(df_summary: pd.DataFrame, out_path: str,
                               term_set_name: str):
    """Write closed-form W_total with w_k(t) substituted by smooth fits."""
    cw = df_summary[df_summary["kind"] == "cross_weight"].copy()
    with open(out_path, "w") as f:
        f.write("Unified juvenile SEDF — closed-form across age\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Term set: {term_set_name}\n\n")
        f.write("W_total(λ; t) = W_iso_poly(λ) + W_aniso_4fiber(λ; rec, disp)\n")
        f.write("                + Σ_k  w_k(t) · ψ_k(λ)\n\n")
        f.write("Cross-term weights as functions of age t (weeks):\n")
        f.write("-" * 60 + "\n")
        for _, r in cw.iterrows():
            coeffs = json.loads(r["coeffs"])
            poly_str = _fmt_poly(coeffs)
            f.write(f"  {r['param']:32s} = {poly_str}     "
                    f"[{r['best_model']}, R²={r['r2_age']:.3f}, n={r['n']}]\n")
        f.write("\nBase parameters (per-timepoint mean shown — full per-animal\n")
        f.write("values in unified_params_long.csv):\n")
        f.write("-" * 60 + "\n")
        bp = df_summary[df_summary["kind"] == "base_param"]
        for _, r in bp.iterrows():
            coeffs = json.loads(r["coeffs"])
            f.write(f"  {r['param']:42s} {r['best_model']:9s}  "
                    f"R²(age)={r['r2_age']:.3f}  "
                    f"coeffs={[round(c, 5) for c in coeffs]}\n")
    print(f"Saved {out_path}")


def _fmt_poly(coeffs: List[float]) -> str:
    """np.polyfit returns coeffs highest-deg first; format as a·t² + b·t + c."""
    deg = len(coeffs) - 1
    if deg < 0:
        return "0"
    parts = []
    for i, c in enumerate(coeffs):
        p = deg - i
        if p == 0:
            parts.append(f"{c:+.5g}")
        elif p == 1:
            parts.append(f"{c:+.5g}·t")
        else:
            parts.append(f"{c:+.5g}·t^{p}")
    return " ".join(parts).lstrip("+ ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Single_fitting root containing the {N}weeks/ subfolders.")
    ap.add_argument("--out_root", default="runs_unified",
                    help="Per-animal output directory holding the variant CSVs.")
    ap.add_argument("--variant", default="f", choices=["f", "g"])
    ap.add_argument("--summary_dir", default=None,
                    help="Where to write the aggregate outputs. "
                         "Default: {root}/unified_summary.")
    ap.add_argument("--term_set", default="v2",
                    help="Label written into unified_equation.txt header.")
    ap.add_argument("--n_boot", type=int, default=500,
                    help="Bootstrap iterations for the age-curve CI (0 disables).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    summary_dir = args.summary_dir or os.path.join(root, "unified_summary")
    os.makedirs(summary_dir, exist_ok=True)

    items = discover_animals(root, args.out_root, args.variant)
    if not items:
        raise SystemExit(
            f"No per-animal {args.out_root}/<animal>_variant_{args.variant}_*.csv "
            f"files found under {root}. Run Stage A first."
        )
    print(f"Found {len(items)} animal(s) across "
          f"{len(set(a for a, *_ in items))} timepoint(s).")

    df_long = build_long_table(items)
    long_path = os.path.join(summary_dir, "unified_params_long.csv")
    df_long.to_csv(long_path, index=False)
    print(f"Saved {long_path} ({len(df_long)} rows)")

    df_summary = summarize_per_param(df_long)
    curves_path = os.path.join(summary_dir, "unified_age_curves.csv")
    df_summary.to_csv(curves_path, index=False)
    print(f"Saved {curves_path}")

    plot_path = os.path.join(summary_dir, "unified_age_curves.png")
    plot_age_curves(df_long, df_summary, plot_path,
                    kind_filter="cross_weight",
                    n_boot=args.n_boot, seed=args.seed)

    base_plot_path = os.path.join(summary_dir, "unified_age_curves_base.png")
    plot_age_curves(df_long, df_summary, base_plot_path,
                    kind_filter="base_param",
                    n_boot=args.n_boot, seed=args.seed)

    eq_path = os.path.join(summary_dir, "unified_equation.txt")
    write_unified_equation_txt(df_summary, eq_path, term_set_name=args.term_set)

    print("\nDone.")


if __name__ == "__main__":
    main()
