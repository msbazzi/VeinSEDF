#!/usr/bin/env python3
"""
Compile final-equation terms across all juvenile specimens, run a local
contribution analysis per animal, run Sobol-index reduction in three
bound regimes side-by-side, and (optionally) refit CombinedEquationSEDF
per-animal restricted to the top-k subset to validate the reduction.

Usage:
    python compile_terms_and_sensitivity.py \
        [--root .] [--out_dir cross_age_sensitivity] \
        [--n_grid 20] [--sobol_N 1024] \
        [--refit_subsets] [--refit_mode raw_uniform] \
        [--refit_ks 5 8 10] [--refit_epochs_base 2000] \
        [--refit_epochs_tweak 1000]

Outputs (under --out_dir):
    term_compilation.csv             one row per (age, animal, term)
    term_frequency.csv               specimens that selected each term
    local_sensitivity.csv            per-animal ranking by |coeff_raw|*RMS(psi)
    sobol_indices_<mode>.csv         per Sobol mode {coeff_max, normalized, raw_uniform}
    reduced_term_set_comparison.txt  side-by-side top-10 from each mode
    [if --refit_subsets]
    combined_refit_metrics.csv       per-animal RMSE for each top-k subset
    combined_refit_summary.txt       cross-animal mean/median RMSE per k

Requirements: torch, numpy, pandas, SALib (`pip install SALib`).
"""
from __future__ import annotations
import argparse, os, re, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

ROW_RE = re.compile(
    r"^\s*\[(?P<bucket>\w+)\s*\]\s*"
    r"(?P<name>.+?)\s+\(idx\s*(?P<idx>\d+)\):\s*"
    r"gate=\s*(?P<gate>[-+0-9.eE]+),\s*"
    r"eff_w_norm=\s*(?P<eff_w>[-+0-9.eE]+),\s*"
    r"coeff_raw=\s*(?P<coeff>[-+0-9.eE]+),\s*"
    r"sigma=\s*(?P<sigma>[-+0-9.eE]+)\s*$"
)


# --------------------------------------------------------------------------
# Step 1: parse all final equations into a master table
# --------------------------------------------------------------------------
def parse_equation_file(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m:
                continue
            d = m.groupdict()
            rows.append({
                "bucket":     d["bucket"].strip(),
                "name":       d["name"].strip(),
                "idx":        int(d["idx"]),
                "gate":       float(d["gate"]),
                "eff_w_norm": float(d["eff_w"]),
                "coeff_raw":  float(d["coeff"]),
                "sigma":      float(d["sigma"]),
            })
    return rows


def discover_specimens(root: Path) -> List[Tuple[str, str, Path]]:
    """Return (age, animal, equation_file_path) tuples."""
    out = []
    for age_dir in sorted(root.glob("*weeks")):
        if not age_dir.is_dir():
            continue
        age = age_dir.name
        for animal_dir in sorted(p for p in age_dir.iterdir() if p.is_dir()):
            eqs = sorted((animal_dir / "runsfinal").glob("*_final_equation.txt"))
            if not eqs:
                continue
            out.append((age, animal_dir.name, eqs[0]))
    return out


def build_compilation(specimens) -> pd.DataFrame:
    records = []
    for age, animal, path in specimens:
        for r in parse_equation_file(path):
            r["age"] = age
            r["animal"] = animal
            r["source"] = str(path)
            records.append(r)
    df = pd.DataFrame(records)
    cols = ["age", "animal", "bucket", "name", "idx",
            "gate", "eff_w_norm", "coeff_raw", "sigma", "source"]
    return df[cols]


# --------------------------------------------------------------------------
# Symbolic library evaluation on a canonical kinematic grid
# --------------------------------------------------------------------------
def import_symbolic_module(root: Path):
    sys.path.insert(0, str(root))
    import Juvenile_SEDF_final as J
    return J


def make_kinematic_grid(n: int = 20,
                        lt_min: float = 0.60, lt_max: float = 1.60,
                        lz_min: float = 1.00, lz_max: float = 1.60) -> torch.Tensor:
    """Canonical biaxial grid covering the experimental envelope.

    Default range covers ~80% of the experimental data while staying
    clear of the fiber-saturation regime (k2*xi^2 -> exp overflow) that
    appears for some animals at lam_theta > 1.7.
    Full data envelope is roughly lam_theta in [0.33, 2.07], lam_z in [1.04, 1.78].
    """
    gt = np.linspace(lt_min, lt_max, n)
    gz = np.linspace(lz_min, lz_max, n)
    LT, LZ = np.meshgrid(gt, gz, indexing="ij")
    LR = 1.0 / (LT * LZ)  # incompressible
    lam = np.stack([LT.ravel(), LZ.ravel(), LR.ravel()], axis=-1)
    return torch.tensor(lam, dtype=torch.float64, requires_grad=True)


def evaluate_basis(J, lam_grid: torch.Tensor) -> Tuple[List[str], np.ndarray]:
    model = J.CombinedEquationSEDF()
    raw = model.symbolic_library_raw(lam_grid)
    names = [name for name, _, _ in raw]
    feats = torch.stack([t for _, t, _ in raw], dim=-1).detach().cpu().numpy()
    return names, feats  # (N_pts, N_terms)


# --------------------------------------------------------------------------
# Step 2: local sensitivity (per-animal contribution proxy)
# --------------------------------------------------------------------------
def local_sensitivity(df: pd.DataFrame,
                      basis_names: List[str],
                      basis_feats: np.ndarray) -> pd.DataFrame:
    rms = {n: float(np.sqrt(np.mean(basis_feats[:, i] ** 2)))
           for i, n in enumerate(basis_names)}
    rec = []
    for (age, animal), g in df.groupby(["age", "animal"], sort=False):
        contribs = []
        for _, row in g.iterrows():
            psi_rms = rms.get(row["name"], np.nan)
            c = abs(row["coeff_raw"]) * psi_rms
            contribs.append((row["name"], row["coeff_raw"], psi_rms, c))
        total = sum(c for *_, c in contribs) or 1.0
        for name, coeff, psi, c in contribs:
            rec.append({
                "age": age, "animal": animal, "name": name,
                "coeff_raw": coeff, "rms_psi": psi,
                "contribution": c, "contribution_frac": c / total,
            })
    out = pd.DataFrame(rec)
    return out.sort_values(["age", "animal", "contribution"],
                           ascending=[True, True, False])


# --------------------------------------------------------------------------
# Step 3: Sobol reduction in three bound regimes
# --------------------------------------------------------------------------
def sobol_reduction(df: pd.DataFrame,
                    basis_names: List[str],
                    basis_feats: np.ndarray,
                    bound_mode: str,
                    N: int = 1024) -> pd.DataFrame:
    """
    bound_mode:
      "coeff_max"   : bounds = +-max|coeff_raw| observed across animals.
                      Surrogate uses raw psi. Outlier-sensitive.
      "normalized"  : bounds = [-1,1] uniform on z-scored psi.
                      Equal-footing; tends to be flat (uninformative).
      "raw_uniform" : bounds = [-1,1] uniform on raw psi.
                      Discriminates by intrinsic basis shape on the grid,
                      independent of historical fit magnitudes.
    Surrogate output Y = RMS over kinematic grid of W(lam) = sum_i w_i*psi_i.
    """
    try:
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
    except ImportError as e:
        raise SystemExit("SALib is required. Install with: pip install SALib") from e

    unique_terms = [t for t in df["name"].unique() if t in basis_names]
    cols = [basis_feats[:, basis_names.index(t)] for t in unique_terms]

    if bound_mode == "coeff_max":
        max_abs = (df.assign(absw=df["coeff_raw"].abs())
                     .groupby("name")["absw"].max())
        bounds = []
        for t in unique_terms:
            w = float(max_abs.get(t, 1.0))
            if w <= 0 or not np.isfinite(w):
                w = 1.0
            bounds.append([-w, w])
        Psi_mat = np.stack(cols, axis=-1)
        feat_mu = [float(c.mean()) for c in cols]
        feat_sd = [float(c.std()) for c in cols]
    elif bound_mode == "normalized":
        bounds = [[-1.0, 1.0]] * len(unique_terms)
        feat_mu = [float(c.mean()) for c in cols]
        feat_sd = [float(c.std()) if c.std() > 0 else 1.0 for c in cols]
        Psi_mat = np.stack(
            [(c - mu) / sd for c, mu, sd in zip(cols, feat_mu, feat_sd)],
            axis=-1,
        )
    elif bound_mode == "raw_uniform":
        bounds = [[-1.0, 1.0]] * len(unique_terms)
        Psi_mat = np.stack(cols, axis=-1)
        feat_mu = [float(c.mean()) for c in cols]
        feat_sd = [float(c.std()) for c in cols]
    else:
        raise ValueError(f"Unknown bound_mode: {bound_mode}")

    problem = {"num_vars": len(unique_terms),
               "names":    unique_terms,
               "bounds":   bounds}

    X = sobol_sample.sample(problem, N, calc_second_order=False)
    W = X @ Psi_mat.T
    Y = np.sqrt((W ** 2).mean(axis=1))

    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                               print_to_console=False)

    obs_w_max = (df.assign(absw=df["coeff_raw"].abs())
                   .groupby("name")["absw"].max())
    obs_n     = df.groupby("name")["animal"].nunique()
    out = pd.DataFrame({
        "name": unique_terms,
        "psi_grid_mean":     feat_mu,
        "psi_grid_std":      feat_sd,
        "n_specimens":       [int(obs_n.get(t, 0)) for t in unique_terms],
        "obs_max_coeff_raw": [float(obs_w_max.get(t, np.nan)) for t in unique_terms],
        "weight_bound":      [b[1] for b in bounds],
        "S1":      Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST":      Si["ST"],
        "ST_conf": Si["ST_conf"],
    }).sort_values("ST", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Step 4 (optional): refit CombinedEquationSEDF per-animal on each top-k
# --------------------------------------------------------------------------
def find_animal_data_dir(animal_dir: Path) -> Optional[Path]:
    data = animal_dir / "data"
    if not data.is_dir():
        return None
    inner = [p for p in data.iterdir() if p.is_dir()]
    return inner[0] if inner else data


def refit_combined(J, specimens: List[Tuple[str, str, Path]],
                   subsets: Dict[int, List[str]],
                   epochs_base: int = 2000,
                   epochs_tweak: int = 1000,
                   epochs_full: int = 0,
                   lr_base: float = 5e-3,
                   lr_tweak: float = 5e-3,
                   seed: int = 0):
    """
    For each animal: fit base once, then refit CombinedEquationSEDF for each
    top-k subset (re-using the base weights), and report RMSE.

    Returns:
      metrics_df : per-(animal, k) RMSE / loss / AIC / BIC
      preds      : dict[(age, animal)] = {
                       "Xth": (N,2), "yth": (N,), "Xz": (N,2), "yz": (N,),
                       "by_k": dict[k] = {"sth": (N,), "sz": (N,)},
                   }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = []
    preds: Dict[Tuple[str, str], Dict] = {}
    for age, animal, eq_path in specimens:
        animal_dir = eq_path.parent.parent
        data_dir = find_animal_data_dir(animal_dir)
        if data_dir is None:
            print(f"  [refit] {animal}: no data/ directory, skipping")
            continue
        print(f"  [refit] {age}/{animal}: parsing data from {data_dir.name}/")
        blob = J.parse_csvs_for_age(str(data_dir))
        Xth, yth, Xz, yz, wth, wz, ids_th, ids_z = J.build_dataset_with_weights(blob)
        if Xth.size == 0 and Xz.size == 0:
            print(f"           no usable PD/FL data, skipping")
            continue

        # Stage 1: fit base model once for this animal
        t0 = time.time()
        base = J.DecomposedSEDFBase(iso_model="poly", aniso_model="4fiber",
                                    use_recruitment=True, use_dispersion=True)
        if hasattr(base.aniso, "dist_type"):
            base.aniso.dist_type = "lognormal"
        try:
            _, _, _, base = J.fit_model(
                base, Xth, yth, Xz, yz,
                epochs=epochs_base, name=f"{animal}-base",
                lr=lr_base, weights_th=wth, weights_z=wz,
                seed=seed, jitter=0.0, jitter_warm_epochs=0,
            )
        except Exception as e:
            print(f"           base fit failed: {e}")
            continue
        base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        t_base = time.time() - t0

        # Pre-compute lam_all for set_stats
        lam_all = J._lam_all_from_X(Xth, Xz, device=device)

        key = (age, animal)
        preds[key] = {
            "Xth": Xth.copy() if Xth.size else np.zeros((0, 2)),
            "yth": yth.copy() if Xth.size else np.zeros((0,)),
            "Xz":  Xz.copy() if Xz.size else np.zeros((0, 2)),
            "yz":  yz.copy() if Xz.size else np.zeros((0,)),
            "ids_th": ids_th if Xth.size else np.array([], object),
            "ids_z":  ids_z  if Xz.size  else np.array([], object),
            "by_k": {},
        }

        for k, terms in subsets.items():
            try:
                model = J.CombinedEquationSEDF(
                    n_basis=45, use_recruitment=True, use_dispersion=True,
                    term_list=terms,
                )
                with torch.no_grad():
                    model.base.load_state_dict(base_state, strict=False)
                if hasattr(model.base.aniso, "dist_type"):
                    model.base.aniso.dist_type = "lognormal"
                with torch.no_grad():
                    model.set_stats(lam_all)
                    for p in model.base.parameters():
                        p.requires_grad_(False)
                    for p in model.tweak.parameters():
                        p.requires_grad_(True)
                t1 = time.time()
                loss, aic, bic, model = J.fit_model(
                    model, Xth, yth, Xz, yz,
                    epochs=epochs_tweak, name=f"{animal}-k{k}-tweak",
                    lr=lr_tweak, weights_th=wth, weights_z=wz,
                    seed=seed,
                )
                if epochs_full > 0:
                    for p in model.parameters():
                        p.requires_grad_(True)
                    loss, aic, bic, model = J.fit_model(
                        model, Xth, yth, Xz, yz,
                        epochs=epochs_full, name=f"{animal}-k{k}-joint",
                        lr=lr_tweak, weights_th=wth, weights_z=wz,
                        seed=seed,
                    )
                t_k = time.time() - t1

                # Compute per-direction RMSE on the actual data
                model.eval()
                rmse_th, rmse_z = float("nan"), float("nan")
                sth_pred = np.zeros((0,))
                sz_pred  = np.zeros((0,))
                if Xth.size:
                    lam = torch.tensor(
                        np.stack([Xth[:, 0], Xth[:, 1],
                                  1.0 / (Xth[:, 0] * Xth[:, 1])], axis=-1),
                        dtype=torch.float64, requires_grad=True)
                    _, sth, _ = model(lam, branch="theta")
                    sth_pred = sth.detach().cpu().numpy()
                    rmse_th = float(np.sqrt(np.mean((sth_pred - yth) ** 2)))
                if Xz.size:
                    lam = torch.tensor(
                        np.stack([Xz[:, 0], Xz[:, 1],
                                  1.0 / (Xz[:, 0] * Xz[:, 1])], axis=-1),
                        dtype=torch.float64, requires_grad=True)
                    _, _, sz = model(lam, branch="z")
                    sz_pred = sz.detach().cpu().numpy()
                    rmse_z = float(np.sqrt(np.mean((sz_pred - yz) ** 2)))
                preds[key]["by_k"][k] = {"sth": sth_pred, "sz": sz_pred}

                rec.append({
                    "age": age, "animal": animal, "k": k,
                    "loss": float(loss), "aic": float(aic), "bic": float(bic),
                    "rmse_theta": rmse_th, "rmse_z": rmse_z,
                    "rmse_combined": float(np.sqrt(np.nanmean([
                        rmse_th ** 2 if not np.isnan(rmse_th) else np.nan,
                        rmse_z ** 2 if not np.isnan(rmse_z) else np.nan,
                    ]))),
                    "t_base_s": t_base, "t_tweak_s": t_k,
                    "n_terms_used": len(model.fixed_term_names),
                    "terms": ";".join(model.fixed_term_names),
                })
                print(f"           k={k:2d}: rmse_theta={rmse_th:.3f}, "
                      f"rmse_z={rmse_z:.3f}, t_tweak={t_k:.1f}s")
            except Exception as e:
                print(f"           k={k} fit failed: {e}")
    return pd.DataFrame(rec), preds


# --------------------------------------------------------------------------
# Plotting helpers for the refit stage
# --------------------------------------------------------------------------
def _group_by_curve(X, y):
    """Group sorted data into individual stretch sweeps for line plotting.
    PD curves share a constant lam_z; FL curves share a constant lam_theta.
    Returns list of (x_axis, y_axis, label) per detected curve.
    """
    if X.size == 0:
        return []
    return [(X, y)]


def plot_refit_results(metrics_df: pd.DataFrame,
                       preds: Dict, subsets: Dict[int, List[str]],
                       out_dir: Path, mode_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "refit_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ks = sorted(subsets.keys())
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c",
                   "#9467bd", "#ff7f0e", "#17becf"]
    palette = {k: color_cycle[i % len(color_cycle)] for i, k in enumerate(ks)}

    # ---- (A) per-animal stress-stretch curves ---------------------------
    # One subplot per individual experimental sweep (PD60, PD95, FL10, ...).
    # Within each subplot: experimental data as markers + fitted model lines
    # for each k in the subset list.
    for (age, animal), bundle in preds.items():
        if not bundle["by_k"]:
            continue
        Xth, yth, ids_th = bundle["Xth"], bundle["yth"], bundle["ids_th"]
        Xz,  yz,  ids_z  = bundle["Xz"],  bundle["yz"],  bundle["ids_z"]

        # Sort PD ids by inflation pressure level (PD60, PD95, ...) and
        # FL ids by axial-stretch level (FL10, FL100, ...).
        def _level(curve_id):
            mm = re.search(r"(\d+)", str(curve_id))
            return int(mm.group(1)) if mm else 0
        pd_ids = sorted(set(ids_th.tolist()), key=_level) if Xth.size else []
        fl_ids = sorted(set(ids_z.tolist()),  key=_level) if Xz.size else []
        n_cols = max(len(pd_ids), len(fl_ids), 1)
        n_rows = (1 if pd_ids else 0) + (1 if fl_ids else 0)
        if n_rows == 0:
            continue

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.4 * n_cols, 3.2 * n_rows),
                                 squeeze=False)
        row = 0
        if pd_ids:
            for col in range(n_cols):
                ax = axes[row, col]
                if col >= len(pd_ids):
                    ax.set_visible(False); continue
                cid = pd_ids[col]
                mask = (ids_th == cid)
                x = Xth[mask, 0]
                order = np.argsort(x)
                x_sorted = x[order]
                ax.scatter(x_sorted, yth[mask][order], s=22,
                           facecolors="none", edgecolors="black",
                           linewidths=0.9, label="experiment", zorder=4)
                for k in ks:
                    pk = bundle["by_k"].get(k)
                    if pk is None or pk["sth"].size == 0:
                        continue
                    ax.plot(x_sorted, pk["sth"][mask][order],
                            color=palette[k], lw=1.8, label=f"k={k}",
                            zorder=3)
                ax.set_xlabel(r"$\lambda_\theta$")
                ax.set_ylabel(r"$\sigma_\theta$ [kPa]")
                ax.set_title(f"{cid}", fontsize=9)
                ax.grid(alpha=0.3)
                if col == 0:
                    ax.legend(fontsize=7, loc="best")
            row += 1
        if fl_ids:
            for col in range(n_cols):
                ax = axes[row, col]
                if col >= len(fl_ids):
                    ax.set_visible(False); continue
                cid = fl_ids[col]
                mask = (ids_z == cid)
                x = Xz[mask, 1]
                order = np.argsort(x)
                x_sorted = x[order]
                ax.scatter(x_sorted, yz[mask][order], s=22,
                           facecolors="none", edgecolors="black",
                           linewidths=0.9, label="experiment", zorder=4)
                for k in ks:
                    pk = bundle["by_k"].get(k)
                    if pk is None or pk["sz"].size == 0:
                        continue
                    ax.plot(x_sorted, pk["sz"][mask][order],
                            color=palette[k], lw=1.8, label=f"k={k}",
                            zorder=3)
                ax.set_xlabel(r"$\lambda_z$")
                ax.set_ylabel(r"$\sigma_z$ [kPa]")
                ax.set_title(f"{cid}", fontsize=9)
                ax.grid(alpha=0.3)
                if col == 0:
                    ax.legend(fontsize=7, loc="best")

        fig.suptitle(f"{age} / {animal} — experiment vs CombinedEquationSEDF "
                     f"refit ({mode_label})", fontsize=10)
        fig.tight_layout()
        out_path = fig_dir / f"{age}_{animal}_stress_strain.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    # ---- (B) cross-animal RMSE summary ---------------------------------
    if metrics_df.empty:
        return

    pivot_th = metrics_df.pivot_table(index=["age", "animal"],
                                      columns="k", values="rmse_theta")
    pivot_z  = metrics_df.pivot_table(index=["age", "animal"],
                                      columns="k", values="rmse_z")
    pivot_c  = metrics_df.pivot_table(index=["age", "animal"],
                                      columns="k", values="rmse_combined")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, pv, title in zip(axes,
                              [pivot_th, pivot_z, pivot_c],
                              ["RMSE θ (PD)", "RMSE z (FL)", "RMSE combined"]):
        if pv.empty:
            ax.set_visible(False); continue
        x = np.arange(len(pv.index))
        width = 0.8 / max(len(ks), 1)
        for i, k in enumerate(ks):
            if k not in pv.columns:
                continue
            ax.bar(x + (i - (len(ks) - 1) / 2) * width,
                   pv[k].values, width, label=f"k={k}",
                   color=palette[k], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}\n{an}" for a, an in pv.index],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("RMSE [kPa]")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Per-animal refit RMSE by subset size — {mode_label}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "refit_rmse_per_animal.png", dpi=140)
    plt.close(fig)

    # ---- (C) cross-animal box/strip distribution -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)
    for ax, col, title in zip(axes,
                               ["rmse_theta", "rmse_z", "rmse_combined"],
                               ["RMSE θ (PD)", "RMSE z (FL)", "RMSE combined"]):
        data = [metrics_df.loc[metrics_df["k"] == k, col].dropna().values
                for k in ks]
        bp = ax.boxplot(data, positions=range(len(ks)), widths=0.55,
                        patch_artist=True, showmeans=True,
                        medianprops=dict(color="black"),
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black", markersize=6))
        for patch, k in zip(bp["boxes"], ks):
            patch.set_facecolor(palette[k]); patch.set_alpha(0.55)
        for i, vals in enumerate(data):
            ax.scatter(np.full_like(vals, i, dtype=float)
                       + np.random.uniform(-0.08, 0.08, size=len(vals)),
                       vals, s=18, color="0.2", alpha=0.7, zorder=3)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_ylabel("RMSE [kPa]")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Cross-animal RMSE distribution by k — {mode_label}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "refit_rmse_distribution.png", dpi=140)
    plt.close(fig)

    print(f"      Wrote refit_rmse_per_animal.png, refit_rmse_distribution.png, "
          f"and {len(preds)} per-animal overlays in refit_figures/")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_dir", default="cross_age_sensitivity")
    ap.add_argument("--n_grid", type=int, default=20)
    ap.add_argument("--lam_theta_min", type=float, default=0.60,
                    help="Min circumferential stretch (full data envelope ~[0.33, 2.07])")
    ap.add_argument("--lam_theta_max", type=float, default=1.60,
                    help="Max circumferential stretch (trimmed to avoid fiber saturation)")
    ap.add_argument("--lam_z_min", type=float, default=1.00,
                    help="Min axial stretch (data is purely tensile)")
    ap.add_argument("--lam_z_max", type=float, default=1.60)
    ap.add_argument("--sobol_N", type=int, default=1024)
    ap.add_argument("--refit_subsets", action="store_true",
                    help="Refit CombinedEquationSEDF per-animal on top-k subsets")
    ap.add_argument("--refit_mode", default="raw_uniform",
                    choices=["coeff_max", "normalized", "raw_uniform"],
                    help="Which Sobol mode supplies the top-k ranking for refit")
    ap.add_argument("--refit_ks", type=int, nargs="+", default=[5, 8, 10])
    ap.add_argument("--refit_epochs_base", type=int, default=2000)
    ap.add_argument("--refit_epochs_tweak", type=int, default=1000)
    ap.add_argument("--refit_epochs_full", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    specimens = discover_specimens(root)
    if not specimens:
        raise SystemExit(f"No runsfinal/*_final_equation.txt found under {root}")
    print(f"[1/5] Found {len(specimens)} specimens.")
    for age, animal, p in specimens:
        print(f"      {age:>8s}  {animal:<12s}  -> {p.name}")

    df = build_compilation(specimens)
    df.to_csv(out_dir / "term_compilation.csv", index=False)

    freq = (df.groupby(["bucket", "name"])
              .agg(n_specimens=("animal", "nunique"),
                   ages=("age", lambda s: ",".join(sorted(set(s)))),
                   mean_abs_coeff=("coeff_raw", lambda s: float(np.abs(s).mean())),
                   max_abs_coeff =("coeff_raw", lambda s: float(np.abs(s).max())))
              .reset_index()
              .sort_values("n_specimens", ascending=False))
    freq.to_csv(out_dir / "term_frequency.csv", index=False)
    print(f"[2/5] Wrote term_compilation.csv ({len(df)} rows) and "
          f"term_frequency.csv ({len(freq)} unique terms).")

    print("[3/5] Evaluating symbolic library on canonical biaxial grid "
          f"({args.n_grid}x{args.n_grid}, "
          f"λθ in [{args.lam_theta_min},{args.lam_theta_max}], "
          f"λz in [{args.lam_z_min},{args.lam_z_max}]) ...")
    J = import_symbolic_module(root)
    lam_grid = make_kinematic_grid(args.n_grid,
                                   args.lam_theta_min, args.lam_theta_max,
                                   args.lam_z_min, args.lam_z_max)
    basis_names, basis_feats = evaluate_basis(J, lam_grid)

    local = local_sensitivity(df, basis_names, basis_feats)
    local.to_csv(out_dir / "local_sensitivity.csv", index=False)
    print(f"      Wrote local_sensitivity.csv ({len(local)} rows).")

    print(f"[4/5] Running Sobol reduction in three bound regimes "
          f"(SALib, N={args.sobol_N}) ...")
    modes = ["coeff_max", "normalized", "raw_uniform"]
    sobol_tables = {}
    for mode in modes:
        sob = sobol_reduction(df, basis_names, basis_feats,
                              bound_mode=mode, N=args.sobol_N)
        sob.to_csv(out_dir / f"sobol_indices_{mode}.csv", index=False)
        sobol_tables[mode] = sob
        print(f"      [{mode}] wrote sobol_indices_{mode}.csv "
              f"(top ST={sob['ST'].iloc[0]:.3f} for {sob['name'].iloc[0]})")

    with open(out_dir / "reduced_term_set_comparison.txt", "w",
              encoding="utf-8") as f:
        f.write("Sobol-ranked reduced SEDF term set: side-by-side comparison\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Specimens analyzed       : {len(specimens)}\n")
        f.write(f"Unique terms in pool     : {len(df['name'].unique())}\n")
        f.write(f"Kinematic grid           : {args.n_grid}x{args.n_grid}, "
                f"λθ in [{args.lam_theta_min},{args.lam_theta_max}], "
                f"λz in [{args.lam_z_min},{args.lam_z_max}]\n")
        f.write(f"Saltelli base samples N  : {args.sobol_N}\n\n")

        f.write("Bound regimes:\n")
        f.write("  coeff_max   : bounds = +-max|coeff_raw| (outlier-sensitive)\n")
        f.write("  normalized  : bounds = [-1,1] on z-scored psi (equal-footing)\n")
        f.write("  raw_uniform : bounds = [-1,1] on raw psi    (intrinsic shape)\n\n")

        for mode in modes:
            sob = sobol_tables[mode]
            f.write(f"--- {mode} ---\n")
            for k, row in sob.head(10).iterrows():
                f.write(f"  {k+1:2d}. {row['name']:<30s}  "
                        f"ST={row['ST']:.3f} (+/- {row['ST_conf']:.3f}), "
                        f"S1={row['S1']:.3f}, "
                        f"n={row['n_specimens']}, "
                        f"|w|_bound={row['weight_bound']:.3g}\n")
            for kk in (5, 8, 10):
                f.write(f"  -> top-{kk}: " +
                        ", ".join(sob['name'].head(kk).tolist()) + "\n")
            f.write("\n")
    print(f"      Wrote reduced_term_set_comparison.txt.")

    if args.refit_subsets:
        print(f"[5/5] Refitting CombinedEquationSEDF per-animal on top-k "
              f"subsets from mode='{args.refit_mode}', k={args.refit_ks} ...")
        ranking = sobol_tables[args.refit_mode]["name"].tolist()
        subsets = {k: ranking[:k] for k in args.refit_ks}
        with open(out_dir / "refit_subsets_used.txt", "w",
                  encoding="utf-8") as f:
            f.write(f"Sobol mode used: {args.refit_mode}\n\n")
            for k, terms in subsets.items():
                f.write(f"k={k}: " + ", ".join(terms) + "\n")
        metrics, preds = refit_combined(
            J, specimens, subsets,
            epochs_base=args.refit_epochs_base,
            epochs_tweak=args.refit_epochs_tweak,
            epochs_full=args.refit_epochs_full,
        )
        if not metrics.empty:
            metrics.to_csv(out_dir / "combined_refit_metrics.csv", index=False)
            summary = (metrics.groupby("k")
                              .agg(n_animals=("animal", "nunique"),
                                   mean_rmse_theta=("rmse_theta", "mean"),
                                   median_rmse_theta=("rmse_theta", "median"),
                                   mean_rmse_z=("rmse_z", "mean"),
                                   median_rmse_z=("rmse_z", "median"),
                                   mean_rmse_combined=("rmse_combined", "mean"),
                                   median_rmse_combined=("rmse_combined", "median"))
                              .reset_index())
            summary.to_csv(out_dir / "combined_refit_summary.csv", index=False)
            with open(out_dir / "combined_refit_summary.txt", "w",
                      encoding="utf-8") as f:
                f.write(f"CombinedEquationSEDF refit summary "
                        f"(mode={args.refit_mode}, base epochs={args.refit_epochs_base}, "
                        f"tweak epochs={args.refit_epochs_tweak})\n")
                f.write("=" * 78 + "\n\n")
                f.write(summary.to_string(index=False))
                f.write("\n")
            print(f"      Wrote combined_refit_metrics.csv and "
                  f"combined_refit_summary.csv.")
            try:
                plot_refit_results(metrics, preds, subsets,
                                   out_dir, mode_label=args.refit_mode)
            except Exception as e:
                print(f"      Plotting failed: {e}")
        else:
            print("      No animals refit successfully.")
    else:
        print("[5/5] Skipping CombinedEquationSEDF refit "
              "(pass --refit_subsets to enable).")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
