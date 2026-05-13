#!/usr/bin/env python3
"""
Morris -> Sobol sensitivity workflow for cross-age juvenile vein SEDF terms.

This script starts from the per-animal final equation files already produced by
the fitting/symbolic-regression pipeline, then:

  1. Compiles selected isotropic, anisotropic, and coupling terms per animal.
  2. Evaluates every candidate term on a common biaxial stretch grid.
  3. Saves the top terms per animal using a local contribution proxy.
  4. Runs Morris screening on the combined cross-animal candidate pool.
  5. Runs Sobol on the Morris-reduced final term set.
  6. Optionally runs PAWN and Delta robustness checks on the final term set.

Typical use:
    python morris_sobol_workflow.py --root . --top_k 10 --robustness

Optional base-model terms:
    python morris_sobol_workflow.py --base_mode iso --top_k 10 --robustness

Outputs are written under --out_dir, default:
    cross_age_morris_sobol_workflow/
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# The local scientific stack can load multiple OpenMP runtimes on macOS when
# torch/numpy/SALib are imported together. Set this before importing them.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-codex"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd


BOUND_MODES = ("coeff_max", "raw_uniform", "normalized")
BASE_MODES = ("none", "iso", "iso_fiber_defaults", "iso_fiber_recruited_avg")


def import_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_existing_helpers(root: Path):
    """Load helper functions from the existing repository scripts."""
    cts_path = root / "compile_terms_and_sensitivity.py"
    if not cts_path.exists():
        raise SystemExit(f"Required helper not found: {cts_path}")
    cts = import_sibling("compile_terms_and_sensitivity_helpers", cts_path)

    swb = None
    swb_path = root / "sensitivity_with_base.py"
    if swb_path.exists():
        swb = import_sibling("sensitivity_with_base_helpers", swb_path)
    return cts, swb


def age_to_number(age: str) -> float:
    txt = "".join(ch for ch in str(age) if ch.isdigit() or ch == ".")
    try:
        return float(txt)
    except ValueError:
        return np.nan


def local_contribution_table(
    df: pd.DataFrame,
    names: List[str],
    psi_matrix: np.ndarray,
) -> pd.DataFrame:
    """Contribution proxy per animal: abs(coefficient) * RMS(term shape)."""
    rms = {
        name: float(np.sqrt(np.mean(psi_matrix[:, j] ** 2)))
        for j, name in enumerate(names)
    }
    records = []
    for (age, animal), g in df.groupby(["age", "animal"], sort=False):
        tmp = []
        for _, row in g.iterrows():
            psi_rms = rms.get(row["name"], np.nan)
            coeff = float(row.get("coeff_raw", np.nan))
            contrib = abs(coeff) * psi_rms
            tmp.append((row, psi_rms, contrib))
        total = sum(v for _, _, v in tmp if np.isfinite(v))
        if total <= 0 or not np.isfinite(total):
            total = 1.0
        for row, psi_rms, contrib in tmp:
            records.append({
                "age": age,
                "age_weeks": age_to_number(age),
                "animal": animal,
                "bucket": row.get("bucket", ""),
                "kind": row.get("kind", ""),
                "name": row["name"],
                "coeff_raw": row.get("coeff_raw", np.nan),
                "psi_grid_rms": psi_rms,
                "contribution": contrib,
                "contribution_frac": contrib / total,
            })
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(
        ["age_weeks", "animal", "contribution"],
        ascending=[True, True, False],
    )


def observed_term_stats(df: pd.DataFrame, names: List[str], psi_matrix: np.ndarray) -> pd.DataFrame:
    records = []
    for j, name in enumerate(names):
        g = df.loc[df["name"] == name]
        bucket = ""
        kind = ""
        if not g.empty:
            bucket = str(g["bucket"].dropna().iloc[0]) if "bucket" in g else ""
            kind = str(g["kind"].dropna().iloc[0]) if "kind" in g else ""
        records.append({
            "name": name,
            "bucket": bucket,
            "kind": kind,
            "n_specimens": int(g[["age", "animal"]].drop_duplicates().shape[0]) if not g.empty else 0,
            "n_ages": int(g["age"].nunique()) if not g.empty else 0,
            "obs_mean_abs_coeff_raw": float(g["coeff_raw"].abs().mean()) if not g.empty else np.nan,
            "obs_max_abs_coeff_raw": float(g["coeff_raw"].abs().max()) if not g.empty else np.nan,
            "psi_grid_mean": float(np.mean(psi_matrix[:, j])),
            "psi_grid_std": float(np.std(psi_matrix[:, j])),
            "psi_grid_rms": float(np.sqrt(np.mean(psi_matrix[:, j] ** 2))),
        })
    return pd.DataFrame(records)


def make_bounds_and_features(
    names: List[str],
    psi_matrix: np.ndarray,
    stats: pd.DataFrame,
    bound_mode: str,
) -> Tuple[Dict, np.ndarray]:
    if bound_mode not in BOUND_MODES:
        raise ValueError(f"Unknown bound_mode={bound_mode!r}; choose from {BOUND_MODES}")

    stat_by_name = stats.set_index("name")
    if bound_mode == "coeff_max":
        bounds = []
        for name in names:
            w = float(stat_by_name.loc[name, "obs_max_abs_coeff_raw"])
            if not np.isfinite(w) or w <= 0:
                w = 1.0
            bounds.append([-w, w])
        psi_use = psi_matrix.copy()
    elif bound_mode == "raw_uniform":
        bounds = [[-1.0, 1.0]] * len(names)
        psi_use = psi_matrix.copy()
    else:
        bounds = [[-1.0, 1.0]] * len(names)
        cols = []
        for j in range(psi_matrix.shape[1]):
            col = psi_matrix[:, j]
            sd = float(np.std(col))
            if sd <= 0 or not np.isfinite(sd):
                sd = 1.0
            cols.append((col - float(np.mean(col))) / sd)
        psi_use = np.stack(cols, axis=-1)

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    return problem, psi_use


def surrogate_response(samples: np.ndarray, psi_use: np.ndarray) -> np.ndarray:
    """Scalar response for SALib: RMS energy perturbation over the grid."""
    values_on_grid = samples @ psi_use.T
    return np.sqrt(np.mean(values_on_grid ** 2, axis=1))


def run_morris(
    names: List[str],
    psi_matrix: np.ndarray,
    stats: pd.DataFrame,
    bound_mode: str,
    trajectories: int,
    num_levels: int,
    optimal_trajectories: int | None,
    seed: int,
) -> pd.DataFrame:
    try:
        from SALib.sample import morris as morris_sample
        from SALib.analyze import morris as morris_analyze
    except ImportError as exc:
        raise SystemExit("SALib is required. Install with: pip install SALib") from exc

    problem, psi_use = make_bounds_and_features(names, psi_matrix, stats, bound_mode)
    X = morris_sample.sample(
        problem,
        N=trajectories,
        num_levels=num_levels,
        optimal_trajectories=optimal_trajectories,
        seed=seed,
    )
    Y = surrogate_response(X, psi_use)
    Si = morris_analyze.analyze(
        problem,
        X,
        Y,
        num_levels=num_levels,
        print_to_console=False,
    )

    out = pd.DataFrame({
        "name": names,
        "mu": Si["mu"],
        "mu_star": Si["mu_star"],
        "mu_star_conf": Si["mu_star_conf"],
        "sigma": Si["sigma"],
    })
    out = out.merge(stats, on="name", how="left")
    return out.sort_values("mu_star", ascending=False).reset_index(drop=True)


def run_sobol(
    names: List[str],
    psi_matrix: np.ndarray,
    stats: pd.DataFrame,
    bound_mode: str,
    n_base: int,
    calc_second_order: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
    except ImportError as exc:
        raise SystemExit("SALib is required. Install with: pip install SALib") from exc

    problem, psi_use = make_bounds_and_features(names, psi_matrix, stats, bound_mode)
    X = sobol_sample.sample(problem, n_base, calc_second_order=calc_second_order, seed=seed)
    Y = surrogate_response(X, psi_use)
    Si = sobol_analyze.analyze(
        problem,
        Y,
        calc_second_order=calc_second_order,
        print_to_console=False,
    )

    first_total = pd.DataFrame({
        "name": names,
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    }).merge(stats, on="name", how="left")
    first_total = first_total.sort_values("ST", ascending=False).reset_index(drop=True)

    second_order = pd.DataFrame()
    if calc_second_order and "S2" in Si:
        rows = []
        s2 = Si["S2"]
        s2_conf = Si["S2_conf"]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rows.append({
                    "term_i": names[i],
                    "term_j": names[j],
                    "S2": float(s2[i, j]),
                    "S2_conf": float(s2_conf[i, j]),
                })
        second_order = pd.DataFrame(rows).sort_values("S2", ascending=False).reset_index(drop=True)

    return first_total, second_order


def run_robustness_checks(
    names: List[str],
    psi_matrix: np.ndarray,
    stats: pd.DataFrame,
    bound_mode: str,
    n_samples: int,
    pawn_slices: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from SALib.sample import latin
        from SALib.analyze import delta, pawn
    except ImportError as exc:
        raise SystemExit("SALib is required. Install with: pip install SALib") from exc

    problem, psi_use = make_bounds_and_features(names, psi_matrix, stats, bound_mode)
    X = latin.sample(problem, n_samples, seed=seed)
    Y = surrogate_response(X, psi_use)

    # Compute only the moment-independent Delta measure here. SALib can also
    # estimate first-order Sobol inside delta.analyze(method="all"), but this
    # workflow already runs Sobol directly and the extra estimate can be less
    # stable with the local scipy/OpenMP stack.
    delta_si = delta.analyze(problem, X, Y, print_to_console=False, seed=seed, method="delta")
    delta_df = pd.DataFrame({
        "name": names,
        "delta": delta_si["delta"],
        "delta_conf": delta_si["delta_conf"],
        "S1_delta_method": delta_si.get("S1", np.full(len(names), np.nan)),
        "S1_delta_conf": delta_si.get("S1_conf", np.full(len(names), np.nan)),
    }).merge(stats, on="name", how="left")
    delta_df = delta_df.sort_values("delta", ascending=False).reset_index(drop=True)

    pawn_si = pawn.analyze(problem, X, Y, S=pawn_slices, print_to_console=False, seed=seed)
    pawn_df = pd.DataFrame({
        "name": names,
        "PAWN_minimum": pawn_si["minimum"],
        "PAWN_mean": pawn_si["mean"],
        "PAWN_median": pawn_si["median"],
        "PAWN_maximum": pawn_si["maximum"],
        "PAWN_CV": pawn_si["CV"],
    }).merge(stats, on="name", how="left")
    pawn_df = pawn_df.sort_values("PAWN_median", ascending=False).reset_index(drop=True)
    return delta_df, pawn_df


def prepare_candidates(args) -> Tuple[pd.DataFrame, List[str], np.ndarray, pd.DataFrame, List[Tuple]]:
    root = Path(args.root).resolve()
    cts, swb = load_existing_helpers(root)

    specimens = cts.discover_specimens(root)
    if not specimens:
        raise SystemExit(f"No runsfinal/*_final_equation.txt found under {root}")

    J = cts.import_symbolic_module(root)
    lam_grid = cts.make_kinematic_grid(
        args.n_grid,
        args.lam_theta_min,
        args.lam_theta_max,
        args.lam_z_min,
        args.lam_z_max,
    )
    cross_names_all, cross_feats_all = cts.evaluate_basis(J, lam_grid)

    df_cross = cts.build_compilation(specimens).assign(kind="symbolic")
    cross_unique = [name for name in df_cross["name"].unique() if name in cross_names_all]
    cross_cols = [cross_feats_all[:, cross_names_all.index(name)] for name in cross_unique]

    df_parts = [df_cross]
    names = []
    cols = []

    if args.base_mode != "none":
        if swb is None:
            raise SystemExit("Base terms requested, but sensitivity_with_base.py was not found.")
        fiber_mode = {
            "iso": "skip",
            "iso_fiber_defaults": "defaults",
            "iso_fiber_recruited_avg": "recruited_avg",
        }[args.base_mode]
        base_names, base_feats = swb.evaluate_base_shapes(
            J,
            lam_grid,
            mode=fiber_mode,
            specimens=specimens,
        )
        df_base = swb.collect_base_coeffs(specimens).assign(kind="base")
        df_parts.append(df_base)
        names.extend(base_names)
        cols.extend([base_feats[:, j] for j in range(base_feats.shape[1])])

    names.extend(cross_unique)
    cols.extend(cross_cols)
    psi_matrix = np.stack(cols, axis=-1)

    df_all = pd.concat(df_parts, ignore_index=True, sort=False)
    df_all = df_all.loc[df_all["name"].isin(names)].copy()
    stats = observed_term_stats(df_all, names, psi_matrix)

    if args.min_specimens > 1:
        keep = set(stats.loc[stats["n_specimens"] >= args.min_specimens, "name"])
        idx = [i for i, name in enumerate(names) if name in keep]
        names = [names[i] for i in idx]
        psi_matrix = psi_matrix[:, idx]
        df_all = df_all.loc[df_all["name"].isin(names)].copy()
        stats = observed_term_stats(df_all, names, psi_matrix)

    if not names:
        raise SystemExit("No candidate terms remained after filtering.")
    return df_all, names, psi_matrix, stats, specimens


def write_final_report(
    out_dir: Path,
    args,
    specimens: List[Tuple],
    selected_terms: List[str],
    morris_df: pd.DataFrame,
    sobol_df: pd.DataFrame,
    delta_df: pd.DataFrame | None,
    pawn_df: pd.DataFrame | None,
) -> None:
    with open(out_dir / "final_term_set.txt", "w", encoding="utf-8") as f:
        f.write("Final reduced term set from Morris screening + Sobol ranking\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Specimens analyzed        : {len(specimens)}\n")
        f.write(f"Base mode                 : {args.base_mode}\n")
        f.write(f"Candidate filter          : n_specimens >= {args.min_specimens}\n")
        f.write(f"Morris bound mode         : {args.morris_bound_mode}\n")
        f.write(f"Sobol bound mode          : {args.sobol_bound_mode}\n")
        f.write(f"Final top_k               : {len(selected_terms)}\n")
        f.write(f"Grid                      : {args.n_grid}x{args.n_grid}, "
                f"lam_theta [{args.lam_theta_min}, {args.lam_theta_max}], "
                f"lam_z [{args.lam_z_min}, {args.lam_z_max}]\n\n")

        f.write("Final terms, in Morris order:\n")
        for i, name in enumerate(selected_terms, start=1):
            sob = sobol_df.loc[sobol_df["name"] == name]
            st = float(sob["ST"].iloc[0]) if not sob.empty else np.nan
            s1 = float(sob["S1"].iloc[0]) if not sob.empty else np.nan
            f.write(f"  {i:2d}. {name}    Sobol ST={st:.4f}, S1={s1:.4f}\n")

        f.write("\nTop Morris screening terms:\n")
        for i, row in morris_df.head(max(args.top_k, 10)).iterrows():
            f.write(f"  {i + 1:2d}. {row['name']}    "
                    f"mu_star={row['mu_star']:.4g}, sigma={row['sigma']:.4g}, "
                    f"n={int(row['n_specimens'])}\n")

        f.write("\nSobol ranking within final set:\n")
        for i, row in sobol_df.iterrows():
            f.write(f"  {i + 1:2d}. {row['name']}    "
                    f"ST={row['ST']:.4f} (+/- {row['ST_conf']:.4f}), "
                    f"S1={row['S1']:.4f} (+/- {row['S1_conf']:.4f})\n")

        if delta_df is not None and pawn_df is not None:
            f.write("\nRobustness checks:\n")
            f.write("  Delta top terms: " + ", ".join(delta_df["name"].head(5).tolist()) + "\n")
            f.write("  PAWN top terms : " + ", ".join(pawn_df["name"].head(5).tolist()) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Run Morris screening followed by Sobol sensitivity on cross-age SEDF terms."
    )
    ap.add_argument("--root", default=".", help="Repository/data root.")
    ap.add_argument("--out_dir", default="cross_age_morris_sobol_workflow")
    ap.add_argument("--base_mode", default="none", choices=BASE_MODES,
                    help="Include base-model terms in the candidate pool.")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Number of Morris-screened terms to carry into final Sobol.")
    ap.add_argument("--top_per_animal", type=int, default=10,
                    help="Number of local top terms to save per animal.")
    ap.add_argument("--min_specimens", type=int, default=1,
                    help="Drop terms selected in fewer than this many specimens.")
    ap.add_argument("--n_grid", type=int, default=20)
    ap.add_argument("--lam_theta_min", type=float, default=0.30)
    ap.add_argument("--lam_theta_max", type=float, default=2.10)
    ap.add_argument("--lam_z_min", type=float, default=1.00)
    ap.add_argument("--lam_z_max", type=float, default=1.60)
    ap.add_argument("--morris_bound_mode", default="coeff_max", choices=BOUND_MODES)
    ap.add_argument("--morris_trajectories", type=int, default=128)
    ap.add_argument("--morris_levels", type=int, default=6)
    ap.add_argument("--morris_optimal_trajectories", type=int, default=32,
                    help="Use 0 to disable optimal trajectory selection.")
    ap.add_argument("--sobol_bound_mode", default="coeff_max", choices=BOUND_MODES)
    ap.add_argument("--sobol_N", type=int, default=1024,
                    help="Base sample size for Sobol sampling.")
    ap.add_argument("--second_order", action="store_true",
                    help="Also estimate Sobol second-order interactions for the final term set.")
    ap.add_argument("--robustness", action="store_true",
                    help="Run Delta and PAWN robustness checks on the final term set.")
    ap.add_argument("--robustness_N", type=int, default=4096)
    ap.add_argument("--pawn_slices", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    optimal = args.morris_optimal_trajectories
    if optimal is not None and optimal <= 0:
        optimal = None

    print("[1/6] Preparing candidate term pool...")
    df_all, names, psi_matrix, stats, specimens = prepare_candidates(args)
    print(f"      specimens: {len(specimens)}")
    print(f"      candidate terms: {len(names)}")

    df_all.to_csv(out_dir / "compiled_terms.csv", index=False)
    stats.to_csv(out_dir / "candidate_term_stats.csv", index=False)

    local = local_contribution_table(df_all, names, psi_matrix)
    local.to_csv(out_dir / "local_contribution_by_animal.csv", index=False)
    if not local.empty:
        local_top = (
            local.groupby(["age", "animal"], group_keys=False)
            .head(args.top_per_animal)
            .reset_index(drop=True)
        )
        local_top.to_csv(out_dir / f"top_{args.top_per_animal}_terms_per_animal.csv", index=False)
    print("[2/6] Saved compiled terms, candidate stats, and per-animal local rankings.")

    age_summary = (
        df_all.groupby(["age", "bucket", "name"])
        .agg(
            n_animals=("animal", "nunique"),
            mean_abs_coeff=("coeff_raw", lambda s: float(np.abs(s).mean())),
            max_abs_coeff=("coeff_raw", lambda s: float(np.abs(s).max())),
        )
        .reset_index()
        .sort_values(["age", "n_animals", "max_abs_coeff"], ascending=[True, False, False])
    )
    age_summary.to_csv(out_dir / "term_frequency_by_age.csv", index=False)

    print(f"[3/6] Running Morris screening ({args.morris_bound_mode}, "
          f"trajectories={args.morris_trajectories})...")
    morris_df = run_morris(
        names,
        psi_matrix,
        stats,
        bound_mode=args.morris_bound_mode,
        trajectories=args.morris_trajectories,
        num_levels=args.morris_levels,
        optimal_trajectories=optimal,
        seed=args.seed,
    )
    morris_df.to_csv(out_dir / "morris_screening.csv", index=False)

    selected_terms = morris_df["name"].head(args.top_k).tolist()
    selected_idx = [names.index(term) for term in selected_terms]
    selected_psi = psi_matrix[:, selected_idx]
    selected_stats = observed_term_stats(df_all, selected_terms, selected_psi)
    selected_stats.to_csv(out_dir / "final_candidate_term_stats.csv", index=False)
    print("      Morris top terms: " + ", ".join(selected_terms))

    print(f"[4/6] Running final Sobol analysis ({args.sobol_bound_mode}, N={args.sobol_N})...")
    sobol_df, s2_df = run_sobol(
        selected_terms,
        selected_psi,
        selected_stats,
        bound_mode=args.sobol_bound_mode,
        n_base=args.sobol_N,
        calc_second_order=args.second_order,
        seed=args.seed,
    )
    sobol_df.to_csv(out_dir / "sobol_final_indices.csv", index=False)
    if not s2_df.empty:
        s2_df.to_csv(out_dir / "sobol_second_order_indices.csv", index=False)
    print("      Sobol top terms: " + ", ".join(sobol_df["name"].head(min(5, len(sobol_df))).tolist()))

    delta_df = None
    pawn_df = None
    if args.robustness:
        print(f"[5/6] Running Delta and PAWN robustness checks (N={args.robustness_N})...")
        delta_df, pawn_df = run_robustness_checks(
            selected_terms,
            selected_psi,
            selected_stats,
            bound_mode=args.sobol_bound_mode,
            n_samples=args.robustness_N,
            pawn_slices=args.pawn_slices,
            seed=args.seed,
        )
        delta_df.to_csv(out_dir / "delta_robustness.csv", index=False)
        pawn_df.to_csv(out_dir / "pawn_robustness.csv", index=False)
    else:
        print("[5/6] Skipping PAWN/Delta robustness checks; pass --robustness to enable.")

    print("[6/6] Writing final report...")
    write_final_report(
        out_dir,
        args,
        specimens,
        selected_terms,
        morris_df,
        sobol_df,
        delta_df,
        pawn_df,
    )

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
