#!/usr/bin/env python3
"""
Sensitivity analysis EXTENDED with base-model terms.

The base model used in the pipeline is:
    W_base = W_iso(poly) + W_aniso(4fiber)

where:
    W_iso  = b_th * E_th^2 + b_z * E_z^2 + b_thz * <E_th>*<E_z>      (3 terms)
    W_aniso = sum_{m=0..3} (k1_m / (4 k2_m)) * (exp(k2_m * xi_m^2) - 1)
              with xi_m driven by the m-th fiber direction:
                m=0  helical(+alpha)        m=1  helical(-alpha)
                m=2  circumferential e_th   m=3  axial e_z              (4 terms)

So this script enlarges the Sobol input pool from the 20 cross-term basis to
20 + 3 + 4 = 27 candidate terms. Each base term is evaluated as a "shape
function" psi_i(lambda) on the canonical biaxial-stretch grid using default
Collagen4Fam parameters (no recruitment, no dispersion) so the iso and fiber
shapes are well defined and animal-independent. Per-animal observed
coefficients (b_th, b_z, b_thz, k1_m) are parsed from runsfinal/*_POLY_params.txt
and fed into the coeff_max bound regime.

Robustness extensions:
  * Two kinematic grids run side-by-side (physiological inner range +
    extended/full-envelope range). Terms ranked highly on both grids are
    robust; terms that only win on one are artefacts of the corner regime.
  * Larger default Saltelli N (4096 vs 1024) tightens ST confidence
    intervals so ranking gaps exceed noise.
  * n_specimens floor (--n_min, default 3) filters out cross terms that
    only one or two animals selected before they enter the reduced set.
  * Leave-one-animal-out (LOO) stability check on the coeff_max regime
    — re-runs Sobol with each animal removed and reports how often each
    term lands in the top-K across folds.

Outputs (under --out_dir, default cross_age_sensitivity_with_base):
    term_compilation_with_base.csv
    sobol_indices_<grid>_<mode>.csv     (grid: physio | extended ;
                                          mode: coeff_max | normalized | raw_uniform)
    loo_rank_stability_<grid>.csv       (per-term LOO stats, coeff_max regime)
    reduced_term_set_with_base.txt      (combined ranking + robust reduced set)
"""
from __future__ import annotations
import argparse, importlib.util, math, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------
# Reuse helpers from the original sensitivity script
# --------------------------------------------------------------------------
def import_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Parsing per-animal base parameters from runsfinal/*_POLY_params.txt
# --------------------------------------------------------------------------
HEADER_RE = re.compile(r"^([^\s:][^:]*):\s*$")


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def parse_poly_params(path: Path) -> Dict[str, np.ndarray]:
    """Line-based parser: collects numeric values listed under each `<key>:`."""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, np.ndarray] = {}
    cur_key, cur_vals = None, []

    def _flush():
        if cur_key and cur_vals:
            cleaned = " ".join(cur_vals).replace("[", " ").replace("]", " ").replace(",", " ")
            try:
                arr = np.array([float(v) for v in cleaned.split() if v])
                if arr.size:
                    out[cur_key] = arr
            except ValueError:
                pass

    for raw in txt.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = HEADER_RE.match(line)
        if m:
            _flush()
            cur_key, cur_vals = m.group(1).strip(), []
        else:
            cur_vals.append(line)
    _flush()
    return out


def base_coeffs_from_params(params: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "base.iso._bth" in params:
        out["b_th E_θ²"] = float(min(_softplus(params["base.iso._bth"][0]), 500.0))
    if "base.iso._bz" in params:
        out["b_z E_z²"]  = float(min(_softplus(params["base.iso._bz"][0]),  500.0))
    if "base.iso.bthz" in params:
        out["b_θz <E_θ><E_z>"] = float(min(params["base.iso.bthz"][0], 500.0))
    if "base.aniso._k1" in params:
        k1 = _softplus(params["base.aniso._k1"])
        labels = [
            "k1₀ fiber(helical+α)", "k1₁ fiber(helical-α)",
            "k1₂ fiber(circumferential)", "k1₃ fiber(axial)",
        ]
        for i, lab in enumerate(labels):
            if i < len(k1):
                out[lab] = float(k1[i])
    return out


def collect_base_coeffs(specimens) -> pd.DataFrame:
    rows = []
    for age, animal, eq_path in specimens:
        runsfinal = eq_path.parent
        pp_files = sorted(runsfinal.glob("*POLY_params.txt"))
        if not pp_files:
            continue
        params = parse_poly_params(pp_files[0])
        coeffs = base_coeffs_from_params(params)
        bucket_of = lambda n: ("iso_base" if n.startswith("b_") else "aniso_base")
        for name, c in coeffs.items():
            rows.append({"age": age, "animal": animal,
                         "bucket": bucket_of(name), "name": name,
                         "coeff_raw": c, "source": str(pp_files[0])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Evaluate base "shape" terms on the canonical kinematic grid
# --------------------------------------------------------------------------
FIBER_LABELS = [
    "k1₀ fiber(helical+α)", "k1₁ fiber(helical-α)",
    "k1₂ fiber(circumferential)", "k1₃ fiber(axial)",
]
ISO_NAMES = ["b_th E_θ²", "b_z E_z²", "b_θz <E_θ><E_z>"]


def _iso_shapes(lam_grid: torch.Tensor) -> List[np.ndarray]:
    lt = lam_grid[:, 0].detach()
    lz = lam_grid[:, 1].detach()
    E_th = 0.5 * (lt ** 2 - 1.0)
    E_z  = 0.5 * (lz ** 2 - 1.0)
    eps = 1e-6
    E_th_pos = torch.nn.functional.softplus(E_th / eps) * eps
    E_z_pos  = torch.nn.functional.softplus(E_z  / eps) * eps
    return [(E_th ** 2).cpu().numpy(),
            (E_z  ** 2).cpu().numpy(),
            (E_th_pos * E_z_pos).cpu().numpy()]


def _fiber_shapes_from_aniso(J, aniso, lam_grid: torch.Tensor) -> List[np.ndarray]:
    cols: List[np.ndarray] = []
    base_k1 = aniso._k1.detach().clone()
    C = J.C_from_lambdas(lam_grid)
    unit_raw = math.log(math.e - 1.0)
    for m in range(4):
        with torch.no_grad():
            new_k1 = torch.full_like(base_k1, -50.0)
            new_k1[m] = unit_raw
            aniso._k1.copy_(new_k1)
        psi = aniso.fiber_sum(C, branch="theta")
        cols.append(psi.detach().cpu().numpy())
    with torch.no_grad():
        aniso._k1.copy_(base_k1)
    return cols


def _load_aniso_state(aniso, params: Dict[str, np.ndarray]):
    with torch.no_grad():
        for k, v in params.items():
            if not k.startswith("base.aniso."):
                continue
            attr = k.split(".", 2)[-1]
            if not hasattr(aniso, attr):
                continue
            tensor = getattr(aniso, attr)
            if not isinstance(tensor, torch.nn.Parameter) and not torch.is_tensor(tensor):
                continue
            try:
                tensor.copy_(torch.as_tensor(v, dtype=tensor.dtype).reshape(tensor.shape))
            except (RuntimeError, ValueError):
                pass


def evaluate_base_shapes(J, lam_grid: torch.Tensor,
                         mode: str = "skip",
                         specimens=None) -> Tuple[List[str], np.ndarray]:
    iso_cols = _iso_shapes(lam_grid)
    if mode == "skip":
        return list(ISO_NAMES), np.stack(iso_cols, axis=-1)
    if mode == "defaults":
        aniso = J.Collagen4Fam(use_recruitment=False, use_dispersion=False)
        if hasattr(aniso, "dist_type"):
            aniso.dist_type = "lognormal"
        fiber_cols = _fiber_shapes_from_aniso(J, aniso, lam_grid)
    elif mode == "recruited_avg":
        if not specimens:
            raise ValueError("recruited_avg mode needs the specimens list")
        per_animal: List[List[np.ndarray]] = []
        n_loaded = 0
        for age, animal, eq_path in specimens:
            pp_files = sorted(eq_path.parent.glob("*POLY_params.txt"))
            if not pp_files:
                continue
            params = parse_poly_params(pp_files[0])
            aniso = J.Collagen4Fam(use_recruitment=True, use_dispersion=True)
            aniso.dist_type = "lognormal"
            _load_aniso_state(aniso, params)
            try:
                fiber_cols_a = _fiber_shapes_from_aniso(J, aniso, lam_grid)
            except Exception as e:
                print(f"      [recruited_avg] {animal}: skipped ({e})")
                continue
            per_animal.append(fiber_cols_a)
            n_loaded += 1
        if n_loaded == 0:
            raise RuntimeError("recruited_avg: no animals loaded successfully")
        fiber_cols = []
        for m in range(4):
            stack = np.stack([per_animal[a][m] for a in range(n_loaded)], axis=0)
            fiber_cols.append(stack.mean(axis=0))
        print(f"      [recruited_avg] averaged fiber shapes across {n_loaded} animals.")
    else:
        raise ValueError(f"Unknown fiber base mode: {mode}")
    return (list(ISO_NAMES) + list(FIBER_LABELS),
            np.stack(iso_cols + fiber_cols, axis=-1))


# --------------------------------------------------------------------------
# Sobol over extended basis
# --------------------------------------------------------------------------
def sobol_extended(unique_terms: List[str],
                   psi_matrix: np.ndarray,
                   coeff_obs: Dict[str, float],
                   n_obs: Dict[str, int],
                   bound_mode: str,
                   N: int,
                   missing_coeff_eps: float = 1e-9) -> pd.DataFrame:
    """Run Sobol on the given basis. `missing_coeff_eps` controls the bound
    for terms with zero/missing observed coefficient in `coeff_max` mode —
    set to 1.0 to reproduce the old behavior; the LOO loop uses a tiny eps
    so dropped-animal terms collapse instead of being inflated to unit bound."""
    try:
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
    except ImportError as e:
        raise SystemExit("SALib is required. Install with: pip install SALib") from e

    if bound_mode == "coeff_max":
        bounds = []
        for t in unique_terms:
            w = float(coeff_obs.get(t, missing_coeff_eps))
            if (not np.isfinite(w)) or w <= 0:
                w = missing_coeff_eps
            bounds.append([-w, w])
        Psi_use = psi_matrix
    elif bound_mode == "normalized":
        bounds = [[-1.0, 1.0]] * len(unique_terms)
        cols = []
        for j in range(psi_matrix.shape[1]):
            c = psi_matrix[:, j]; sd = c.std()
            cols.append((c - c.mean()) / (sd if sd > 0 else 1.0))
        Psi_use = np.stack(cols, axis=-1)
    elif bound_mode == "raw_uniform":
        bounds = [[-1.0, 1.0]] * len(unique_terms)
        Psi_use = psi_matrix
    else:
        raise ValueError(f"Unknown bound_mode: {bound_mode}")

    problem = {"num_vars": len(unique_terms),
               "names":    unique_terms,
               "bounds":   bounds}

    X = sobol_sample.sample(problem, N, calc_second_order=False)
    W = X @ Psi_use.T
    Y = np.sqrt((W ** 2).mean(axis=1))
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                               print_to_console=False)
    out = pd.DataFrame({
        "name": unique_terms,
        "n_specimens":       [int(n_obs.get(t, 0)) for t in unique_terms],
        "obs_max_coeff_raw": [float(coeff_obs.get(t, np.nan)) for t in unique_terms],
        "weight_bound":      [b[1] for b in bounds],
        "psi_grid_rms":      [float(np.sqrt(np.mean(psi_matrix[:, j] ** 2)))
                              for j in range(psi_matrix.shape[1])],
        "S1":      Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST":      Si["ST"],
        "ST_conf": Si["ST_conf"],
    }).sort_values("ST", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Helpers for the new pipeline
# --------------------------------------------------------------------------
def compute_coeff_obs_n_obs(df_cross: pd.DataFrame,
                            df_base: pd.DataFrame
                            ) -> Tuple[Dict[str, float], Dict[str, int]]:
    coeff_obs: Dict[str, float] = {}
    n_obs:     Dict[str, int]   = {}
    for df in (df_cross, df_base):
        if df is None or df.empty:
            continue
        gb = df.assign(absw=df["coeff_raw"].abs()).groupby("name")
        for nm, sub in gb:
            coeff_obs[nm] = float(sub["absw"].max())
            n_obs[nm]     = int(sub["animal"].nunique())
    return coeff_obs, n_obs


def evaluate_full_basis(J, cts, lam_grid: torch.Tensor,
                        fiber_mode: str, specimens,
                        df_cross: pd.DataFrame
                        ) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    cross_names, cross_feats = cts.evaluate_basis(J, lam_grid)
    cross_unique = [t for t in df_cross["name"].unique() if t in cross_names]
    cross_cols = [cross_feats[:, cross_names.index(t)] for t in cross_unique]
    base_names, base_feats = evaluate_base_shapes(J, lam_grid,
                                                  mode=fiber_mode,
                                                  specimens=specimens)
    base_cols = [base_feats[:, j] for j in range(base_feats.shape[1])]
    all_names = base_names + cross_unique
    all_psi   = np.stack(base_cols + cross_cols, axis=-1)
    return all_names, all_psi, base_names, cross_unique


def run_loo_coeff_max(specimens, df_cross: pd.DataFrame, df_base: pd.DataFrame,
                      all_names: List[str], all_psi: np.ndarray,
                      N: int, top_k: int) -> pd.DataFrame:
    """For each animal: drop, recompute coeff_obs/n_obs, rerun coeff_max Sobol.
    Returns a per-term stability summary."""
    rank_per_fold: Dict[str, List[Optional[int]]] = {t: [] for t in all_names}
    st_per_fold:   Dict[str, List[Optional[float]]] = {t: [] for t in all_names}
    fold_animals:  List[str] = []

    for age, animal, _ in specimens:
        df_c = df_cross[df_cross["animal"] != animal] if not df_cross.empty else df_cross
        df_b = df_base [df_base ["animal"] != animal] if not df_base .empty else df_base
        co, no = compute_coeff_obs_n_obs(df_c, df_b)
        sob = sobol_extended(all_names, all_psi, co, no,
                             bound_mode="coeff_max", N=N,
                             missing_coeff_eps=1e-9)
        sob_by_name = {row["name"]: (i + 1, row["ST"])
                       for i, row in sob.iterrows()}
        for t in all_names:
            if t in sob_by_name and no.get(t, 0) > 0:
                r, st = sob_by_name[t]
                rank_per_fold[t].append(int(r))
                st_per_fold[t].append(float(st))
            else:
                rank_per_fold[t].append(None)
                st_per_fold[t].append(None)
        fold_animals.append(animal)

    rows = []
    for t in all_names:
        ranks = [r for r in rank_per_fold[t] if r is not None]
        sts   = [s for s in st_per_fold[t]   if s is not None]
        n_folds = len(ranks)
        rows.append({
            "name": t,
            "n_loo_folds_present": n_folds,
            "mean_rank": float(np.mean(ranks))      if ranks else np.nan,
            "std_rank":  float(np.std(ranks))       if ranks else np.nan,
            "best_rank": int(np.min(ranks))         if ranks else -1,
            "worst_rank":int(np.max(ranks))         if ranks else -1,
            "freq_in_topK":  (sum(1 for r in ranks if r <= top_k) / n_folds)
                              if n_folds else 0.0,
            "mean_ST": float(np.mean(sts))          if sts   else np.nan,
        })
    return (pd.DataFrame(rows)
            .sort_values(["freq_in_topK", "mean_rank"], ascending=[False, True])
            .reset_index(drop=True))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_dir", default="cross_age_sensitivity_with_base")

    # Grid selection.
    ap.add_argument("--grid", choices=["physio", "extended", "both"],
                    default="both",
                    help="Which kinematic grid(s) to run. Default 'both' runs "
                         "the physiological inner range and the full-envelope "
                         "extended range, then reports robust intersection.")

    # Physiological grid (tighter, near in-vivo loading).
    ap.add_argument("--phys_n_grid", type=int, default=20)
    ap.add_argument("--phys_lam_theta_min", type=float, default=0.80)
    ap.add_argument("--phys_lam_theta_max", type=float, default=1.60)
    ap.add_argument("--phys_lam_z_min",     type=float, default=1.00)
    ap.add_argument("--phys_lam_z_max",     type=float, default=1.40)

    # Extended grid (legacy default: full data envelope).
    ap.add_argument("--n_grid", type=int, default=20)
    ap.add_argument("--lam_theta_min", type=float, default=0.30)
    ap.add_argument("--lam_theta_max", type=float, default=2.10)
    ap.add_argument("--lam_z_min",     type=float, default=1.00)
    ap.add_argument("--lam_z_max",     type=float, default=1.60)

    # Sobol budget.
    ap.add_argument("--sobol_N", type=int, default=4096,
                    help="Saltelli base samples (total evals ~ N*(2D+2)).")

    ap.add_argument("--fiber_mode", default="skip",
                    choices=["skip", "defaults", "recruited_avg"])

    # Robustness knobs.
    ap.add_argument("--n_min", type=int, default=2,
                    help="Minimum n_specimens required for a term to enter "
                         "the robust reduced set.")
    ap.add_argument("--top_k", type=int, default=15,
                    help="Top-K cutoff used for the reduced set and for LOO "
                         "stability frequency.")
    ap.add_argument("--loo_stability_freq", type=float, default=0.75,
                    help="Min fraction of LOO folds in which a term must "
                         "appear in top-K to be called LOO-stable.")
    ap.add_argument("--loo", dest="loo", action="store_true", default=True,
                    help="Run leave-one-animal-out stability (default on).")
    ap.add_argument("--no_loo", dest="loo", action="store_false",
                    help="Skip the LOO stability loop.")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    helper_path = root / "compile_terms_and_sensitivity.py"
    if not helper_path.exists():
        raise SystemExit(f"Required helper not found: {helper_path}")
    cts = import_sibling("cts", helper_path)

    specimens = cts.discover_specimens(root)
    if not specimens:
        raise SystemExit(f"No runsfinal/*_final_equation.txt found under {root}")
    print(f"[1/5] Found {len(specimens)} specimens.")

    df_cross = cts.build_compilation(specimens)
    df_base  = collect_base_coeffs(specimens)
    df_all   = pd.concat([df_cross.assign(kind="cross"),
                          df_base.assign(kind="base")],
                         ignore_index=True, sort=False)
    df_all.to_csv(out_dir / "term_compilation_with_base.csv", index=False)
    print(f"[2/5] Wrote term_compilation_with_base.csv "
          f"(cross rows={len(df_cross)}, base rows={len(df_base)}).")

    coeff_obs_full, n_obs_full = compute_coeff_obs_n_obs(df_cross, df_base)
    J = cts.import_symbolic_module(root)

    grids: List[Tuple[str, int, float, float, float, float]] = []
    if args.grid in ("physio", "both"):
        grids.append(("physio",
                      args.phys_n_grid,
                      args.phys_lam_theta_min, args.phys_lam_theta_max,
                      args.phys_lam_z_min,     args.phys_lam_z_max))
    if args.grid in ("extended", "both"):
        grids.append(("extended",
                      args.n_grid,
                      args.lam_theta_min, args.lam_theta_max,
                      args.lam_z_min,     args.lam_z_max))

    print(f"[3/5] Evaluating extended basis on {len(grids)} grid(s) "
          f"(fiber base mode: {args.fiber_mode}) ...")
    results: Dict[str, dict] = {}
    for label, ng, lthmin, lthmax, lzmin, lzmax in grids:
        print(f"      [{label}] {ng}x{ng}, "
              f"λθ in [{lthmin},{lthmax}], λz in [{lzmin},{lzmax}]")
        lam_grid = cts.make_kinematic_grid(ng, lthmin, lthmax, lzmin, lzmax)
        all_names, all_psi, base_names, cross_unique = evaluate_full_basis(
            J, cts, lam_grid, args.fiber_mode, specimens, df_cross)
        results[label] = {
            "grid_info": (ng, lthmin, lthmax, lzmin, lzmax),
            "all_names": all_names, "all_psi": all_psi,
            "base_names": base_names, "cross_unique": cross_unique,
            "tables": {},
        }
        print(f"      [{label}] basis: {len(base_names)} base + "
              f"{len(cross_unique)} cross = {len(all_names)} terms.")

    print(f"[4/5] Running Sobol in three bound regimes per grid "
          f"(SALib, N={args.sobol_N}) ...")
    modes = ["coeff_max", "normalized", "raw_uniform"]
    for label in results:
        all_names = results[label]["all_names"]
        all_psi   = results[label]["all_psi"]
        for mode in modes:
            sob = sobol_extended(all_names, all_psi,
                                 coeff_obs_full, n_obs_full,
                                 bound_mode=mode, N=args.sobol_N,
                                 missing_coeff_eps=1.0)
            sob.to_csv(out_dir / f"sobol_indices_{label}_{mode}.csv",
                       index=False)
            results[label]["tables"][mode] = sob
            print(f"      [{label}/{mode}] top: "
                  f"{sob['name'].iloc[0]:<30s}  ST={sob['ST'].iloc[0]:.3f}")

    loo_tables: Dict[str, pd.DataFrame] = {}
    if args.loo:
        print(f"[5/5] Leave-one-animal-out stability (coeff_max, "
              f"{len(specimens)} folds) ...")
        for label in results:
            print(f"      [{label}] running LOO ...")
            loo_df = run_loo_coeff_max(
                specimens, df_cross, df_base,
                results[label]["all_names"], results[label]["all_psi"],
                N=args.sobol_N, top_k=args.top_k)
            loo_df.to_csv(out_dir / f"loo_rank_stability_{label}.csv",
                          index=False)
            loo_tables[label] = loo_df
            top_stable = loo_df.head(args.top_k)["name"].tolist()
            print(f"      [{label}] top-{args.top_k} by LOO freq-in-topK: "
                  f"{', '.join(top_stable)}")
    else:
        print("[5/5] LOO stability skipped (--no_loo).")

    # ---- Robust reduced set: intersection of evidence ---------------------
    primary = "physio" if "physio" in results else "extended"
    secondary = "extended" if primary == "physio" else None

    primary_ru = results[primary]["tables"]["raw_uniform"]
    primary_topK = primary_ru.head(args.top_k)["name"].tolist()

    if secondary is not None:
        secondary_ru = results[secondary]["tables"]["raw_uniform"]
        secondary_topK = secondary_ru.head(args.top_k)["name"].tolist()
    else:
        secondary_topK = primary_topK

    loo_stable: List[str] = []
    if args.loo and primary in loo_tables:
        ld = loo_tables[primary]
        loo_stable = ld[ld["freq_in_topK"] >= args.loo_stability_freq]["name"].tolist()

    robust_rows = []
    for name in primary_ru["name"].tolist():
        row = primary_ru[primary_ru["name"] == name].iloc[0]
        n_spec = int(row["n_specimens"])
        in_primary_topK   = name in primary_topK
        in_secondary_topK = name in secondary_topK
        is_loo_stable     = name in loo_stable if args.loo else True
        meets_n           = n_spec >= args.n_min
        is_base = name in results[primary]["base_names"]
        passes = (in_primary_topK and in_secondary_topK
                  and (meets_n or is_base)
                  and is_loo_stable)
        robust_rows.append({
            "name": name,
            "ST_primary":          float(row["ST"]),
            "n_specimens":         n_spec,
            "primary_topK":        in_primary_topK,
            "secondary_topK":      in_secondary_topK,
            "n_meets_floor":       meets_n,
            "loo_stable":          is_loo_stable,
            "is_base":             is_base,
            "in_robust_set":       passes,
        })
    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(out_dir / "robust_reduced_set.csv", index=False)
    robust_terms = robust_df[robust_df["in_robust_set"]]["name"].tolist()

    # ---- Write the human-readable report ---------------------------------
    with open(out_dir / "reduced_term_set_with_base.txt", "w",
              encoding="utf-8") as f:
        f.write("Sobol-ranked term set (with base contributions): "
                "robustness comparison\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Specimens analyzed       : {len(specimens)}\n")
        f.write(f"Fiber base mode          : {args.fiber_mode}\n")
        f.write(f"Saltelli base samples N  : {args.sobol_N}\n")
        f.write(f"n_specimens floor (n_min): {args.n_min}\n")
        f.write(f"top-K cutoff             : {args.top_k}\n")
        f.write(f"LOO stability threshold  : "
                f"{args.loo_stability_freq:.0%} of folds in top-K "
                f"({'on' if args.loo else 'off'})\n\n")

        f.write("Grids:\n")
        for label, info in results.items():
            ng, lthmin, lthmax, lzmin, lzmax = info["grid_info"]
            f.write(f"  {label:<10s} : {ng}x{ng}, "
                    f"λθ in [{lthmin},{lthmax}], λz in [{lzmin},{lzmax}]\n")
        f.write("\nBound regimes:\n")
        f.write("  coeff_max   : bounds = +-max|coeff_raw| observed across animals\n")
        f.write("                (base coeffs from runsfinal/*_POLY_params.txt)\n")
        f.write("  normalized  : bounds = [-1,1] on z-scored psi (equal-footing)\n")
        f.write("  raw_uniform : bounds = [-1,1] on raw psi (intrinsic shape)\n\n")

        for label, info in results.items():
            f.write(f"{'#' * 78}\n# Grid: {label}\n{'#' * 78}\n\n")
            base_names = info["base_names"]
            for mode in modes:
                sob = info["tables"][mode]
                f.write(f"--- {label} / {mode} ---\n")
                for k, row in sob.head(15).iterrows():
                    marker = " *base*" if row['name'] in base_names else ""
                    f.write(f"  {k+1:2d}. {row['name']:<30s}  "
                            f"ST={row['ST']:.3f} (+/- {row['ST_conf']:.3f}), "
                            f"S1={row['S1']:.3f}, "
                            f"n={row['n_specimens']}, "
                            f"|w|_bound={row['weight_bound']:.3g}"
                            f"{marker}\n")
                for kk in (5, 8, 10):
                    f.write(f"  -> top-{kk}: " +
                            ", ".join(sob['name'].head(kk).tolist()) + "\n")
                f.write("\n")

        if args.loo:
            for label, ld in loo_tables.items():
                f.write(f"--- LOO stability (coeff_max, grid={label}) ---\n")
                for k, row in ld.head(15).iterrows():
                    f.write(f"  {k+1:2d}. {row['name']:<30s}  "
                            f"freq_in_top{args.top_k}={row['freq_in_topK']:.2f}, "
                            f"mean_rank={row['mean_rank']:.1f}, "
                            f"std_rank={row['std_rank']:.1f}, "
                            f"folds_present={row['n_loo_folds_present']}/"
                            f"{len(specimens)}\n")
                f.write("\n")

        f.write("=" * 78 + "\n")
        f.write("ROBUST REDUCED SET\n")
        f.write("=" * 78 + "\n")
        f.write("Inclusion criteria:\n")
        f.write(f"  (1) in top-{args.top_k} of raw_uniform on the PRIMARY grid "
                f"('{primary}')\n")
        if secondary is not None:
            f.write(f"  (2) in top-{args.top_k} of raw_uniform on the SECONDARY grid "
                    f"('{secondary}')\n")
        f.write(f"  (3) n_specimens >= {args.n_min}  "
                f"(base terms exempt from this floor)\n")
        if args.loo:
            f.write(f"  (4) LOO freq-in-top{args.top_k} (coeff_max, "
                    f"grid '{primary}') >= {args.loo_stability_freq:.0%}\n")
        f.write("\n")
        if robust_terms:
            f.write(f"  -> {len(robust_terms)} term(s) survive all criteria:\n")
            for t in robust_terms:
                f.write(f"     - {t}\n")
        else:
            f.write("  -> NO term survives all criteria. Try lowering --top_k, "
                    "--n_min, or --loo_stability_freq.\n")
        f.write("\nFull per-term decision matrix written to "
                "robust_reduced_set.csv.\n")

    print(f"      Wrote reduced_term_set_with_base.txt and robust_reduced_set.csv.")
    if robust_terms:
        print(f"\nRobust reduced set ({len(robust_terms)} terms): "
              f"{', '.join(robust_terms)}")
    else:
        print("\nRobust reduced set is empty under current thresholds.")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
