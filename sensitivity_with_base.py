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

Outputs (under --out_dir, default cross_age_sensitivity_with_base):
    term_compilation_with_base.csv
    sobol_indices_with_base_<mode>.csv     (mode: coeff_max | normalized | raw_uniform)
    reduced_term_set_with_base.txt
"""
from __future__ import annotations
import argparse, importlib.util, math, re, sys
from pathlib import Path
from typing import Dict, List, Tuple

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
# A line matches a parameter header if it starts with a non-space token
# ending in ":" — e.g. "base.iso.bthz:" or "base.aniso._α:". The token may
# contain Unicode letters (e.g. Greek alpha), so we accept everything up to
# the colon rather than restricting to ASCII.
HEADER_RE = re.compile(r"^([^\s:][^:]*):\s*$")


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)  # numerically stable


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
    """Apply the documented transforms to obtain effective base coefficients.

    iso (poly):
        b_th  = clamp(softplus(_bth),  max=500)
        b_z   = clamp(softplus(_bz),   max=500)
        b_thz = clamp(_bthz_raw,       max=500)         (no softplus)
    aniso (4fiber):
        k1_m  = softplus(_k1[m])    for m = 0..3
    """
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
            "k1₀ fiber(helical+α)",
            "k1₁ fiber(helical-α)",
            "k1₂ fiber(circumferential)",
            "k1₃ fiber(axial)",
        ]
        for i, lab in enumerate(labels):
            if i < len(k1):
                out[lab] = float(k1[i])
    return out


def collect_base_coeffs(specimens) -> pd.DataFrame:
    """One row per (animal, base term)."""
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
    "k1₀ fiber(helical+α)",
    "k1₁ fiber(helical-α)",
    "k1₂ fiber(circumferential)",
    "k1₃ fiber(axial)",
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
    """Toggle k1 to one family at a time and evaluate fiber_sum."""
    cols: List[np.ndarray] = []
    base_k1 = aniso._k1.detach().clone()
    C = J.C_from_lambdas(lam_grid)
    unit_raw = math.log(math.e - 1.0)  # softplus^-1(1)
    for m in range(4):
        with torch.no_grad():
            new_k1 = torch.full_like(base_k1, -50.0)  # softplus(-50) ≈ 0
            new_k1[m] = unit_raw
            aniso._k1.copy_(new_k1)
        psi = aniso.fiber_sum(C, branch="theta")
        cols.append(psi.detach().cpu().numpy())
    with torch.no_grad():
        aniso._k1.copy_(base_k1)  # restore
    return cols


def _load_aniso_state(aniso, params: Dict[str, np.ndarray]):
    """Copy parsed `base.aniso.<name>` raw parameters into a Collagen4Fam."""
    with torch.no_grad():
        for k, v in params.items():
            if not k.startswith("base.aniso."):
                continue
            attr = k.split(".", 2)[-1]   # e.g., "_k1", "_lambda_lb_raw"
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
    """Returns (names, Psi[N_pts, n_base]).

    mode:
      "skip"          : DEFAULT. Drop the 4 fiber-family base terms from the
                        Sobol pool. The recruited 4-fiber base is treated as
                        an always-on backbone, and Sobol ranks the perturbations
                        on top of it. Returns 3 iso terms only.
      "defaults"      : Include 4 fiber families with Collagen4Fam defaults
                        (no recruitment, no dispersion). Animal-independent.
      "recruited_avg" : Include 4 fiber families with each animal's fitted
                        recruitment+dispersion params, averaged across animals.
                        Note: the recruited fiber psi is so much larger than
                        any cross term that it tends to dominate Sobol entirely.
    """
    iso_cols = _iso_shapes(lam_grid)

    if mode == "skip":
        Psi = np.stack(iso_cols, axis=-1)
        return list(ISO_NAMES), Psi

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

    Psi = np.stack(iso_cols + fiber_cols, axis=-1)
    return list(ISO_NAMES) + list(FIBER_LABELS), Psi


# --------------------------------------------------------------------------
# Sobol over extended basis (cross terms + base terms)
# --------------------------------------------------------------------------
def sobol_extended(unique_terms: List[str],
                   psi_matrix: np.ndarray,
                   coeff_obs: Dict[str, float],
                   n_obs: Dict[str, int],
                   bound_mode: str,
                   N: int) -> pd.DataFrame:
    try:
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
    except ImportError as e:
        raise SystemExit("SALib is required. Install with: pip install SALib") from e

    if bound_mode == "coeff_max":
        bounds, Psi_use = [], psi_matrix.copy()
        for t in unique_terms:
            w = float(coeff_obs.get(t, 1.0))
            if w <= 0 or not np.isfinite(w):
                w = 1.0
            bounds.append([-w, w])
    elif bound_mode == "normalized":
        bounds = [[-1.0, 1.0]] * len(unique_terms)
        cols = []
        for j in range(psi_matrix.shape[1]):
            c = psi_matrix[:, j]; sd = c.std()
            cols.append((c - c.mean()) / (sd if sd > 0 else 1.0))
        Psi_use = np.stack(cols, axis=-1)
    elif bound_mode == "raw_uniform":
        bounds = [[-1.0, 1.0]] * len(unique_terms)
        Psi_use = psi_matrix.copy()
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_dir", default="cross_age_sensitivity_with_base")
    ap.add_argument("--n_grid", type=int, default=20)
    ap.add_argument("--lam_theta_min", type=float, default=0.60,
                    help="Min circumferential stretch (full data ~[0.33, 2.07])")
    ap.add_argument("--lam_theta_max", type=float, default=1.60,
                    help="Max circumferential stretch (trimmed to avoid fiber saturation)")
    ap.add_argument("--lam_z_min", type=float, default=1.00)
    ap.add_argument("--lam_z_max", type=float, default=1.60)
    ap.add_argument("--sobol_N", type=int, default=1024)
    ap.add_argument("--fiber_mode", default="skip",
                    choices=["skip", "defaults", "recruited_avg"],
                    help="How to handle the 4 fiber-family base terms: "
                         "'skip' (default) drops them from Sobol entirely so the "
                         "ranking covers iso base + cross-term perturbations on "
                         "top of the recruited 4-fiber backbone; "
                         "'defaults' adds them with neutral Collagen4Fam params; "
                         "'recruited_avg' adds them with averaged fitted params "
                         "(usually swamps every other term).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load helper functions from the existing sensitivity script ---
    helper_path = root / "compile_terms_and_sensitivity.py"
    if not helper_path.exists():
        raise SystemExit(f"Required helper not found: {helper_path}")
    cts = import_sibling("cts", helper_path)

    specimens = cts.discover_specimens(root)
    if not specimens:
        raise SystemExit(f"No runsfinal/*_final_equation.txt found under {root}")
    print(f"[1/4] Found {len(specimens)} specimens.")

    # --- compile cross-term selections (same as original) -------------
    df_cross = cts.build_compilation(specimens)
    df_base  = collect_base_coeffs(specimens)
    df_all   = pd.concat([df_cross.assign(kind="cross"),
                          df_base.assign(kind="base")],
                         ignore_index=True, sort=False)
    df_all.to_csv(out_dir / "term_compilation_with_base.csv", index=False)
    print(f"[2/4] Wrote term_compilation_with_base.csv "
          f"(cross rows={len(df_cross)}, base rows={len(df_base)}).")

    # --- evaluate full basis on canonical grid ------------------------
    print("[3/4] Evaluating extended basis on canonical biaxial grid "
          f"({args.n_grid}x{args.n_grid}, "
          f"λθ in [{args.lam_theta_min},{args.lam_theta_max}], "
          f"λz in [{args.lam_z_min},{args.lam_z_max}]) ...")
    J = cts.import_symbolic_module(root)
    lam_grid = cts.make_kinematic_grid(args.n_grid,
                                       args.lam_theta_min, args.lam_theta_max,
                                       args.lam_z_min, args.lam_z_max)

    cross_names, cross_feats = cts.evaluate_basis(J, lam_grid)
    cross_unique = [t for t in df_cross["name"].unique() if t in cross_names]
    cross_cols   = [cross_feats[:, cross_names.index(t)] for t in cross_unique]

    print(f"      Fiber base mode          : {args.fiber_mode}")
    base_names, base_feats = evaluate_base_shapes(
        J, lam_grid, mode=args.fiber_mode, specimens=specimens)
    base_cols  = [base_feats[:, j] for j in range(base_feats.shape[1])]

    all_names = base_names + cross_unique
    all_psi   = np.stack(base_cols + cross_cols, axis=-1)
    print(f"      Extended basis: {len(base_names)} base + {len(cross_unique)} cross "
          f"= {len(all_names)} terms.")

    # observed coefficient ranges per term (for coeff_max mode)
    coeff_obs: Dict[str, float] = {}
    n_obs:     Dict[str, int]   = {}
    if not df_cross.empty:
        gb = df_cross.assign(absw=df_cross["coeff_raw"].abs()).groupby("name")
        for nm, sub in gb:
            coeff_obs[nm] = float(sub["absw"].max())
            n_obs[nm]     = int(sub["animal"].nunique())
    if not df_base.empty:
        gb = df_base.assign(absw=df_base["coeff_raw"].abs()).groupby("name")
        for nm, sub in gb:
            coeff_obs[nm] = float(sub["absw"].max())
            n_obs[nm]     = int(sub["animal"].nunique())

    # --- Sobol in three regimes ---------------------------------------
    print(f"[4/4] Running Sobol on extended basis in three bound regimes "
          f"(SALib, N={args.sobol_N}) ...")
    modes = ["coeff_max", "normalized", "raw_uniform"]
    sobol_tables = {}
    for mode in modes:
        sob = sobol_extended(all_names, all_psi, coeff_obs, n_obs,
                             bound_mode=mode, N=args.sobol_N)
        sob.to_csv(out_dir / f"sobol_indices_with_base_{mode}.csv", index=False)
        sobol_tables[mode] = sob
        print(f"      [{mode}] top: {sob['name'].iloc[0]:<30s}  "
              f"ST={sob['ST'].iloc[0]:.3f}")

    with open(out_dir / "reduced_term_set_with_base.txt", "w",
              encoding="utf-8") as f:
        f.write("Sobol-ranked term set (with base contributions): "
                "side-by-side comparison\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Specimens analyzed       : {len(specimens)}\n")
        f.write(f"Base terms                : {len(base_names)} "
                f"({'iso only' if args.fiber_mode == 'skip' else '3 iso + 4 fiber'})\n")
        f.write(f"Cross terms               : {len(cross_unique)}\n")
        f.write(f"Total candidate terms    : {len(all_names)}\n")
        f.write(f"Kinematic grid           : {args.n_grid}x{args.n_grid}, "
                f"λθ in [{args.lam_theta_min},{args.lam_theta_max}], "
                f"λz in [{args.lam_z_min},{args.lam_z_max}]\n")
        f.write(f"Fiber base mode           : {args.fiber_mode}\n")
        f.write(f"Saltelli base samples N  : {args.sobol_N}\n\n")

        f.write("Bound regimes:\n")
        f.write("  coeff_max   : bounds = +-max|coeff_raw| observed across animals\n")
        f.write("                (base coeffs from runsfinal/*_POLY_params.txt)\n")
        f.write("  normalized  : bounds = [-1,1] on z-scored psi (equal-footing)\n")
        f.write("  raw_uniform : bounds = [-1,1] on raw psi (intrinsic shape)\n\n")

        for mode in modes:
            sob = sobol_tables[mode]
            f.write(f"--- {mode} ---\n")
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
    print(f"      Wrote reduced_term_set_with_base.txt.")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
