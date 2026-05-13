#!/usr/bin/env python3
"""
Refit a FLAT symbolic SEDF per animal using top-K term sets from the
robustness-extended Sobol output of sensitivity_with_base.py:

  * loo_coeff_max  : top-K terms by leave-one-animal-out
                     frequency-in-top-K on the coeff_max regime (defaults:
                     loo_rank_stability_physio.csv; use loo_rank_stability_extended.csv
                     with the extended Sobol outputs).
  * raw_uniform    : top-K terms by ST on the raw_uniform regime (defaults:
                     sobol_indices_physio_raw_uniform.csv; pair extended with
                     sobol_indices_extended_raw_uniform.csv).

There is no longer a separate base+tweak split. The fitted model is

    W(λ) = Σ_i  w_i · ψ_i(λ)

where ψ_i ranges over whichever terms the sensitivity analysis put in the
top-K — cross terms, iso quadratic base terms (b_th E_θ², b_z E_z²,
b_θz <E_θ><E_z>), all on equal footing. Terms the analysis dropped (e.g.
the four fiber-family k1 entries when sensitivity_with_base.py was run with
--fiber_mode skip) do not enter the model.

Every I₄- and I₈-derived ψ_i is evaluated with fiber recruitment (and
dispersion for helicals): the SymbolicInvariantLayer is swapped for a
RecruitedSymbolicInvariantLayer that exposes

    I4m - 1 := ξ_m  (recruitment-weighted expected fiber activation)
    I8D1D2  := sign(I8_raw) · ξ_D1 · ξ_D2

so any downstream feature consuming these (<I4..>², (<I4..>)³,
exp(<I4..>)-1, <I4..>·<E_*>, ...) rides on recruitment-weighted strain.

Recruitment + dispersion + fiber-angle parameters (the Collagen4Fam knobs
that drive ξ_m) are trainable alongside the linear coefficients w_i.

For each animal, two figures are produced:
    fig1  σ_θ vs λ_θ   (all PD sweeps, exp + each subset)
    fig2  σ_z vs λ_z   (all FL sweeps, exp + each subset)
"""
from __future__ import annotations
import argparse, importlib.util, math, os, re, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
def import_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# Iso quadratic base shapes are also valid library entries here, since the
# flat model doesn't separate base from cross. The 4 fiber-family k1 terms
# are NOT supported as library entries: each one represents a whole exponential
# fiber-energy contribution rather than a single scalar shape function, so
# they are dropped with a warning if a top-K ranking surfaced them.
ISO_BASE_NAMES = {"b_th E_θ²", "b_z E_z²", "b_θz <E_θ><E_z>"}
FIBER_BASE_PREFIX = "k1"


def is_unsupported_in_flat_basis(name: str) -> bool:
    """k1₀ .. k1₃ fiber-family terms aren't scalar ψ shapes — drop them."""
    return name.startswith(FIBER_BASE_PREFIX)


# Terms whose physical role is purely a Macaulay-positive monotone stiffener
# of one strain or fiber invariant; their coefficients should be nonneg to
# preserve convexity of W in that direction and to prevent cancellation-style
# artifacts in the fit. Everything else (cross products, signed strains,
# the iso cross b_θz, and I₈) stays signed.
_SIGNED_TERMS = {"b_θz <E_θ><E_z>", "<E_θ>", "<E_z>", "I₈D1D2"}


def is_stiffening_term(name: str) -> bool:
    """True if `name` is a single-invariant Macaulay-positive monomial /
    transform whose weight should be constrained ≥ 0. Cross-products
    (anything containing '·') and the explicit signed list are excluded."""
    if "·" in name:
        return False
    if name in _SIGNED_TERMS:
        return False
    if name.startswith(FIBER_BASE_PREFIX):
        return False
    return True


# --------------------------------------------------------------------------
# Subset selection from sensitivity_with_base.py outputs (no base filtering)
# --------------------------------------------------------------------------
def topk_from_loo(loo_csv: Path, k: int) -> List[str]:
    """Top-k terms from LOO/coeff_max stability ranking.
    File is already sorted by freq_in_topK desc, mean_rank asc; we re-sort
    defensively. Fiber-family k1 entries are skipped (not representable as a
    single scalar ψ in the flat basis); iso base terms are kept."""
    df = pd.read_csv(loo_csv).sort_values(
        ["freq_in_topK", "mean_rank"], ascending=[False, True])
    out, skipped = [], []
    for n in df["name"].tolist():
        if is_unsupported_in_flat_basis(n):
            skipped.append(n); continue
        out.append(n)
        if len(out) >= k:
            break
    if skipped:
        print(f"      [loo] skipped fiber-family entries (not in flat basis): {skipped}")
    return out


def write_final_equation(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """Emit a human-readable summary of the fitted W(λ) per animal, plus the
    per-family recruitment range [λ_lb, λ_ub], dispersion κ, and reference
    stretch. One file shared across animals.
    """
    EQ_HEADER = (
        "Fitted Flat-Symbolic SEDF — Final Equation per Animal\n"
        + "=" * 78 + "\n"
        "Form:  W(λ) = Σ_i w_i · ψ_i(λ)\n"
        "        - ψ_i with I₄_m / I₈ entries are evaluated on a recruitment-\n"
        "          weighted expected fiber activation ξ_m (lognormal recruitment\n"
        "          integrated over slack λ_s ∈ (λ_lb, λ_ub], one bracket per family).\n"
        "        - Helical families d1, d2 carry GOH dispersion κ ∈ [0, 1/3]\n"
        "          (κ=0: perfectly aligned; κ=1/3: fully isotropic).\n"
        "        - The per-curve recruitment shift is\n"
        "              λ_lb_eff(λ_z) = λ_lb · (λ_z / λ_z_iv)\n"
        "          where λ_z_iv is the in-vivo axial stretch inferred from FL\n"
        "          curve intersections (Weizsäcker / Ferruzzi method).\n"
        "\n"
    )
    blocks: List[str] = [EQ_HEADER]
    fam_labels = ["d1", "d2", "theta", "z"]

    for _, row in metrics.iterrows():
        age    = str(row.get("age", "?"))
        animal = str(row.get("animal", "?"))
        subset = str(row.get("subset", "?"))
        terms  = [t for t in str(row.get("terms", "")).split(";") if t]
        coeffs = [(t, float(row[f"w[{t}]"]))
                  for t in terms
                  if f"w[{t}]" in row.index and pd.notna(row[f"w[{t}]"])]
        coeffs_sorted = sorted(coeffs, key=lambda x: abs(x[1]), reverse=True)

        lines: List[str] = []
        lines.append(f"--- {age} / {animal}   (term set: {subset}) ---")
        lines.append("")
        lines.append("Fitted weights (|w| descending):")
        for t, w in coeffs_sorted:
            lines.append(f"   {w:+13.6f}   ·   {t}")

        # One-line algebraic form, signs absorbed, terms in fitted order.
        parts: List[str] = []
        for i, (t, w) in enumerate(coeffs):
            sign = "+" if w >= 0 else "−"
            mag  = f"{abs(w):.4f}"
            if i == 0 and sign == "+":
                parts.append(f"{mag}·{t}")
            else:
                parts.append(f"{sign} {mag}·{t}")
        expr = " ".join(parts) if parts else "(no terms)"
        lines.append("")
        lines.append("Closed form:")
        lines.append(f"   W(λ) = {expr}")
        lines.append("")

        # Fit quality.
        lines.append("Fit quality:")
        lines.append(f"   RMSE_θ = {float(row.get('rmse_theta', float('nan'))):.4f} kPa")
        lines.append(f"   RMSE_z = {float(row.get('rmse_z',     float('nan'))):.4f} kPa")
        lines.append(f"   R²_θ   = {float(row.get('r2_theta',   float('nan'))):.4f}")
        lines.append(f"   R²_z   = {float(row.get('r2_z',       float('nan'))):.4f}")
        lines.append("")

        # Recruitment.
        ref_stretch = float(row.get("recruit_ref_stretch", 1.0))
        rs_init     = float(row.get("recruit_start_init", float("nan")))
        rs_learn    = bool(row.get("recruit_start_learn", False))
        rs_mode     = "learned" if rs_learn else "fixed"
        lines.append("Recruitment:")
        lines.append(f"   λ_z_iv (in-vivo axial reference) = {ref_stretch:.4f}")
        lines.append(f"   λ_lb shared init = {rs_init:.4f}   ({rs_mode})")
        lines.append("   Per-family fitted recruitment range:")
        for fam in fam_labels:
            lb = float(row.get(f"lam_lb[{fam}]", float("nan")))
            ub = float(row.get(f"lam_ub[{fam}]", float("nan")))
            lines.append(f"      {fam:6s}:  λ_lb = {lb:.4f},   λ_ub = {ub:.4f}   "
                         f"(span Δ = {ub - lb:.4f})")
        lines.append(f"   Per-curve effective start: "
                     f"λ_lb_eff(λ_z) = λ_lb · (λ_z / {ref_stretch:.4f})")
        lines.append("")

        # Dispersion (only helical families have κ in Collagen4Fam).
        d_init  = float(row.get("dispersion_init", float("nan")))
        d_learn = bool(row.get("dispersion_learn", False))
        d_mode  = "learned" if d_learn else "fixed"
        lines.append("Dispersion (helical fibers; κ ∈ [0, 1/3]):")
        lines.append(f"   κ shared init = {d_init:.4f}   ({d_mode})")
        for fam in ["d1", "d2"]:
            k = row.get(f"kappa[{fam}]", float("nan"))
            try:
                k = float(k)
            except Exception:
                k = float("nan")
            lines.append(f"      {fam}:  κ = {k:.4f}")
        lines.append("")
        lines.append("")

        blocks.append("\n".join(lines))

    out_path = out_dir / "final_equation.txt"
    out_path.write_text("".join(blocks), encoding="utf-8")
    return out_path


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination 1 − SS_res/SS_tot. Returns NaN if y_true
    has zero variance or is empty."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------
# Recruitment-aware symbolic invariant layer
# --------------------------------------------------------------------------
# Family index in Collagen4Fam.dirs(): [helical+α, helical-α, circumferential, axial]
_FAMILY_FOR_KEY = {"d1": 0, "d2": 1, "theta": 2, "z": 3}


def make_recruited_invariant_layer(J):
    class RecruitedSymbolicInvariantLayer(nn.Module):
        """I4m - 1 → ξ_m (recruitment expectation, mirrors fiber_sum incl.
        dispersion Q for helicals). I8D1D2 → sign(I8_raw) · ξ_D1 · ξ_D2.
        I1-3, I2-3, I5_*-1, E_θ, E_z are passed through unchanged."""
        def __init__(self, aniso_module, branch: str = "theta"):
            super().__init__()
            self.aniso_module = aniso_module
            self.branch = branch

        def forward(self, lam):
            a = self.aniso_module
            C = J.C_from_lambdas(lam)
            E_th = 0.5 * (lam[..., 0] ** 2 - 1.0)
            E_z  = 0.5 * (lam[..., 1] ** 2 - 1.0)
            I1v  = J.I1(C) - 3.0
            I2v  = J.I2(C) - 3.0

            dir_map = a.symbolic_dirs()
            tau = a.tau()
            lam_z_grid = torch.sqrt(torch.clamp(C[..., 1, 1], min=1e-12))
            lam0  = 1.00 + 0.40 * torch.sigmoid(a._lam0_raw)
            lam0h = 1.00 + 0.20 * torch.sigmoid(a._lam0h_raw)

            zero = torch.zeros_like(I1v)
            recruited: Dict[str, torch.Tensor] = {}
            I4_raw:    Dict[str, torch.Tensor] = {}
            for key, m in _FAMILY_FOR_KEY.items():
                if key not in dir_map or dir_map[key] is None:
                    continue
                I4m = J.I4(C, dir_map[key])
                I4_raw[key] = I4m
                if a.use_recruitment:
                    if a.use_dispersion and m < 2:
                        kappa = torch.clamp(
                            torch.sigmoid(a._kappa[m]) * (1.0 / 3.0 - 1e-4),
                            max=1.0 / 3.0 - 1e-4)
                        Q = kappa * I1v + (1.0 - 3.0 * kappa) * (I4m - 1.0)
                        lam_eff = torch.sqrt(torch.clamp(1.0 + Q, min=1e-12))
                    else:
                        lam_eff = torch.sqrt(torch.clamp(I4m, min=1e-12))
                    xi = a._recruit_expectation(
                        lam_eff, lam_z_grid, tau[m], m, branch=self.branch)
                else:
                    if a.use_dispersion and m < 2:
                        kappa = torch.clamp(
                            torch.sigmoid(a._kappa[m]) * (1.0 / 3.0 - 1e-4),
                            max=1.0 / 3.0 - 1e-4)
                        Q = kappa * I1v + (1.0 - 3.0 * kappa) * (I4m - lam0h[m] ** 2)
                        xi = a._smoothpos(Q, tau[m])
                    else:
                        xi = a._smoothpos(I4m - lam0[m] ** 2, tau[m])
                recruited[key] = torch.clamp(xi, 0.0, 5.0)

            I4th = recruited.get("theta", zero)
            I4zv = recruited.get("z",     zero)
            I4D1 = recruited.get("d1",    zero)
            I4D2 = recruited.get("d2",    zero)
            I5D1 = J.I5(C, dir_map["d1"]) - 1.0 if dir_map.get("d1") is not None else zero
            I5D2 = J.I5(C, dir_map["d2"]) - 1.0 if dir_map.get("d2") is not None else zero
            if dir_map.get("d1") is not None and dir_map.get("d2") is not None:
                I8raw = J.I8(C, dir_map["d1"], dir_map["d2"])
                if "d1" in recruited and "d2" in recruited:
                    I8val = torch.sign(I8raw) * recruited["d1"] * recruited["d2"]
                else:
                    I8val = I8raw
            else:
                I8val = zero

            return {
                "I₁-3": I1v, "I₂-3": I2v,
                "I₄θ-1": I4th, "I₄z-1": I4zv,
                "I₄D1-1": I4D1, "I₄D2-1": I4D2,
                "I₅D1-1": I5D1, "I₅D2-1": I5D2,
                "I₈D1D2": I8val,
                "E_z": E_z, "E_θ": E_th,
            }

    return RecruitedSymbolicInvariantLayer


# --------------------------------------------------------------------------
# Flat symbolic SEDF (no base+tweak)
# --------------------------------------------------------------------------
def make_flat_model(J, branch: str = "theta"):
    InvLayer = make_recruited_invariant_layer(J)

    class FlatRecruitedSEDF(nn.Module):
        """W(λ) = Σ_i w_i · ψ_i(λ) over a fixed term list.

        ψ_i is drawn from:
          * the recruitment-aware symbolic library (any feat_map entry
            produced by PreActivation → Operator → Activation layers); or
          * the iso quadratic base shapes (b_th E_θ², b_z E_z²,
            b_θz <E_θ><E_z>) — added explicitly because the symbolic library
            does not emit them.

        Recruitment params live in the embedded Collagen4Fam module and are
        trainable. The k1/k2 fiber-energy params of that module are present
        but unused (no fiber_sum is called); they receive no gradient unless
        a term ends up consuming them indirectly (none do).

        Recruitment start λ_lb:
          * The sigmoid-mapped range [RECRUIT_LB_MIN, RECRUIT_LB_MAX] is set
            module-wide before construction via J.set_recruitment_lb_bounds.
          * `recruit_start_init` initializes _lambda_lb_raw for all 4 fiber
            families to the inverse-sigmoid of (init - lb_min)/(lb_max - lb_min).
          * `recruit_start_learn=False` freezes _lambda_lb_raw (fixed mode);
            True keeps it trainable but starts at the same shared value.

        The upper bound λ_ub stays trainable across whatever bounds were set
        via J.set_recruitment_ub_bounds before construction."""
        def __init__(self, term_list: List[str],
                     basis_branch: str = "theta",
                     recruit_start_init: float = 0.95,
                     recruit_start_learn: bool = True,
                     nonneg_fiber_terms: bool = True,
                     dispersion_init: float = 0.167,
                     dispersion_learn: bool = True):
            super().__init__()
            self.term_list = list(term_list)
            self.aniso = J.Collagen4Fam(use_recruitment=True, use_dispersion=True)
            if hasattr(self.aniso, "dist_type"):
                self.aniso.dist_type = "lognormal"
            self.invariant_layer    = InvLayer(self.aniso, branch=basis_branch)
            self.pre_activation_layer = J.SymbolicPreActivationLayer()
            self.operator_layer       = J.SymbolicOperatorLayer()
            self.activation_layer     = J.SymbolicActivationLayer()
            self.basis_branch         = basis_branch

            # --- Initialize the shared recruitment start λ_lb for all 4 families.
            # The module-level bounds (set via J.set_recruitment_lb_bounds) define
            # the sigmoid range, so we invert it to seed _lambda_lb_raw.
            lb_min = float(J.RECRUIT_LB_MIN)
            lb_max = float(J.RECRUIT_LB_MAX)
            init = float(recruit_start_init)
            if not (lb_min < init < lb_max):
                # Clamp the init into the open interval before logit().
                pad = 1e-3 * (lb_max - lb_min)
                init = max(lb_min + pad, min(lb_max - pad, init))
            p = (init - lb_min) / (lb_max - lb_min)
            p = min(1.0 - 1e-9, max(1e-9, p))
            raw = math.log(p / (1.0 - p))
            with torch.no_grad():
                self.aniso._lambda_lb_raw.fill_(raw)
            self.aniso._lambda_lb_raw.requires_grad_(bool(recruit_start_learn))
            self.recruit_start_init  = init
            self.recruit_start_learn = bool(recruit_start_learn)

            # --- Initialize helical-fiber dispersion κ for both helical families.
            # Effective κ = sigmoid(_kappa) · (1/3 − 1e-4), bounded in [0, ~1/3].
            #   κ = 0      → perfectly aligned fibers
            #   κ = 1/6    → moderate dispersion (sigmoid midpoint)
            #   κ → 1/3    → fully isotropic dispersion
            kappa_init = float(dispersion_init)
            kappa_max  = 1.0 / 3.0 - 1e-4
            if not (0.0 < kappa_init < kappa_max):
                pad = 1e-3 * kappa_max
                kappa_init = max(pad, min(kappa_max - pad, kappa_init))
            pκ = kappa_init / kappa_max
            pκ = min(1.0 - 1e-9, max(1e-9, pκ))
            kappa_raw = math.log(pκ / (1.0 - pκ))
            with torch.no_grad():
                self.aniso._kappa.fill_(kappa_raw)
            self.aniso._kappa.requires_grad_(bool(dispersion_learn))
            self.dispersion_init  = kappa_init
            self.dispersion_learn = bool(dispersion_learn)

            self.n_terms = len(self.term_list)
            self.n_basis = self.n_terms  # for fit_model compatibility

            # --- per-term sign constraint ------------------------------------
            # `nonneg_mask[i] == True`  -> w_i = softplus(_w_raw[i])  (always ≥ 0)
            # `nonneg_mask[i] == False` -> w_i = _w_raw[i]            (signed)
            # Initial effective weights ≈ 0 for both branches:
            #   raw=0   -> signed weight = 0
            #   raw=-4.6-> softplus ≈ 0.01 (≈ 0 in practice)
            self.nonneg_fiber_terms = bool(nonneg_fiber_terms)
            mask = torch.tensor(
                [self.nonneg_fiber_terms and is_stiffening_term(t)
                 for t in self.term_list],
                dtype=torch.bool)
            self.register_buffer("nonneg_mask", mask)
            raw_init = torch.where(mask,
                                   torch.full((self.n_terms,), -4.6, dtype=torch.float64),
                                   torch.zeros(self.n_terms, dtype=torch.float64))
            self._w_raw = nn.Parameter(raw_init)
            self.register_buffer("b_mu",    torch.zeros(self.n_terms, dtype=torch.float64))
            self.register_buffer("b_sigma", torch.ones (self.n_terms, dtype=torch.float64))

            # Validate term resolvability on a probe; warn on misses.
            probe = torch.ones((1, 3), dtype=torch.float64, requires_grad=True)
            feat_map = self._build_feature_map(probe)
            self.missing_terms = [t for t in self.term_list if t not in feat_map]
            self.matched_terms = [t for t in self.term_list if t in feat_map]
            if self.missing_terms:
                print(f"  [FlatRecruitedSEDF] WARN: {len(self.missing_terms)} "
                      f"term(s) not found in basis: {self.missing_terms}")

        def _build_feature_map(self, lam: torch.Tensor) -> Dict[str, torch.Tensor]:
            inv = self.invariant_layer(lam)
            h0  = self.pre_activation_layer(inv)
            op  = self.operator_layer(h0)
            feat_map = {name: value for name, value in self.activation_layer(op)}
            E_th = inv["E_θ"]
            E_z  = inv["E_z"]
            eps = 1e-6
            E_th_pos = torch.nn.functional.softplus(E_th / eps) * eps
            E_z_pos  = torch.nn.functional.softplus(E_z  / eps) * eps
            feat_map["b_th E_θ²"]       = E_th ** 2
            feat_map["b_z E_z²"]        = E_z ** 2
            feat_map["b_θz <E_θ><E_z>"] = E_th_pos * E_z_pos
            return feat_map

        def basis_feats_raw(self, lam: torch.Tensor) -> torch.Tensor:
            feat_map = self._build_feature_map(lam)
            cols = []
            for t in self.term_list:
                v = feat_map.get(t)
                if v is None:
                    v = torch.zeros(lam.shape[:-1], dtype=lam.dtype, device=lam.device)
                cols.append(v)
            return torch.stack(cols, dim=-1)

        def basis_feats(self, lam: torch.Tensor) -> torch.Tensor:
            return (self.basis_feats_raw(lam) - self.b_mu) / self.b_sigma

        @property
        def weights(self) -> torch.Tensor:
            """Effective weights: softplus on the nonneg-mask entries, raw on the rest.
            Use this anywhere a forward computation reads weights, and in
            post-fit reporting. fit_model regularizes _w_raw (the underlying
            parameter), not this property."""
            sp = torch.nn.functional.softplus(self._w_raw)
            return torch.where(self.nonneg_mask, sp, self._w_raw)

        @torch.no_grad()
        def set_stats(self, lam_all: Optional[torch.Tensor]):
            if lam_all is None or lam_all.numel() == 0:
                self.b_mu[:] = 0.0
                self.b_sigma[:] = 1.0
                return
            feats = self.basis_feats_raw(lam_all)
            self.b_mu.copy_(feats.mean(0))
            self.b_sigma.copy_(feats.std(0).clamp_min(1e-6))

        def forward(self, lam: torch.Tensor, create_graph: bool = True,
                    branch: str = "theta"):
            if not lam.requires_grad:
                lam = lam.requires_grad_(True)
            self.invariant_layer.branch = branch
            z = self.basis_feats(lam)
            W = (self.weights * z).sum(dim=-1)
            sigma_th, sigma_z = J.cauchy_from_W(W, lam, create_graph=create_graph)
            return W, sigma_th, sigma_z

    return FlatRecruitedSEDF


# --------------------------------------------------------------------------
# Per-animal refit (single-stage joint fit: weights + recruitment params)
# --------------------------------------------------------------------------
def refit_flat(J, cts, FlatModel, specimens, subsets: Dict[str, List[str]],
               epochs: int, lr: float, seed: int = 0,
               robust_kind: str = "huber", robust_delta: float = 1.5,
               wtheta_boost: float = 1.0,
               w_l2: float = 0.0,
               recruit_ref_stretch_mode: str = "auto",
               recruit_ref_stretch_override: Optional[float] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec: List[dict] = []
    preds: Dict[Tuple[str, str], Dict] = {}

    for age, animal, eq_path in specimens:
        animal_dir = eq_path.parent.parent
        data_dir = cts.find_animal_data_dir(animal_dir)
        if data_dir is None:
            print(f"  [refit] {animal}: no data/ directory, skipping")
            continue
        print(f"  [refit] {age}/{animal}: parsing data from {data_dir.name}/")
        blob = J.parse_csvs_for_age(str(data_dir))
        Xth, yth, Xz, yz, wth, wz, ids_th, ids_z = J.build_dataset_with_weights(blob)
        if Xth.size == 0 and Xz.size == 0:
            print(f"           no usable PD/FL data, skipping")
            continue

        # Per-curve recruitment shift: estimate in-vivo λ_z (the axial stretch
        # at which the fibers were "set" in the tissue's reference state) and
        # use it as the reference stretch in the effective_lambda_lb shift
        # λ_lb_eff = λ_lb · (λ_z / λ_z_iv). Mirrors Juvenile_SEDF_final's
        # configure_recruitment_reference_stretch path.
        if recruit_ref_stretch_mode == "fixed":
            in_vivo_lamz = float(recruit_ref_stretch_override
                                 if recruit_ref_stretch_override is not None
                                 else 1.0)
        else:
            try:
                in_vivo_lamz = float(J.infer_in_vivo_axial_stretch(blob))
            except Exception as e:
                print(f"           [warn] infer_in_vivo_axial_stretch failed ({e}); "
                      f"falling back to 1.0")
                in_vivo_lamz = 1.0
            if not (0.5 < in_vivo_lamz < 2.5):
                print(f"           [warn] inferred λ_z_iv={in_vivo_lamz:.3f} "
                      f"out of plausible range; falling back to 1.0")
                in_vivo_lamz = 1.0
        print(f"           recruit_ref_stretch (λ_z_iv) = {in_vivo_lamz:.4f}")

        lam_all = J._lam_all_from_X(Xth, Xz, device=device)
        key = (age, animal)
        preds[key] = {
            "Xth": Xth.copy() if Xth.size else np.zeros((0, 2)),
            "yth": yth.copy() if Xth.size else np.zeros((0,)),
            "Xz":  Xz.copy()  if Xz.size  else np.zeros((0, 2)),
            "yz":  yz.copy()  if Xz.size  else np.zeros((0,)),
            "ids_th": ids_th if Xth.size else np.array([], object),
            "ids_z":  ids_z  if Xz.size  else np.array([], object),
            "by_k": {},
        }

        for lbl, terms in subsets.items():
            try:
                model = FlatModel(term_list=terms)
                # Configure the per-animal recruitment reference stretch BEFORE
                # any forward pass (set_stats triggers one).
                if hasattr(model.aniso, "_recruit_ref_stretch"):
                    J.configure_recruitment_reference_stretch(
                        model.aniso, in_vivo_lamz)
                with torch.no_grad():
                    model.set_stats(lam_all)
                t0 = time.time()
                # L2 regularization on the linear-combination raw weights
                # _w_raw, anchored at zero. Uses the existing anchor mechanism
                # in fit_model so we don't have to touch the loss assembly.
                anchor_lambda  = float(w_l2)
                anchor_params  = ({"_w_raw": torch.zeros_like(model._w_raw)}
                                  if anchor_lambda > 0 else None)
                loss, aic, bic, model = J.fit_model(
                    model, Xth, yth, Xz, yz,
                    epochs=epochs, name=f"{animal}-{lbl}",
                    lr=lr, weights_th=wth, weights_z=wz,
                    seed=seed, jitter=0.0, jitter_warm_epochs=0,
                    robust_kind=robust_kind, robust_delta=robust_delta,
                    wtheta_boost=wtheta_boost,
                    anchor_lambda=anchor_lambda, anchor_params=anchor_params,
                )
                t_fit = time.time() - t0

                model.eval()
                rmse_th, rmse_z = float("nan"), float("nan")
                r2_th, r2_z     = float("nan"), float("nan")
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
                    r2_th   = _r2(yth, sth_pred)
                if Xz.size:
                    lam = torch.tensor(
                        np.stack([Xz[:, 0], Xz[:, 1],
                                  1.0 / (Xz[:, 0] * Xz[:, 1])], axis=-1),
                        dtype=torch.float64, requires_grad=True)
                    _, _, sz = model(lam, branch="z")
                    sz_pred = sz.detach().cpu().numpy()
                    rmse_z = float(np.sqrt(np.mean((sz_pred - yz) ** 2)))
                    r2_z   = _r2(yz, sz_pred)
                preds[key]["by_k"][lbl] = {"sth": sth_pred, "sz": sz_pred,
                                           "r2_theta": r2_th, "r2_z": r2_z}

                # Record fitted linear coefficients (one per matched term) so
                # the final equation is easy to read off.
                w_eff = (model.weights.detach().cpu().numpy()
                         / model.b_sigma.detach().cpu().numpy())
                # offset constant: shift = -Σ w_i * mu_i / sigma_i  (absorbed into W)
                coeffs = {f"w[{t}]": float(w) for t, w in zip(model.term_list, w_eff)}

                # Post-fit recruitment params (per fiber family): λ_lb, λ_ub.
                # Index order matches Collagen4Fam.dirs(): [helical+α, helical-α,
                # circumferential, axial].
                with torch.no_grad():
                    lam_lb_post = model.aniso.lambda_lb().detach().cpu().numpy()
                    lam_ub_post = model.aniso.lambda_ub().detach().cpu().numpy()
                    # Post-fit dispersion κ (helical families only, shape (2,)).
                    kappa_eff_post = (torch.sigmoid(model.aniso._kappa)
                                      * (1.0/3.0 - 1e-4)).detach().cpu().numpy()
                fam_labels = ["d1", "d2", "theta", "z"]
                recruit_cols = {}
                for i, lab in enumerate(fam_labels):
                    if i < len(lam_lb_post):
                        recruit_cols[f"lam_lb[{lab}]"] = float(lam_lb_post[i])
                    if i < len(lam_ub_post):
                        recruit_cols[f"lam_ub[{lab}]"] = float(lam_ub_post[i])
                # Dispersion entries (only helicals carry κ in Collagen4Fam).
                helical_labels = ["d1", "d2"]
                for i, lab in enumerate(helical_labels):
                    if i < len(kappa_eff_post):
                        recruit_cols[f"kappa[{lab}]"] = float(kappa_eff_post[i])

                rec.append({
                    "age": age, "animal": animal, "subset": lbl,
                    "loss": float(loss), "aic": float(aic), "bic": float(bic),
                    "rmse_theta": rmse_th, "rmse_z": rmse_z,
                    "rmse_combined": float(np.sqrt(np.nanmean([
                        rmse_th ** 2 if not np.isnan(rmse_th) else np.nan,
                        rmse_z ** 2 if not np.isnan(rmse_z) else np.nan,
                    ]))),
                    "r2_theta": r2_th, "r2_z": r2_z,
                    "t_fit_s": t_fit,
                    "n_terms_used": len(model.matched_terms),
                    "n_terms_missing": len(model.missing_terms),
                    "terms": ";".join(model.term_list),
                    "missing": ";".join(model.missing_terms),
                    "recruit_start_init":  float(model.recruit_start_init),
                    "recruit_start_learn": bool(model.recruit_start_learn),
                    "recruit_ref_stretch": float(in_vivo_lamz),
                    "dispersion_init":     float(model.dispersion_init),
                    "dispersion_learn":    bool(model.dispersion_learn),
                    **recruit_cols,
                    **coeffs,
                })
                print(f"           {lbl:15s}: rmse_θ={rmse_th:.3f}, "
                      f"rmse_z={rmse_z:.3f}, R²_θ={r2_th:.3f}, "
                      f"R²_z={r2_z:.3f}, t_fit={t_fit:.1f}s")
            except Exception as e:
                print(f"           {lbl}: fit failed: {e}")
    return pd.DataFrame(rec), preds


# --------------------------------------------------------------------------
# Two-figure-per-animal stress-stretch plot (overlaid sweeps)
# --------------------------------------------------------------------------
def _level(curve_id) -> int:
    m = re.search(r"(\d+)", str(curve_id))
    return int(m.group(1)) if m else 0


def plot_proposed_equation(preds: Dict, out_dir: Path):
    """One side-by-side σ_θ vs λ_θ / σ_z vs λ_z figure per animal.
    Experimental sweeps in solid black, fitted proposed equation in red dashed.
    R² for each direction is printed in the title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "refit_figures_per_direction"
    fig_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.ticker import MaxNLocator

    EXP_COLOR   = "black"
    MODEL_COLOR = "crimson"
    LW_EXP, LW_MODEL = 3.2, 3.8
    TITLE_FS, AXLBL_FS, TICK_FS = 18, 24, 18

    def _strip_spines(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", direction="out",
                       length=7, width=1.6, labelsize=TICK_FS)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(1.6)
        # Exactly 3 ticks per axis (matches the reference figure style).
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))

    for (age, animal), bundle in preds.items():
        if not bundle["by_k"]:
            continue
        lbl = next(iter(bundle["by_k"].keys()))
        pk = bundle["by_k"][lbl]
        sth_pred = pk["sth"]; sz_pred = pk["sz"]
        r2_th = pk.get("r2_theta", float("nan"))
        r2_z  = pk.get("r2_z",     float("nan"))

        Xth, yth, ids_th = bundle["Xth"], bundle["yth"], bundle["ids_th"]
        Xz,  yz,  ids_z  = bundle["Xz"],  bundle["yz"],  bundle["ids_z"]

        fig, (ax_th, ax_z) = plt.subplots(1, 2, figsize=(14, 5.6))
        title = (f"{age} / {animal}: Experimental vs Proposed Equation     "
                 f"R²$_\\theta$={r2_th:.3f}    R²$_z$={r2_z:.3f}")
        fig.suptitle(title, fontsize=TITLE_FS, fontweight="bold")

        # --- σ_θ vs λ_θ (left) ---
        if Xth.size:
            for cid in sorted(set(ids_th.tolist()), key=_level):
                mask = (ids_th == cid)
                x = Xth[mask, 0]; order = np.argsort(x); xs = x[order]
                ax_th.plot(xs, yth[mask][order],
                           color=EXP_COLOR, lw=LW_EXP, alpha=0.9, zorder=3)
                if sth_pred.size:
                    ax_th.plot(xs, sth_pred[mask][order],
                               color=MODEL_COLOR, linestyle="--",
                               lw=LW_MODEL, alpha=0.95, zorder=4)
        ax_th.set_xlabel(r"$\lambda_\theta$",  fontsize=AXLBL_FS, fontweight="bold")
        ax_th.set_ylabel(r"$\sigma_\theta$ [kPa]", fontsize=AXLBL_FS, fontweight="bold")
        _strip_spines(ax_th)

        # --- σ_z vs λ_z (right) ---
        if Xz.size:
            for cid in sorted(set(ids_z.tolist()), key=_level):
                mask = (ids_z == cid)
                x = Xz[mask, 1]; order = np.argsort(x); xs = x[order]
                ax_z.plot(xs, yz[mask][order],
                          color=EXP_COLOR, lw=LW_EXP, alpha=0.9, zorder=3)
                if sz_pred.size:
                    ax_z.plot(xs, sz_pred[mask][order],
                              color=MODEL_COLOR, linestyle="--",
                              lw=LW_MODEL, alpha=0.95, zorder=4)
        ax_z.set_xlabel(r"$\lambda_z$",          fontsize=AXLBL_FS, fontweight="bold")
        ax_z.set_ylabel(r"$\sigma_z$ [kPa]",     fontsize=AXLBL_FS, fontweight="bold")
        _strip_spines(ax_z)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(fig_dir / f"{age}_{animal}_proposed_eq.png", dpi=160)
        plt.close(fig)

    print(f"      Wrote proposed-equation figures for {len(preds)} animals.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--sens_dir", default="cross_age_sensitivity_with_base")
    ap.add_argument("--loo_csv", default="loo_rank_stability_physio.csv",
                    help="LOO rank-stability table from sensitivity_with_base.py "
                         "(freq_in_topK, mean_rank), e.g. loo_rank_stability_physio.csv "
                         "or loo_rank_stability_extended.csv, not a sobol_indices_*.csv). "
                         "This is the ONLY ranking the script uses now — one flat "
                         "equation is fit per animal.")
    ap.add_argument("--out_dir", default="refit_morris_sobol_15terms")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-1)
    ap.add_argument("--basis_branch", choices=["theta", "z"], default="theta")
    ap.add_argument("--only_animal", default= "Gus")
    ap.add_argument("--only_age",    default= "10weeks")
    # Recruitment-start (shared λ_lb across the 4 fiber families) controls.
    ap.add_argument("--recruit_start_mode", choices=["learn", "fixed"], default="learn",
                    help="Treat the shared recruitment start λ_lb as a learned "
                         "parameter or hold it fixed at --recruit_start_init.")
    ap.add_argument("--recruit_start_init", type=float, default=1.65,
                    help="Initial guess (learn mode) or fixed value (fixed mode) "
                         "for the shared recruitment start λ_lb.")
    ap.add_argument("--recruit_start_min", type=float, default=0.30,
                    help="Lower sigmoid bound for the learned recruitment start λ_lb.")
    ap.add_argument("--recruit_start_max", type=float, default=2.10,
                    help="Upper sigmoid bound for the learned recruitment start λ_lb.")
    ap.add_argument("--recruit_end_min", type=float, default=1.15,
                    help="Lower sigmoid bound for the learned recruitment upper stretch λ_ub.")
    ap.add_argument("--recruit_end_max", type=float, default=2.80,
                    help="Upper sigmoid bound for the learned recruitment upper stretch λ_ub.")
    # Helical-fiber dispersion κ (only the 2 helical families carry it).
    ap.add_argument("--dispersion_mode", choices=["learn", "fixed"], default="learn",
                    help="Treat helical fiber dispersion κ as learned or held fixed "
                         "at --dispersion_init. κ ∈ [0, 1/3]: 0=aligned, 1/3=isotropic.")
    ap.add_argument("--dispersion_init", type=float, default=0.08,
                    help="Initial κ for both helical fiber families. 1/6 ≈ 0.167 "
                         "is the sigmoid midpoint of the κ ∈ [0, 1/3] interval; "
                         "use ~0.05 for nearly-aligned fibers or ~0.30 for "
                         "near-isotropic dispersion.")
    # Robust loss controls (forwarded to J.fit_model).
    ap.add_argument("--robust", choices=["huber", "charbonnier", "mse"],
                    default="huber",
                    help="Robust loss kernel used by fit_model.")
    ap.add_argument("--robust_delta", type=float, default=0.5,
                    help="Robust loss δ (Huber / Charbonnier knee).")
    ap.add_argument("--wtheta_boost", type=float, default=5.5,
                    help="Multiplier on the σ_θ branch loss relative to σ_z. "
                         "Raise above 1.0 if σ_θ RMSE is much larger than σ_z "
                         "after a balanced fit (e.g. 2.0–5.0).")
    # Sign constraints on the linear-combination weights.
    ap.add_argument("--nonneg_fiber_terms", dest="nonneg_fiber_terms",
                    action="store_true", default=True,
                    help="Constrain Macaulay-positive fiber/iso stiffening "
                         "monomial weights (I₄ powers, exp, xlog1p; I₁-3, I₂-3, "
                         "b_th E_θ², b_z E_z²) to ≥ 0 via softplus. Default ON. "
                         "Kills cancellation-style σ-vs-λ oscillations.")
    ap.add_argument("--no_nonneg_fiber_terms", dest="nonneg_fiber_terms",
                    action="store_false",
                    help="Disable the nonneg constraint and let every weight be signed.")
    ap.add_argument("--w_l2", type=float, default=0.0,
                    help="L2 regularization strength on the raw weight vector "
                         "(anchored at 0). Discourages large opposing-magnitude "
                         "coefficients. Try 1e-3 to 1e-2.")
    # Per-curve recruitment shift (λ_lb_eff = λ_lb · λ_z / λ_z_iv).
    ap.add_argument("--recruit_ref_stretch_mode", choices=["auto", "fixed"],
                    default="auto",
                    help="'auto' (default) estimates the in-vivo axial stretch "
                         "from FL curves via Juvenile_SEDF's infer_in_vivo_axial_stretch "
                         "(Weizsäcker/Ferruzzi FL-intersection method). This makes "
                         "each PD curve (different held λ_z) see a different "
                         "effective recruitment start. 'fixed' uses --recruit_ref_stretch.")
    ap.add_argument("--recruit_ref_stretch", type=float, default=None,
                    help="Fixed λ_z_iv override for --recruit_ref_stretch_mode=fixed. "
                         "Defaults to 1.0 (which disables the per-curve shift) "
                         "if not given.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cts = import_sibling("cts", root / "compile_terms_and_sensitivity.py")
    J = cts.import_symbolic_module(root)

    sens_dir = root / args.sens_dir
    loo_csv  = sens_dir / args.loo_csv
    if not loo_csv.exists():
        raise SystemExit(f"Missing {loo_csv}. Run sensitivity_with_base.py first.")

    proposed_terms = topk_from_loo(loo_csv, args.k)
    subsets: Dict[str, List[str]] = {"proposed": proposed_terms}
    with open(out_dir / "subsets_used.txt", "w", encoding="utf-8") as f:
        f.write(f"k = {args.k}\n")
        f.write(f"Source LOO     : {loo_csv}\n")
        f.write(f"Basis branch   : {args.basis_branch}\n")
        f.write("Term filter    : drop only k1₀..k1₃ (not representable as scalar ψ);\n")
        f.write("                 iso base shapes (b_th E_θ², b_z E_z², b_θz <E_θ><E_z>) are kept.\n\n")
        f.write(f"proposed (top-{args.k}): " + ", ".join(proposed_terms) + "\n")
    print("[1/3] Proposed equation term set:")
    print(f"      proposed (top-{args.k}): {proposed_terms}")

    # Apply recruitment sigmoid bounds module-wide BEFORE any Collagen4Fam is
    # constructed (they are read at lambda_lb()/lambda_ub() time but the model
    # was building with default bounds previously; do this once up front).
    if args.recruit_start_min >= args.recruit_start_max:
        raise SystemExit("--recruit_start_min must be < --recruit_start_max")
    if args.recruit_end_min >= args.recruit_end_max:
        raise SystemExit("--recruit_end_min must be < --recruit_end_max")
    J.set_recruitment_lb_bounds(args.recruit_start_min, args.recruit_start_max)
    J.set_recruitment_ub_bounds(args.recruit_end_min,   args.recruit_end_max)
    print(f"      Recruitment:  start λ_lb ∈ [{args.recruit_start_min}, "
          f"{args.recruit_start_max}], init={args.recruit_start_init} "
          f"({args.recruit_start_mode}); "
          f"end λ_ub ∈ [{args.recruit_end_min}, {args.recruit_end_max}], learned.")

    FlatModel = make_flat_model(J, branch=args.basis_branch)
    recruit_learn   = (args.recruit_start_mode == "learn")
    dispersion_learn = (args.dispersion_mode == "learn")
    Flat = lambda term_list: FlatModel(
        term_list=term_list,
        basis_branch=args.basis_branch,
        recruit_start_init=args.recruit_start_init,
        recruit_start_learn=recruit_learn,
        nonneg_fiber_terms=args.nonneg_fiber_terms,
        dispersion_init=args.dispersion_init,
        dispersion_learn=dispersion_learn,
    )
    print(f"      Weights: nonneg_fiber_terms={args.nonneg_fiber_terms}, w_l2={args.w_l2}")
    print(f"      Dispersion: κ init={args.dispersion_init:.3f} "
          f"({args.dispersion_mode}); κ_max=1/3≈0.333")

    specimens = cts.discover_specimens(root)
    if args.only_age:
        specimens = [s for s in specimens if s[0].lower() == args.only_age.lower()]
    if args.only_animal:
        specimens = [s for s in specimens if s[1].lower() == args.only_animal.lower()]
    if not specimens:
        raise SystemExit(
            f"No specimens left after filters "
            f"(--only_age={args.only_age!r}, --only_animal={args.only_animal!r}).")
    if args.only_age or args.only_animal:
        print(f"      Filter active: {len(specimens)} specimen(s) selected — "
              f"{[(a, n) for a, n, _ in specimens]}")

    print(f"[2/3] Fitting flat SEDF per animal (epochs={args.epochs}, lr={args.lr}, "
          f"robust={args.robust}, δ={args.robust_delta}, "
          f"wtheta_boost={args.wtheta_boost}) ...")
    metrics, preds = refit_flat(J, cts, Flat, specimens, subsets,
                                epochs=args.epochs, lr=args.lr,
                                robust_kind=args.robust,
                                robust_delta=args.robust_delta,
                                wtheta_boost=args.wtheta_boost,
                                w_l2=args.w_l2,
                                recruit_ref_stretch_mode=args.recruit_ref_stretch_mode,
                                recruit_ref_stretch_override=args.recruit_ref_stretch)
    if metrics.empty:
        raise SystemExit("No animals fit successfully.")

    metrics.to_csv(out_dir / "refit_metrics.csv", index=False)
    summary = (metrics.groupby("subset")
               .agg(n_animals=("animal", "nunique"),
                    mean_rmse_theta=("rmse_theta", "mean"),
                    median_rmse_theta=("rmse_theta", "median"),
                    mean_rmse_z=("rmse_z", "mean"),
                    median_rmse_z=("rmse_z", "median"),
                    mean_rmse_combined=("rmse_combined", "mean"),
                    median_rmse_combined=("rmse_combined", "median"),
                    mean_r2_theta=("r2_theta", "mean"),
                    median_r2_theta=("r2_theta", "median"),
                    mean_r2_z=("r2_z", "mean"),
                    median_r2_z=("r2_z", "median"))
               .reset_index())
    summary.to_csv(out_dir / "refit_summary.csv", index=False)
    with open(out_dir / "refit_summary.txt", "w", encoding="utf-8") as f:
        f.write("Flat-symbolic SEDF refit on LOO/coeff_max reduced term set\n")
        f.write("(recruited I4/I8 basis, no base+tweak split)\n")
        f.write("=" * 80 + "\n\n")
        for lbl, terms in subsets.items():
            f.write(f"{lbl}: " + ", ".join(terms) + "\n")
        f.write("\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

    print("[3/3] Plotting Experimental vs Proposed Equation per animal ...")
    plot_proposed_equation(preds, out_dir)

    final_eq_path = write_final_equation(metrics, out_dir)
    print(f"      Wrote {final_eq_path.name}")

    final_banner = (f"All outputs in: {out_dir}\n"
                    f"{summary.to_string(index=False)}\n")
    with open(out_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(final_banner)
    print(f"\n{final_banner}")


if __name__ == "__main__":
    main()
