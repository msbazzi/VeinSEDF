#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eq9 (+Gent or PoleZero) with fiber recruitment (half-normal over slack stretches).
- Stable stresses via autograd (compute σ from W with gradients).
- Robust weighting; equal weight per physical test (original + resampled balanced).
- Optional fiber dispersion (GOH κ) and per-family knee width τ.
- Half-normal recruitment Γ_s(λ_s) on λ_s ∈ (1, λ_ub] with parameter σ_s, integrated by quadrature.
- VeinSEDFDiscovered: Eq9 base + 9 free cross-term weights, single-stage fit.
"""

import os, re, glob, argparse, math, copy
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

# ============================== Globals ===============================
torch.set_default_dtype(torch.float64)
EXP_ARG_MAX = 500.0
ZERO_TRANSITION_EPS = 1e-2
RECRUIT_LB_MIN = 0.3
RECRUIT_LB_MAX = 1.5
RECRUIT_UB_MIN = 0.5
RECRUIT_UB_MAX = 1.80

# ============================== Variant Configs ===============================
VARIANT_CONFIGS = {
    "a": dict(iso="neohookean", recruit=False, disp=False,
              label="a: NeoHookean+4Fib (no rec, no disp)"),
    "b": dict(iso="poly",       recruit=False, disp=False,
              label="b: Quad+4Fib (no rec, no disp)"),
    "c": dict(iso="poly",       recruit=True,  disp=False,
              label="c: Quad+4Fib (rec, no disp)"),
    "d": dict(iso="poly",       recruit=False, disp=True,
              label="d: Quad+4Fib (no rec, disp)"),
    "e": dict(iso="poly",       recruit=True,  disp=True,
              label="e: Quad+4Fib (rec, disp)"),
    "h": dict(iso="poly",       recruit=True,  disp=True,
              label="h: VeinDiscovered (Quad+4Fib+rec+disp+cross)"),
}
# ============================== Helpers ===============================
def _raw_from_alpha(x):  # inverse of 0.2 + 7.8*sigmoid(raw)
    s = (x - 0.2)/7.8; s = np.clip(s, 1e-6, 1-1e-6); return float(np.log(s/(1-s)))

def _raw_from_bounded(x, lo: float, hi: float):
    s = (x - lo) / (hi - lo)
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return float(np.log(s / (1 - s)))

def set_recruitment_lb_bounds(lo: float, hi: float):
    global RECRUIT_LB_MIN, RECRUIT_LB_MAX
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid recruitment bounds: lo={lo}, hi={hi}")
    RECRUIT_LB_MIN = lo
    RECRUIT_LB_MAX = hi

def set_recruitment_ub_bounds(lo: float, hi: float):
    global RECRUIT_UB_MIN, RECRUIT_UB_MAX
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid recruitment upper bounds: lo={lo}, hi={hi}")
    RECRUIT_UB_MIN = lo
    RECRUIT_UB_MAX = hi

def set_seed(seed: Optional[int]):
    if seed is None: return
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def expm1_clamped(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(torch.clamp(x, max=EXP_ARG_MAX))

def softplus_pos(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) + 1e-8  # strictly > 0

def _inv_softplus_pos(x, eps: float = 1e-8):
    x = np.asarray(x, dtype=np.float64)
    y = np.maximum(x - eps, 1e-12)
    return np.log(np.expm1(y))

def smooth_xlog1p_pos(x: torch.Tensor, tau: float = ZERO_TRANSITION_EPS) -> torch.Tensor:
    xp = smooth_relu_zero(x, eps=tau)
    return xp * torch.log1p(xp)

def smooth_abs_zero(x: torch.Tensor, eps: float = ZERO_TRANSITION_EPS) -> torch.Tensor:
    e = torch.tensor(eps, dtype=x.dtype, device=x.device)
    return torch.sqrt(x * x + e * e) - e

def smooth_relu_zero(x: torch.Tensor, eps: float = ZERO_TRANSITION_EPS) -> torch.Tensor:
    # C1 approximation of Macauley brackets with zero value at x=0.
    return 0.5 * (x + smooth_abs_zero(x, eps=eps))

def smooth_pos(x: torch.Tensor, tau: float = 1e-3) -> torch.Tensor:
    # Smooth approximation of max(x, 0) with temperature tau
    t = torch.tensor(tau, dtype=x.dtype, device=x.device)
    return torch.nn.functional.softplus(x / t) * t

def _inv_std_safe(arr: Optional[torch.Tensor], eps: float = 1e-6, cap: float = 100.0) -> float:
    if arr is None or arr.numel() == 0: return 0.0
    s = float(arr.std().item())
    if not np.isfinite(s) or s < eps: return 0.0
    return float(min(1.0 / s, cap))

def C_from_lambdas(lam: torch.Tensor) -> torch.Tensor:
    lam2 = lam**2
    C = torch.zeros(lam.shape[:-1] + (3,3), dtype=lam.dtype, device=lam.device)
    C[...,0,0] = lam2[...,0]; C[...,1,1] = lam2[...,1]; C[...,2,2] = lam2[...,2]
    return C

def I1(C: torch.Tensor) -> torch.Tensor: return torch.einsum("...ii->...", C)
def I2(C: torch.Tensor) -> torch.Tensor:
    trC = I1(C)
    trC2 = torch.einsum("...ij,...ji->...", C, C)
    return 0.5 * (trC * trC - trC2)
def I4(C: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...i,i->...", torch.einsum("...ij,j->...i", C, n), n)
def I5(C: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """I5 = n·C²n = ||Cn||². Genuinely new info only for off-axis (helical) fibers."""
    Cn = torch.einsum("...ij,j->...i", C, n)
    return torch.einsum("...i,...i->...", Cn, Cn)
def I8(C: torch.Tensor, n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
    """I8 = n1·Cn2. For ±helical pair: λθ²cos²α − λz²sin²α (signed circ-vs-axial difference)."""
    return torch.einsum("...i,i->...", torch.einsum("...ij,j->...i", C, n2), n1)

def cauchy_from_W(W: torch.Tensor, lam: torch.Tensor, create_graph=True) -> Tuple[torch.Tensor, torch.Tensor]:
    if lam.ndim < 1 or lam.size(-1) != 3: raise ValueError("lam must be (...,3)")
    if not lam.requires_grad: lam = lam.requires_grad_(True)
    Wsum = W if W.ndim == 0 else W.sum()
    (dW_dlam,) = torch.autograd.grad(Wsum, lam, create_graph=create_graph, retain_graph=create_graph)
    lamθ, lamz, lamr = lam.unbind(-1); dth, dz, dr = dW_dlam.unbind(-1)
    p = lamr * dr
    σθ = lamθ * dth - p; σz = lamz * dz - p
    return σθ, σz

# ======================= Fiber base (+ recruitment) ====================
class FiberBase(nn.Module):
    """
    4 families: ± helical + circumferential eθ + axial ez.
    - GOH dispersion κ for helicals
    - per-family smooth-ReLU temperature τm (knee width)
    - helix slack λ0h; eθ/ez slack λ0 (used only if recruitment is OFF)
    - NEW: half-normal recruitment over slack λs ∈ (1, λub] with σs
    """
    def __init__(self, nfam=4, denom=4.0, use_dispersion=True, use_recruitment=True, n_quad=14):
        super().__init__()
        self.nfam, self.denom = nfam, denom
        self.use_dispersion = use_dispersion
        self.use_recruitment = use_recruitment
        self.n_quad = n_quad

        # helix angle + family coefficients
        self._α  = nn.Parameter(torch.tensor(35.0))                    # helix angle (deg)
        self._k1 = nn.Parameter(torch.full((nfam,), 0.6))              # scale
        self._k2 = nn.Parameter(torch.full((nfam,), 0.1))              # sharpness

        # classic single-slack (fallback when recruitment off)
        self._lam0_raw  = nn.Parameter(torch.full((nfam,), -2.0))       # lambda raw is mapped into a sigmoid later λ0 ∈ [1.00,1.40] (eθ, ez)
        self._lam0h_raw = nn.Parameter(torch.full((2,),   -2.8))        # helicals λ0h ∈ [1.00,1.20]

        # --- recruitment knobs (per-family) ---
        self.dist_type = "beta"  # options: "halfnormal", "lognormal", "beta"


        if use_dispersion and nfam >= 2:
            self._kappa = nn.Parameter(torch.full((2,), 0.1))          # GOH dispersion for helicals

        # learnable per-family τ in [tau_min, tau_max]
        self._tau_raw = nn.Parameter(torch.full((nfam,), -4.0))
        self.tau_min, self.tau_max = 1e-2, 0.15

        # recruitment params (per-family)
        self._sigma_s_raw  = nn.Parameter(torch.full((nfam,), -2.0))  # σs in ~[0.01,0.20]
        self._lambda_lb_raw= nn.Parameter(torch.full((nfam,), -4.0, dtype=torch.float64))  # per-family λlb, mapped to [0.90,1.25]
        self._lambda_ub_raw= nn.Parameter(torch.full((nfam,), 1.0))  # λub in ~[RECRUIT_UB_MIN, RECRUIT_UB_MAX]
        self.register_buffer("_recruit_ref_stretch", torch.tensor(1.0, dtype=torch.float64))

        # lognormal on (λs-1): μ, σ (per-family)
        self._logn_mu_raw   = nn.Parameter(torch.full((nfam,), -2.2))   # maps to μ ∈ [-3, +1]
        self._logn_sig_raw  = nn.Parameter(torch.full((nfam,), -2.0))   # σ ∈ [0.05, 0.60]

        # beta on u = (λs-1)/(λ_ub-1): α,β > 0 (per-family)
        self._beta_a_raw    = nn.Parameter(torch.full((nfam,),  0.2))   # α ∈ (0.2, 8]
        self._beta_b_raw    = nn.Parameter(torch.full((nfam,),  0.2))   # β ∈ (0.2, 8]

        # Gauss–Legendre nodes on [0,1] (12-pt)
        x_gl = torch.tensor([0.005299,0.027712,0.067184,0.122299,0.191061,0.270991,
                            0.359009,0.451846,0.546156,0.638756,0.726781,0.807785], dtype=torch.float64)
        w_gl = torch.tensor([0.013576,0.031126,0.047579,0.062314,0.074797,0.084578,
                            0.091322,0.094827,0.095038,0.092045,0.086079,0.077500], dtype=torch.float64)
        self.n_quad = len(x_gl)
        self.register_buffer("quad_x", x_gl)
        self.register_buffer("quad_w", w_gl)  # GL weights already sum to 1 for [0,1] interval
        
    # maps - encode physics-aware bounds while keeping the training landscape smooth and stable
    def tau(self) -> torch.Tensor:
        s = torch.sigmoid(self._tau_raw)
        return self.tau_min + (self.tau_max - self.tau_min) * s
    def sigma_s(self) -> torch.Tensor:
        s = torch.sigmoid(self._sigma_s_raw)
        return 0.01 + 0.19 * s
    
    def logn_mu(self):       # μ ∈ [-3, +1]
        s = torch.tanh(self._logn_mu_raw); return -3.0 + 2.0*(s+1.0)        # (nfam,)
        
    def logn_sig(self):      # σ ∈ [0.05, 0.60]
        s = torch.sigmoid(self._logn_sig_raw); return 0.12 + 0.48*s         # (nfam,)

    def beta_ab(self):       # α,β ∈ (0.2, 8]
        a = 0.2 + 7.8*torch.sigmoid(self._beta_a_raw)
        b = 0.2 + 7.8*torch.sigmoid(self._beta_b_raw)
        return a, b
        
    def lambda_ub(self) -> torch.Tensor:
        s = torch.sigmoid(self._lambda_ub_raw)
        return RECRUIT_UB_MIN + (RECRUIT_UB_MAX - RECRUIT_UB_MIN) * s
    def lambda_lb(self) -> torch.Tensor:
        s = torch.sigmoid(self._lambda_lb_raw)
        return RECRUIT_LB_MIN + (RECRUIT_LB_MAX - RECRUIT_LB_MIN) * s
    def recruit_ref_stretch(self) -> torch.Tensor:
        return self._recruit_ref_stretch
    def effective_lambda_lb(self, lam_z: Optional[torch.Tensor] = None, branch: str = "theta", m: int = 0) -> torch.Tensor:
        lam_lb_ref = self.lambda_lb()[m].reshape(())
        if lam_z is None or branch != "theta":
            return lam_lb_ref
        ref = torch.clamp(self.recruit_ref_stretch().to(lam_z), min=1e-6)
        lam_lb_eff = lam_lb_ref.to(lam_z) * (lam_z / ref)
        return torch.clamp(lam_lb_eff, min=RECRUIT_LB_MIN, max=RECRUIT_LB_MAX)

    def dirs(self) -> List[torch.Tensor]:
        a_deg = torch.sigmoid(self._α) * 80.0 + 5.0        # 5°..85°
        a = a_deg * (math.pi / 180.0)
        c, s = torch.cos(a), torch.sin(a)
        dir1 = torch.stack([ c,  s, torch.zeros((), dtype=a.dtype, device=a.device)], 0)
        dir2 = torch.stack([ c, -s, torch.zeros((), dtype=a.dtype, device=a.device)], 0)
        dirs = [dir1, dir2]
        if self.nfam == 4:
            eθ = torch.tensor([1., 0., 0.], dtype=a.dtype, device=a.device)
            ez = torch.tensor([0., 1., 0.], dtype=a.dtype, device=a.device)
            dirs += [eθ, ez]
        return dirs

    @staticmethod
    def _smoothpos(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x / tau) * tau

    def fiber_sum(self, C: torch.Tensor, branch: str = "theta") -> torch.Tensor:
        out = 0.0
        I1v   = I1(C) - 3.0
        dirs  = self.dirs()
        tau   = self.tau()                                  # [nfam]
        lam0  = 1.00 + 0.40 * torch.sigmoid(self._lam0_raw) # [nfam]  (eθ, ez)
        lam0h = 1.00 + 0.20 * torch.sigmoid(self._lam0h_raw)# [2]     (helicals)
        lam_z = torch.sqrt(torch.clamp(C[...,1,1], min=1e-12))

        for m, n in enumerate(dirs):
            k1  = softplus_pos(self._k1[m])
            k2  = torch.clamp(softplus_pos(self._k2[m]), max=200.0)
            I4m = I4(C, n)

            if self.use_recruitment:
                # effective λ = sqrt(I4-like); for helicals include GOH Q shift
                if self.use_dispersion and m < 2:
                    kappa = torch.clamp(torch.sigmoid(self._kappa[m])*(1/3 - 1e-4), max=1/3 - 1e-4)
                    Q  = kappa*I1v + (1.0 - 3.0*kappa) * (I4m - 1.0)
                    lam_eff = torch.sqrt(torch.clamp(1.0 + Q, min=1e-12))
                else:
                    lam_eff = torch.sqrt(torch.clamp(I4m, min=1e-12))
                xi = self._recruit_expectation(lam_eff, lam_z, tau[m], m, branch=branch)
            else:
                # fallback single-slack smoothed positive
                if self.use_dispersion and m < 2:
                    kappa = torch.clamp(torch.sigmoid(self._kappa[m])*(1/3 - 1e-4), max=1/3 - 1e-4)
                    Q  = kappa*I1v + (1.0 - 3.0*kappa) * (I4m - lam0h[m]**2)
                    xi = self._smoothpos(Q, tau[m])
                else:
                    xi = self._smoothpos(I4m - lam0[m]**2, tau[m])

            xi  = torch.clamp(xi, 0.0, 5.0)
            arg = torch.clamp(xi**2, 0.0, 50.0)
            out = out + (k1 / (self.denom * k2)) * (expm1_clamped(k2 * arg))
        return out
    
    def _pdf_lambda_s(self, lam_s: torch.Tensor, m: int, lam_lb_m: torch.Tensor, lam_ub_m: torch.Tensor):
    # returns pdf(λs) on (1, λub], normalized over that interval
        lam_lb_q = lam_lb_m.unsqueeze(-1) if lam_lb_m.ndim > 0 else lam_lb_m
        lam_ub_q = lam_ub_m.unsqueeze(-1) if lam_ub_m.ndim > 0 else lam_ub_m
        if self.dist_type == "halfnormal":
            # Truncated half-normal on t = (λs-λlb)/σ ∈ [0, a] where a = (λub-λlb)/σ
            # Z = P(0 < T < a) for T~N(0,1) = Φ(a)-Φ(0) = 0.5*erf(a/sqrt(2))
            # This makes phi/(σ*Z) equivalent to truncated half-normal on t>=0
            sig = self.sigma_s()[m].reshape(()) ; 
            a = (lam_ub_m - lam_lb_m)/sig
            t = (lam_s - lam_lb_q)/sig
            phi = torch.exp(-0.5*t*t) / math.sqrt(2.0*math.pi)          # N(0,1) density
            Z   = 0.5*torch.clamp(torch.erf(a/math.sqrt(2.0)), min=1e-9)    # Φ(a)-Φ(0) normalization
            Zq = Z.unsqueeze(-1) if Z.ndim > 0 else Z
            return phi / (sig*Zq)

        elif self.dist_type == "lognormal":
            # shifted lognormal on y=λs-λlb ∈ (0, λub-λlb], truncated
            mu = self.logn_mu()[m].reshape(())
            sg = self.logn_sig()[m].reshape(())
            y  = torch.clamp(lam_s - lam_lb_q, min=1e-12)
            base = torch.exp(-0.5*((torch.log(y)-mu)/sg)**2)/(y*sg*math.sqrt(2.0*math.pi))
            # truncation constant over y∈(0, yub], yub=λub-λlb
            yub = torch.clamp(lam_ub_m - lam_lb_m, min=1e-12)
            Za  = 0.5*(1.0 + torch.erf((torch.log(yub)-mu)/(sg*math.sqrt(2.0))))  # Φ(log(yub))
            Zaq = Za.unsqueeze(-1) if Za.ndim > 0 else Za
            return base / torch.clamp(Zaq, min=1e-9)

        else:  # "beta" on u=(λs-λlb)/(λub-λlb), u∈(0,1]
            a,b = self.beta_ab()
            a_m, b_m = a[m].reshape(()), b[m].reshape(())
            yub = torch.clamp(lam_ub_m - lam_lb_m, min=1e-12)
            yub_q = yub.unsqueeze(-1) if yub.ndim > 0 else yub
            u = torch.clamp((lam_s - lam_lb_q)/yub_q, 1e-12, 1.0-1e-12)
            # pdf_u(u) = u^{α-1}(1-u)^{β-1}/B(α,β); pdf_λ = pdf_u(u) * (du/dλ) = ... / (λub-1)
            logB = torch.lgamma(a_m) + torch.lgamma(b_m) - torch.lgamma(a_m+b_m)
            pdf_u = torch.exp((a_m-1.0)*torch.log(u) + (b_m-1.0)*torch.log(1.0-u) - logB)
            return pdf_u / yub_q
        
    def _recruit_expectation(self, lam_mag: torch.Tensor, lam_z: torch.Tensor, tau_m: torch.Tensor, m: int, branch: str = "theta") -> torch.Tensor:
        # Use λ_lb as the in-vivo recruitment start, then shift it with the companion axial prestretch.
        lam_lb_m = self.effective_lambda_lb(lam_z.to(lam_mag), branch=branch, m=m).to(lam_mag)
        lam_ub_m = torch.clamp(self.lambda_ub()[m].reshape(()).to(lam_mag), min=1.001)
        lam_lb_floor = torch.full_like(lam_lb_m, 0.90)
        lam_lb_cap = lam_ub_m - 1e-3
        lam_lb_m = torch.maximum(lam_lb_m, lam_lb_floor)
        lam_lb_m = torch.minimum(lam_lb_m, lam_lb_cap)
        span = lam_ub_m - lam_lb_m

        # map nodes u∈[0,1] → λs ∈ (λlb, λub] for Gauss–Legendre quadrature
        u  = self.quad_x # preload quadrature nodes
        w  = self.quad_w
        lam_s = lam_lb_m.unsqueeze(-1) + span.unsqueeze(-1) * u
        pdf   = self._pdf_lambda_s(lam_s, m, lam_lb_m, lam_ub_m)
        # Precompute effective quadrature weights once: pdf * jacobian * GL weights
        # expectation ∫ f(λs) pdf(λs) dλs ≈ Σ f(λs_i) pdf(λs_i) (λub-1) w_i
        quad_weight = (pdf * span.unsqueeze(-1) * w).to(lam_mag)   # (..., n_quad)

        lm  = lam_mag.unsqueeze(-1) / lam_s                # (..., n_quad)
        act = self._smoothpos(lm - 1.0, tau_m.unsqueeze(-1))
        ex  = torch.sum(act * quad_weight, dim=-1)  # return expected activation for family m
        return ex

# ======================= Isotropic bases (Eq9/Gent/PoleZero) ==========
class Collagen4Fam(FiberBase):
    """Collagen: 4 fiber families with optional dispersion + recruitment."""
    def __init__(self, use_recruitment=True, use_dispersion=True, **kw):
        super().__init__(nfam=4, denom=4.0, use_dispersion=use_dispersion, use_recruitment=use_recruitment, **kw)

    def energy(self, lam: torch.Tensor, branch: str = "theta") -> torch.Tensor:
        C = C_from_lambdas(lam)
        return self.fiber_sum(C, branch=branch)

    def symbolic_dirs(self):
        dirs = self.dirs()
        return {"theta": dirs[2], "z": dirs[3], "d1": dirs[0], "d2": dirs[1]}

class TwoFiberRecruit(FiberBase):
    """Two anisotropic families aligned with circumferential and axial directions."""
    def __init__(self, use_recruitment=True, **kw):
        super().__init__(nfam=2, denom=2.0, use_dispersion=False, use_recruitment=use_recruitment, **kw)

    def dirs(self) -> List[torch.Tensor]:
        ref = self._k1
        eθ = torch.tensor([1., 0., 0.], dtype=ref.dtype, device=ref.device)
        ez = torch.tensor([0., 1., 0.], dtype=ref.dtype, device=ref.device)
        return [eθ, ez]

    def energy(self, lam: torch.Tensor, branch: str = "theta") -> torch.Tensor:
        return self.fiber_sum(C_from_lambdas(lam), branch=branch)

    def symbolic_dirs(self):
        dirs = self.dirs()
        return {"theta": dirs[0], "z": dirs[1], "d1": None, "d2": None}

class IsoEq9Quadratic(nn.Module):
    """Quadratic-only Eq9 isotropic part."""
    def __init__(self):
        super().__init__()
        self._bth = nn.Parameter(torch.tensor(1.0))
        self._bz  = nn.Parameter(torch.tensor(1.0))
        self.bthz = nn.Parameter(torch.tensor(0.5))

    def energy(self, lam: torch.Tensor) -> torch.Tensor:
        E_th = 0.5 * (lam[...,0]**2 - 1.0)
        E_z  = 0.5 * (lam[...,1]**2 - 1.0)
        bth  = torch.clamp(softplus_pos(self._bth), max=500.0)
        bz   = torch.clamp(softplus_pos(self._bz),  max=500.0)
        bthz = torch.clamp(self.bthz, max=500.0)
        E_th_pos = smooth_pos(E_th,  ZERO_TRANSITION_EPS)
        E_z_pos  = smooth_pos(E_z,  ZERO_TRANSITION_EPS)
        return bth*E_th**2 + bz*E_z**2 + bthz*E_th_pos*E_z_pos

class ElastinNeoHookean(nn.Module):
    def __init__(self):
        super().__init__()
        self._mu = nn.Parameter(torch.tensor(1.0))

    def energy(self, lam: torch.Tensor) -> torch.Tensor:
        C = C_from_lambdas(lam)
        I1v = I1(C) - 3.0
        mu = softplus_pos(self._mu)
        return 0.5 * mu * I1v

class IsoGent(nn.Module):
    def __init__(self):
        super().__init__()
        self._mu = nn.Parameter(torch.tensor(2.0))
        self._Jm = nn.Parameter(torch.tensor(60.0))

    def energy(self, lam: torch.Tensor) -> torch.Tensor:
        C = C_from_lambdas(lam); I1v = I1(C) - 3.0
        mu = softplus_pos(self._mu); Jm = softplus_pos(self._Jm) + 1e-6
        x = torch.clamp(I1v / Jm, max=0.98)
        return -0.5 * mu * Jm * torch.log1p(-x)

class IsoPoleZero(nn.Module):
    def __init__(self):
        super().__init__()
        self._k = nn.Parameter(torch.tensor([2.0, 0.3, 0.3], dtype=torch.float64))
        self._a = nn.Parameter(torch.tensor([0.28, 0.43, 1.04], dtype=torch.float64))
        self._b = nn.Parameter(torch.tensor([1.6, 2.48, 0.40], dtype=torch.float64))
        self.bthz = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    @staticmethod
    def _pz_term(e, k, a, b):
        e_abs = torch.abs(e)
        k  = softplus_pos(k)
        a = softplus_pos(a) + 1e-6
        b = softplus_pos(b) + 1e-6
        denom = torch.clamp(a - e_abs, min=5e-7)
        return k * (e*e) / (denom**b)

    def energy(self, lam: torch.Tensor) -> torch.Tensor:
        e_th = 0.5*(lam[...,0]**2 - 1.0)
        e_z  = 0.5*(lam[...,1]**2 - 1.0)
        e_r  = 0.5*(lam[...,2]**2 - 1.0)
        return (self._pz_term(e_th, self._k[0], self._a[0], self._b[0]) +
                self._pz_term(e_z,  self._k[1], self._a[1], self._b[1]) +
                self._pz_term(e_r,  self._k[2], self._a[2], self._b[2]) +
                torch.clamp(self.bthz, max=5.0) * torch.clamp(e_th, min=0.0) * torch.clamp(e_z, min=0.0))

def build_iso_module(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "poly":
        return IsoEq9Quadratic()
    if kind == "gent":
        return IsoGent()
    if kind == "polezero":
        return IsoPoleZero()
    if kind == "neohookean":
        return ElastinNeoHookean()
    raise ValueError(f"Unsupported isotropic model: {kind}")

def build_aniso_module(kind: str, *, use_recruitment: bool, use_dispersion: bool) -> nn.Module:
    kind = kind.lower()
    if kind == "4fiber":
        return Collagen4Fam(use_recruitment=use_recruitment, use_dispersion=use_dispersion)
    if kind == "2fiber":
        return TwoFiberRecruit(use_recruitment=use_recruitment)
    raise ValueError(f"Unsupported anisotropic model: {kind}")

class DecomposedSEDFBase(nn.Module):
    """
    Additive iso + aniso base model with no extra cross terms.
    Used for variants a-e in the comparison (single-stage fit, no symbolic terms).
    """
    def __init__(self, iso_model="poly", aniso_model="4fiber", *, use_recruitment=True,
                 use_dispersion=True):
        super().__init__()
        self.iso_model = iso_model
        self.aniso_model = aniso_model
        self.iso = build_iso_module(iso_model)
        self.aniso = build_aniso_module(aniso_model, use_recruitment=use_recruitment,
                                        use_dispersion=use_dispersion)

    def energy(self, lam: torch.Tensor, branch: str = "theta") -> torch.Tensor:
        return self.iso.energy(lam) + self.aniso.energy(lam, branch=branch)

    def forward(self, lam: torch.Tensor, create_graph: bool = True,
                branch: str = "theta"):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        W = self.energy(lam, branch=branch)
        sigma_th, sigma_z = cauchy_from_W(W, lam, create_graph=create_graph)
        return W, sigma_th, sigma_z

class VeinSEDFDiscovered(nn.Module):
    """
    Vein SEDF with discovered cross-terms fitted in a single stage.
    W = W_iso (IsoEq9Quadratic) + W_fib (4-fiber) + W_disc (9 weighted invariant terms).
    Same forward interface as Eq9: returns (W, sigma_th, sigma_z).
    """
    def __init__(self, iso_model="poly", aniso_model="4fiber", *, use_recruitment=True,
                 use_dispersion=True):
        super().__init__()
        self.iso_model = iso_model
        self.aniso_model = aniso_model
        self.iso = build_iso_module(iso_model)
        self.aniso = build_aniso_module(aniso_model, use_recruitment=use_recruitment,
                                        use_dispersion=use_dispersion)
        # Discovered cross-term weights; init to 0 so model starts identical to base.
        # Solved analytically by linreg_vein_weights() after the base is trained.
        self._w1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # I1 - 3
        self._w2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # I2 - 3
        self._w3 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # <E_θ>
        self._w4 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # <E_z>
        self._w5 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # (I2-3)·log1p(I2-3)
        self._w6 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # <I4z - 1>
        self._w7 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # <I4D1 - 1>
        self._w8 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # <I4D2 - 1>
        self._w9 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # I8(D1,D2)
        self._w10 = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))  # <I2-3>·<E_z>

    # Ordered (param_attr, human_name) pairs — single source of truth for
    # linear-regression assembly and parameter file output.
    DISC_WEIGHTS = (
        ("_w1",  "I1-3"),
        ("_w2",  "I2-3"),
        ("_w3",  "<E_θ>"),
        ("_w4",  "<E_z>"),
        ("_w5",  "<I2-3>·log1p(<I2-3>)"),
        ("_w6",  "<I4z-1>"),
        ("_w7",  "<I4D1-1>"),
        ("_w8",  "<I4D2-1>"),
        ("_w9",  "I8(D1,D2)"),
        ("_w10", "<I2-3>·<E_z>"),
    )

    @torch.no_grad()
    def set_disc_weights(self, values):
        """Set _w1.._w10 from an iterable of 10 floats."""
        for (attr, _), v in zip(self.DISC_WEIGHTS, values):
            getattr(self, attr).fill_(float(v))

    def disc_weight_params(self):
        """Return list of nn.Parameter handles for _w1.._w10 in order."""
        return [getattr(self, attr) for attr, _ in self.DISC_WEIGHTS]

    def energy(self, lam: torch.Tensor, branch: str = "theta") -> torch.Tensor:
        C = C_from_lambdas(lam)

        W_iso = self.iso.energy(lam)
        W_fib = self.aniso.energy(lam, branch=branch)

        I1v = I1(C) - 3.0
        I2v = I2(C) - 3.0
        E_th = 0.5 * (lam[..., 0] ** 2 - 1.0)
        E_z  = 0.5 * (lam[..., 1] ** 2 - 1.0)

        # Helical directions are learned (depend on self.aniso._α)
        dirs = self.aniso.dirs()   # [d1, d2, eθ, ez]
        d1, d2 = dirs[0], dirs[1]
        ez = torch.tensor([0., 1., 0.], dtype=lam.dtype, device=lam.device)

        I4z  = I4(C, ez) - 1.0
        I4d1 = I4(C, d1) - 1.0
        I4d2 = I4(C, d2) - 1.0
        I8v  = I8(C, d1, d2)

        sr = smooth_relu_zero
        #logI2 = sr(I2v) * torch.log1p(sr(I2v))
        logI2 = sr(I2v) * torch.log1p(sr(I2v))

        W_disc = (self._w1 * sr(I1v)
                  + self._w2 * sr(I2v)
                  + self._w3 * sr(E_th)
                  + self._w4 * sr(E_z)
                  + self._w5 * logI2
                  + self._w6 * sr(I4z)
                  + self._w7 * sr(I4d1)
                  + self._w8 * sr(I4d2)
                  + self._w9 * I8v
                  + self._w10 * sr(I2v) * sr(E_z))
        # W_disc = (self._w1 * sr(I1v)
        #           + self._w2 * sr(I2v)
        #           + self._w3 * (E_th)
        #           + self._w4 * (E_z)
        #           + self._w5 * logI2
        #           + self._w6 * (I4z)
        #           + self._w7 * (I4d1)
        #           + self._w8 * (I4d2)
        #           + self._w9 * I8v
        #           + self._w10 * sr(I2v) * (E_z))

        return W_iso + W_fib + W_disc

    def forward(self, lam: torch.Tensor, create_graph: bool = True,
                branch: str = "theta"):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        W = self.energy(lam, branch=branch)
        σθ, σz = cauchy_from_W(W, lam, create_graph=create_graph)
        return W, σθ, σz


class Elastin2LinearFibers(nn.Module):
    """
    Two elastin fiber families: eθ and ez.
    Energy ~ k_e * smoothpos(I4-1)^2 (small stiffness, linear-ish at low strain).
    """
    def __init__(self):
        super().__init__()
        self._k = nn.Parameter(torch.tensor([0.2, 0.2], dtype=torch.float64))  # θ,z
        self._tau_raw = nn.Parameter(torch.tensor([-4.0, -4.0], dtype=torch.float64))
        self.tau_min, self.tau_max = 1e-2, 0.5

    def tau(self):
        s = torch.sigmoid(self._tau_raw)
        return self.tau_min + (self.tau_max - self.tau_min)*s

    @staticmethod
    def _smoothpos(x, tau):
        return torch.nn.functional.softplus(x/tau) * tau

    def energy(self, lam: torch.Tensor) -> torch.Tensor:
        C = C_from_lambdas(lam)
        eθ = torch.tensor([1., 0., 0.], dtype=lam.dtype, device=lam.device)
        ez = torch.tensor([0., 1., 0.], dtype=lam.dtype, device=lam.device)
        I4θ = I4(C, eθ)
        I4z = I4(C, ez)
        tau = self.tau()
        aθ = self._smoothpos(I4θ - 1.0, tau[0])
        az = self._smoothpos(I4z - 1.0, tau[1])
        k  = softplus_pos(self._k)
        return 0.5*k[0]*aθ*aθ + 0.5*k[1]*az*az
class CompositeSEDFBase(nn.Module):
    """
    Slide-21 framework base:
      W_mix = w_iso W_iso + w_aniso W_aniso
    with either
      - normalized weights: w_i >= 0 and sum_i w_i = 1
      - independent weights: w_i >= 0 with no sum-to-one constraint
    """
    def __init__(self, elastin_type="neohookean", use_recruitment=True, phi_constraint="sum1"):
        super().__init__()
        self.iso = IsoEq9Quadratic()
        self.col = Collagen4Fam(use_recruitment=use_recruitment)  # recruitment controlled via parameter
        if elastin_type == "neohookean":
            self.ela = ElastinNeoHookean()
        else:
            self.ela = Elastin2LinearFibers()

        if phi_constraint not in {"sum1", "independent"}:
            raise ValueError(f"Unsupported phi_constraint={phi_constraint!r}")
        self.phi_constraint = phi_constraint
        self._phi_logits = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))

    def phi(self):
        if self.phi_constraint == "sum1":
            return torch.softmax(self._phi_logits, dim=0)
        return softplus_pos(self._phi_logits)

    def phi_constraint_label(self):
        return "sum-to-one" if self.phi_constraint == "sum1" else "independent-nonnegative"

    def energy_parts(self, lam: torch.Tensor):
        W_iso = self.iso.energy(lam)
        W_col = self.col.energy(lam)
        W_ela = self.ela.energy(lam)
        W_aniso = W_col + W_ela
        φ = self.phi()
        W_mix = φ[0]*W_iso + φ[1]*W_aniso
        return W_mix, W_iso, W_aniso, W_col, W_ela, φ

    def forward(self, lam: torch.Tensor, create_graph: bool = True):
        """
        Return (W, σθ, σz) for compatibility with fit_model.
        """
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        W, _, _, _, _, _ = self.energy_parts(lam)
        σθ, σz = cauchy_from_W(W, lam, create_graph=create_graph)
        return W, σθ, σz

class Eq9(FiberBase):
    def __init__(self, **fiber_kw):
        super().__init__(nfam=4, denom=4.0, use_dispersion=True, **fiber_kw)
        self._bth = nn.Parameter(torch.tensor(1.0))
        self._bz  = nn.Parameter(torch.tensor(1.0))
        self.bthz = nn.Parameter(torch.tensor(0.5))
    def energy_parts(self, lam):
        E_th = 0.5 * (lam[...,0]**2 - 1.0)
        E_z  = 0.5 * (lam[...,1]**2 - 1.0)
        bth = torch.clamp(softplus_pos(self._bth), max=5.0)
        bz  = torch.clamp(softplus_pos(self._bz),  max=5.0)
        # Prevent dip near λ≈1: only couple in tensile regime (E≥0)
        bthz = torch.clamp(self.bthz, max=5.0)
        E_th_pos = torch.clamp(E_th, min=0.0)
        E_z_pos  = torch.clamp(E_z,  min=0.0)
        q_iso = bth*E_th**2 + bz*E_z**2 + bthz*E_th_pos*E_z_pos
        q_fib = self.fiber_sum(C_from_lambdas(lam))
        return q_iso, q_fib
    def energy(self, lam): 
        q_iso, q_fib = self.energy_parts(lam); return q_iso + q_fib
    def forward(self, lam, create_graph=True):
        if not lam.requires_grad: lam = lam.requires_grad_(True)
        W = self.energy(lam); σθ, σz = cauchy_from_W(W, lam, create_graph)
        return W, σθ, σz

class Eq9_Gent(FiberBase):
    def __init__(self, **fiber_kw):
        super().__init__(nfam=4, denom=4.0, use_dispersion=True, **fiber_kw)
        self._mu = nn.Parameter(torch.tensor(2.0))
        self._Jm = nn.Parameter(torch.tensor(60.0))
        self.bthz = nn.Parameter(torch.tensor(0.5))
    def energy_parts(self, lam):
        C = C_from_lambdas(lam); I1v = I1(C) - 3.0
        mu = softplus_pos(self._mu); Jm = softplus_pos(self._Jm) + 1e-6
        x = torch.clamp(I1v / Jm, max=0.98)
        q_iso = -0.5 * mu * Jm * torch.log1p(-x)     # Gent isotropic
        q_fib = self.fiber_sum(C)
        return q_iso, q_fib
    def energy(self, lam): 
        q_iso, q_fib = self.energy_parts(lam); return q_iso + q_fib
    def forward(self, lam, create_graph=True):
        if not lam.requires_grad: lam = lam.requires_grad_(True)
        W = self.energy(lam); σθ, σz = cauchy_from_W(W, lam, create_graph)
        return W, σθ, σz

class Eq9_NeoHookean(FiberBase):
    """
    Neo-Hookean isotropic energy + 4 fiber families.
    W_iso = (μ/2) * (I₁ - 3) where I₁ = tr(C) = λθ² + λz² + λr²
    """
    def __init__(self, **fiber_kw):
        super().__init__(nfam=4, denom=4.0, use_dispersion=True, **fiber_kw)
        self._mu = nn.Parameter(torch.tensor(2.0))  # shear modulus
    def energy_parts(self, lam):
        C = C_from_lambdas(lam)
        I1v = I1(C) - 3.0  # I₁ - 3
        mu = softplus_pos(self._mu)
        q_iso = 0.5 * mu * I1v  # Neo-Hookean: (μ/2) * (I₁ - 3)
        q_fib = self.fiber_sum(C)
        return q_iso, q_fib
    def energy(self, lam): 
        q_iso, q_fib = self.energy_parts(lam); return q_iso + q_fib
    def forward(self, lam, create_graph=True):
        if not lam.requires_grad: lam = lam.requires_grad_(True)
        W = self.energy(lam); σθ, σz = cauchy_from_W(W, lam, create_graph)
        return W, σθ, σz

class PoleZero(FiberBase):
    """
    Isotropic pole–zero (normal strains only); fiber part from FiberBase (w/ recruitment).
    """
    def __init__(self, **fiber_kw):
        super().__init__(nfam=4, denom=4.0, use_dispersion=True, **fiber_kw)
        self._k = nn.Parameter(torch.tensor([2.0, 0.3, 0.3], dtype=torch.float64))
        self._a = nn.Parameter(torch.tensor([0.28, 0.43, 1.04], dtype=torch.float64))
        self._b = nn.Parameter(torch.tensor([1.6, 2.48, 0.40], dtype=torch.float64))
        self.bthz = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    @staticmethod
    def _pz_term(e, k, a, b):
        #e_pos = torch.clamp(e, min=0.0) # only tensile part
        e_abs = torch.abs(e)
        k  = softplus_pos(k) 
        a = softplus_pos(a) + 1e-6
        b = softplus_pos(b) + 1e-6
        denom = torch.clamp(a - e_abs, min=5e-7)
        return k * (e*e) / (denom**b)
    def energy_parts(self, lam):
        e_th = 0.5*(lam[...,0]**2 - 1.0) # Green -Lagrange strains using the lambdas
        e_z  = 0.5*(lam[...,1]**2 - 1.0)
        e_r  = 0.5*(lam[...,2]**2 - 1.0)
        q_iso = (self._pz_term(e_th, self._k[0], self._a[0], self._b[0]) +
                 self._pz_term(e_z,  self._k[1], self._a[1], self._b[1]) +
                 self._pz_term(e_r,  self._k[2], self._a[2], self._b[2]) +
                 torch.clamp(self.bthz, max=5.0) * torch.clamp(e_th, min=0.0) * torch.clamp(e_z, min=0.0))
        q_fib = self.fiber_sum(C_from_lambdas(lam))
        return q_iso, q_fib
    def energy(self, lam): 
        q_iso, q_fib = self.energy_parts(lam); return q_iso + q_fib
    def forward(self, lam, create_graph=True):
        if not lam.requires_grad: lam = lam.requires_grad_(True)
        W = self.energy(lam); σθ, σz = cauchy_from_W(W, lam, create_graph)
        return W, σθ, σz

    
# ============================= Data utils =============================
def find_ages(root: str):
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])

def _first_col_optional(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_csvs_for_age(age_dir: str):
    pd_s, fl_s = [], []
    files = glob.glob(os.path.join(age_dir, "*.csv"))
    pat = re.compile(r".*(?:^|[_\-.])(pd|fl)(\d+)(?:[_\-.].*)?\.csv$", flags=re.I)

    for csv in files:
        base = os.path.basename(csv); m = pat.match(base)
        if not m: continue
        kind = m.group(1).lower(); level_num = int(m.group(2))
        df = pd.read_csv(csv)
        df.columns = [c.strip().lower() for c in df.columns]

        lam_th_col = _first_col_optional(df, ["lambda_theta","lam_theta","lam_th","ltheta","lambda_th","stretch_theta"])
        lam_z_col  = _first_col_optional(df, ["lambda_z","lam_z","lz","lambdaaxial","lambda_longitudinal","stretch_z"])

        if kind == "pd":
            sig_th_col = _first_col_optional(df, ["sigma_theta_kpa","sigma_theta","sig_theta","stheta","sigma_th","circstresskpa"])
            if sig_th_col is None or lam_th_col is None: 
                print(f"[skip] {base}: missing θ stress or θ stretch"); continue
            lamθ = df[lam_th_col].to_numpy(); σθ = df[sig_th_col].to_numpy()
            if lam_z_col is not None:
                lamz = df[lam_z_col].to_numpy()
                if lamz.size != lamθ.size: lamz = np.full_like(lamθ, float(np.median(lamz)))
            else:
                lamz = np.full_like(lamθ, level_num/100.0, dtype=float)
            pd_s.append(dict(lamθ=lamθ, lamz=lamz, sig=σθ, _file=base))
        else:
            sig_z_col = _first_col_optional(df, ["sigma_z_kpa","sigma_z","sig_z","sz","sigma_axial"])
            if sig_z_col is None or lam_z_col is None:
                print(f"[skip] {base}: missing z stress or z stretch"); continue
            lamz = df[lam_z_col].to_numpy(); σz = df[sig_z_col].to_numpy()
            if lam_th_col is not None:
                lamθ = df[lam_th_col].to_numpy()
                if lamθ.size != lamz.size: lamθ = np.full_like(lamz, float(np.median(lamθ)))
            else:
                lamθ = np.full_like(lamz, level_num/100.0, dtype=float)
            fl_s.append(dict(lamθ=lamθ, lamz=lamz, sig=σz, _file=base))

    print(f"  [parse] found {len(pd_s)} PD and {len(fl_s)} FL in {os.path.basename(age_dir)}")
    if not pd_s and not fl_s:
        print("  tip: expected files matching '*_pd95.csv', '*_pd100.csv', '*_fl100.csv', etc.")
    return {"pd": pd_s, "fl": fl_s}

def build_dataset(blob):
    if blob["pd"]:
        Xth = np.stack([np.concatenate([s["lamθ"] for s in blob["pd"]]),
                        np.concatenate([s["lamz"] for s in blob["pd"]])], 1)
        yth = np.concatenate([s["sig"] for s in blob["pd"]])
    else:
        Xth, yth = np.zeros((0,2)), np.zeros((0,))
    if blob["fl"]:
        Xz = np.stack([np.concatenate([s["lamθ"] for s in blob["fl"]]),
                       np.concatenate([s["lamz"] for s in blob["fl"]])], 1)
        yz = np.concatenate([s["sig"] for s in blob["fl"]])
    else:
        Xz, yz = np.zeros((0,2)), np.zeros((0,))
    return Xth, yth, Xz, yz

def build_dataset_with_weights(blob):
    # builds per-sample weights so that each curve/file contributes equally to the loss, no matter how many points it has
    Xth, yth, Xz, yz = build_dataset(blob)
    def _gid(filename: str) -> str:
        m = re.search(r"(pd|fl)(\d+)", filename.lower())
        return f"{m.group(1)}{m.group(2)}" if m else filename

    wth_chunks, ids_th = [], []
    for s in blob["pd"]:
        gid = _gid(s["_file"]); n = max(len(s["sig"]),1)
        wth_chunks.append(np.full(n, 1.0/n, float))
        ids_th.append(np.array([gid]*n, object))
    wth = np.concatenate(wth_chunks) if wth_chunks else np.zeros((0,), float)
    ids_th = np.concatenate(ids_th) if ids_th else np.array([], object)
    if wth.size:
        for u in np.unique(ids_th):
            mask = (ids_th==u); wth[mask] /= wth[mask].sum()

    wz_chunks, ids_z = [], []
    for s in blob["fl"]:
        gid = _gid(s["_file"]); n = max(len(s["sig"]),1)
        wz_chunks.append(np.full(n, 1.0/n, float))
        ids_z.append(np.array([gid]*n, object))
    wz = np.concatenate(wz_chunks) if wz_chunks else np.zeros((0,), float)
    ids_z = np.concatenate(ids_z) if ids_z else np.array([], object)
    if wz.size:
        for u in np.unique(ids_z):
            mask = (ids_z==u); wz[mask] /= wz[mask].sum()

    # return curve ids as well so plotting can group per-file curves
    return Xth, yth, Xz, yz, wth, wz, ids_th, ids_z

def identify_problematic_curves(blob, Xth, yth, Xz, yz, base_model):
    """Find which experimental curves are badly fit"""
    device = next(base_model.parameters()).device
    
    print("\n=== Per-Curve Error Analysis ===")
    
    # Analyze PD curves
    for i, s in enumerate(blob["pd"]):
        lam_curve = np.column_stack([s["lamθ"], s["lamz"], 1.0/(s["lamθ"]*s["lamz"])])
        lam_t = torch.tensor(lam_curve, device=device)
        
        with torch.enable_grad():
            _, sθ, _ = model_forward_branch(base_model, lam_t.requires_grad_(True), branch="theta", create_graph=True)
        
        resid = (sθ.detach().cpu().numpy() - s["sig"])
        rmse = np.sqrt((resid**2).mean())
        mean_err = resid.mean()
        
        print(f"PD {s['_file']:50s}: n={len(s['sig']):4d}, RMSE={rmse:6.3f}, mean={mean_err:+7.3f}, "
              f"λθ=[{s['lamθ'].min():.2f},{s['lamθ'].max():.2f}]")
        
        # Flag if bad
        if rmse > 3.0 or abs(mean_err) > 2.0:
            print(f"  ⚠️  HIGH ERROR CURVE!")
    
    print()
    # Similar for FL curves
    for i, s in enumerate(blob["fl"]):
        lam_curve = np.column_stack([s["lamθ"], s["lamz"], 1.0/(s["lamθ"]*s["lamz"])])
        lam_t = torch.tensor(lam_curve, device=device)
        
        with torch.enable_grad():
            _, _, sz = model_forward_branch(base_model, lam_t.requires_grad_(True), branch="z", create_graph=True)
        
        resid = (sz.detach().cpu().numpy() - s["sig"])
        rmse = np.sqrt((resid**2).mean())
        mean_err = resid.mean()
        
        print(f"FL {s['_file']:50s}: n={len(s['sig']):4d}, RMSE={rmse:6.3f}, mean={mean_err:+7.3f}, "
              f"λz=[{s['lamz'].min():.2f},{s['lamz'].max():.2f}]")
        
        if rmse > 2.0 or abs(mean_err) > 1.5:
            print(f"  ⚠️  HIGH ERROR CURVE!")
def apply_fixed_phi(base_like, phi_iso: float, phi_aniso: float, eps: float = 1e-8):
    """
    base_like: CompositeSEDFBase instance (or any module that has ._phi_logits)
    Sets the internal weight parameters so phi() returns the desired fixed weights,
    respecting the selected constraint mode, and freezes them.
    """
    phi_np = np.array([phi_iso, phi_aniso], dtype=np.float64)
    phi_np = np.clip(phi_np, eps, None)
    if getattr(base_like, "phi_constraint", "sum1") == "sum1":
        phi_np = phi_np / phi_np.sum()
        raw = np.log(phi_np)
    else:
        raw = _inv_softplus_pos(phi_np, eps=eps)
    phi = torch.tensor(phi_np, dtype=torch.float64, device=base_like._phi_logits.device)

    with torch.no_grad():
        base_like._phi_logits.copy_(torch.tensor(raw, dtype=torch.float64, device=base_like._phi_logits.device))
    base_like._phi_logits.requires_grad_(False)
    return phi.detach().cpu().numpy()

def configure_recruitment_start(fiber_like, start: float, mode: str = "learn"):
    """
    Initialize all per-family λ_lb to the same starting value.
    mode='fixed' freezes them; mode='learn' lets optimization update them.
    """
    raw = _raw_from_bounded(start, RECRUIT_LB_MIN, RECRUIT_LB_MAX)
    with torch.no_grad():
        fiber_like._lambda_lb_raw.fill_(raw)
    fiber_like._lambda_lb_raw.requires_grad_(mode == "learn")
    return float(fiber_like.lambda_lb().detach().cpu().mean().item())

def preserve_recruitment_start(fiber_like, mode: str = "learn"):
    """
    Keep the currently loaded per-family λ_lb values; update only the requires_grad flag.
    """
    fiber_like._lambda_lb_raw.requires_grad_(mode == "learn")
    return float(fiber_like.lambda_lb().detach().cpu().mean().item())

def configure_recruitment_reference_stretch(fiber_like, ref_stretch: float):
    """
    Set the in-vivo axial stretch used to shift λ_lb for different circumferential curves.
    """
    ref = max(float(ref_stretch), 1e-6)
    with torch.no_grad():
        fiber_like._recruit_ref_stretch.copy_(torch.tensor(ref, dtype=torch.float64, device=fiber_like._recruit_ref_stretch.device))
    return float(fiber_like.recruit_ref_stretch().detach().cpu().item())

def infer_in_vivo_axial_stretch(blob) -> float:
    """
    Use the PD100 curve as the in-vivo axial reference when available.
    Fall back to the median PD axial prestretch, then 1.0 if no PD data exist.
    """
    for s in blob.get("pd", []):
        if re.search(r"pd100", s.get("_file", ""), flags=re.I):
            return float(np.median(np.asarray(s["lamz"], dtype=float)))
    if blob.get("pd"):
        meds = [float(np.median(np.asarray(s["lamz"], dtype=float))) for s in blob["pd"]]
        meds = sorted(meds)
        return float(meds[len(meds)//2])
    return 1.0

def is_decomposed_model(model) -> bool:
    return False

def model_forward_branch(model, lam, *, branch: str, create_graph: bool):
    try:
        return model(lam, create_graph=create_graph, branch=branch)
    except TypeError:
        return model(lam, create_graph=create_graph)


# ============================ Train utils =============================
def _stack_lams(X):
    lamθ, lamz = X[:,0], X[:,1]; lamr = 1.0/(lamθ*lamz)
    return np.column_stack([lamθ, lamz, lamr])

def _aic_bic(loss_val: float, n: int, model: nn.Module):
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rss = float(loss_val) * max(n,1)
    aic = n * math.log(rss/max(n,1)+1e-12) + 2*k
    bic = n * math.log(rss/max(n,1)+1e-12) + math.log(max(n,1))*k
    return aic, bic

def _full_plain_mse(model: nn.Module, lam_th, yth, lam_z, yz_t) -> float:
    """Comparison metric used by plots: unweighted full-dataset stress MSE."""
    model.eval()
    loss_val = 0.0
    with torch.enable_grad():
        if lam_th is not None and yth is not None:
            lam_eval = lam_th.detach().clone().requires_grad_(True)
            _, sθ, _ = model_forward_branch(model, lam_eval, branch="theta", create_graph=False)
            loss_val += ((sθ.detach() - yth) ** 2).mean().item()
        if lam_z is not None and yz_t is not None:
            lam_eval = lam_z.detach().clone().requires_grad_(True)
            _, _, sz = model_forward_branch(model, lam_eval, branch="z", create_graph=False)
            loss_val += ((sz.detach() - yz_t) ** 2).mean().item()
    return float(loss_val)

def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or y_true.size != y_pred.size:
        return float("nan")
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot

def compute_fit_metrics(model: nn.Module, Xth, yth, Xz, yz, *,
                        label: str, loss: float = np.nan,
                        aic: float = np.nan, bic: float = np.nan) -> dict:
    device = next(model.parameters()).device
    row = {
        "model": label,
        "r2_circumferential": float("nan"),
        "r2_axial": float("nan"),
        "mse_circumferential": float("nan"),
        "mse_axial": float("nan"),
        "loss": float(loss),
        "aic": float(aic),
        "bic": float(bic),
    }
    model.eval()
    mse_parts = []
    with torch.enable_grad():
        if Xth.size and yth.size:
            lam = torch.tensor(_stack_lams(Xth), dtype=torch.float64, device=device).requires_grad_(True)
            _, sθ, _ = model_forward_branch(model, lam, branch="theta", create_graph=False)
            pred = sθ.detach().cpu().numpy()
            row["r2_circumferential"] = _r2_score_np(yth, pred)
            row["mse_circumferential"] = float(np.mean((np.asarray(yth) - pred) ** 2))
            mse_parts.append(row["mse_circumferential"])
        if Xz.size and yz.size:
            lam = torch.tensor(_stack_lams(Xz), dtype=torch.float64, device=device).requires_grad_(True)
            _, _, sz = model_forward_branch(model, lam, branch="z", create_graph=False)
            pred = sz.detach().cpu().numpy()
            row["r2_axial"] = _r2_score_np(yz, pred)
            row["mse_axial"] = float(np.mean((np.asarray(yz) - pred) ** 2))
            mse_parts.append(row["mse_axial"])
    if not np.isfinite(row["loss"]):
        row["loss"] = float(np.sum(mse_parts)) if mse_parts else float("nan")
    if (not np.isfinite(row["aic"]) or not np.isfinite(row["bic"])) and np.isfinite(row["loss"]):
        n_total = max(int(np.size(yth)) + int(np.size(yz)), 1)
        row["aic"], row["bic"] = _aic_bic(row["loss"], n_total, model)
    return row

def save_fit_metrics_table(rows: List[dict], age: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    columns = [
        "model", "r2_circumferential", "r2_axial",
        "mse_circumferential", "mse_axial", "loss", "aic", "bic"
    ]
    df = pd.DataFrame(rows, columns=columns)
    csv_path = os.path.join(outdir, f"{age}_fit_metrics.csv")
    txt_path = os.path.join(outdir, f"{age}_fit_metrics.txt")
    df.to_csv(csv_path, index=False)

    with open(txt_path, "w") as f:
        f.write(f"Fit metrics comparison (Age {age})\n")
        f.write("=" * 92 + "\n")
        f.write(f"{'Model':18s} {'R2 circ':>10s} {'R2 axial':>10s} {'MSE circ':>12s} {'MSE axial':>12s} {'Loss':>12s} {'AIC':>10s} {'BIC':>10s}\n")
        f.write("-" * 92 + "\n")
        for row in rows:
            f.write(
                f"{row['model']:18s} "
                f"{row['r2_circumferential']:10.4f} "
                f"{row['r2_axial']:10.4f} "
                f"{row['mse_circumferential']:12.4e} "
                f"{row['mse_axial']:12.4e} "
                f"{row['loss']:12.4e} "
                f"{row['aic']:10.2f} "
                f"{row['bic']:10.2f}\n"
            )
    print(f"Fit metrics saved to: {txt_path}")
    print(f"Fit metrics CSV saved to: {csv_path}")


# ====================== Linear regression for cross weights ======================

@torch.no_grad()
def _stack_branch_stresses(model, lam_th_t, lam_z_t, device):
    """Return (sigma_th, sigma_z) numpy arrays for the current model state.

    Uses the existing autograd-based forward but evaluates with no grad
    accumulation back into the parameters (we only differentiate w.r.t. lambda).
    """
    sigma_th = None
    sigma_z = None
    model.eval()
    with torch.enable_grad():
        if lam_th_t is not None and lam_th_t.numel() > 0:
            lam = lam_th_t.detach().clone().to(device).requires_grad_(True)
            _, sθ, _ = model_forward_branch(model, lam, branch="theta", create_graph=False)
            sigma_th = sθ.detach().cpu().numpy().ravel()
        if lam_z_t is not None and lam_z_t.numel() > 0:
            lam = lam_z_t.detach().clone().to(device).requires_grad_(True)
            _, _, sz = model_forward_branch(model, lam, branch="z", create_graph=False)
            sigma_z = sz.detach().cpu().numpy().ravel()
    return sigma_th, sigma_z


def linreg_vein_weights(model, Xth, yth, Xz, yz, *,
                        wtheta_boost: float, device,
                        ridge: float = 1e-3) -> dict:
    """
    Solve _w1.._w10 of a VeinSEDFDiscovered model in closed form via weighted
    *column-normalised* ridge regression, with the base held fixed.

    Stresses are linear in (_w1.._w10) because W = W_base + Σ w_k ψ_k(λ) and
    the Cauchy stress is a linear functional of W. So per data point i and per
    basis term k, σ_k(λ_i) - σ_base(λ_i) gives column k of the design matrix
    A, and the residual r_i = y_i - σ_base(λ_i) is the regression target.

    The weighting matches fit_model's adaptive weighting:
        w_th = wtheta_boost / std(yth) * (n_th + n_z) / (2 n_th)
        w_z  = 1 / std(yz)             * (n_th + n_z) / (2 n_z)

    The 10 basis terms are mutually correlated (e.g. <I4D1-1> and <I4D2-1>
    carry near-identical information for symmetric helical fibers), so plain
    LS yields huge opposing weights that produce visible ripples between data
    points. We column-normalise A to unit Euclidean length and solve
        (Ãᵀ Ã + ridge·I) w̃ = Ãᵀ r        with  Ã = A / ‖A_:,k‖
    then unscale w_k = w̃_k / ‖A_:,k‖. With column-norm = 1, `ridge` is
    dimensionless: 0 = plain LS, 1e-3 mild, 1e-1 strong, 1.0 nearly zero
    weights. Default 1e-3.

    Returns a dict with the solved weights, residual norm, per-branch R²,
    the ridge actually used, and the column norms (for diagnostics).
    """
    if not hasattr(model, "DISC_WEIGHTS"):
        raise TypeError("linreg_vein_weights expects a VeinSEDFDiscovered model")

    K = len(model.DISC_WEIGHTS)
    # Save current weights so we can restore on early return.
    saved = [float(p.detach().item()) for p in model.disc_weight_params()]

    lam_th_t = (torch.tensor(_stack_lams(Xth), dtype=torch.float64, device=device)
                if Xth.size else None)
    lam_z_t  = (torch.tensor(_stack_lams(Xz),  dtype=torch.float64, device=device)
                if Xz.size  else None)
    yth_arr = np.asarray(yth, dtype=float).ravel() if yth.size else np.array([])
    yz_arr  = np.asarray(yz,  dtype=float).ravel() if yz.size  else np.array([])
    n_th = yth_arr.size
    n_z  = yz_arr.size
    if n_th + n_z == 0:
        return dict(weights=saved, residual_norm=float("nan"),
                    r2_th=float("nan"), r2_z=float("nan"))

    # Per-sample weights matching fit_model's weighting (square-root form for LS).
    yth_t = torch.tensor(yth_arr, dtype=torch.float64) if n_th else None
    yz_t  = torch.tensor(yz_arr,  dtype=torch.float64) if n_z  else None
    w_th = float(wtheta_boost) * _inv_std_safe(yth_t) if n_th else 0.0
    w_z  = _inv_std_safe(yz_t) if n_z else 0.0
    if n_th and n_z:
        w_th *= (n_th + n_z) / (2 * n_th + 1e-12)
        w_z  *= (n_th + n_z) / (2 * n_z  + 1e-12)
    sqrt_w_th = math.sqrt(max(w_th, 1e-30))
    sqrt_w_z  = math.sqrt(max(w_z,  1e-30))

    # Baseline: all weights at 0
    model.set_disc_weights([0.0] * K)
    sth_base, sz_base = _stack_branch_stresses(model, lam_th_t, lam_z_t, device)
    sth_base = sth_base if sth_base is not None else np.zeros(0)
    sz_base  = sz_base  if sz_base  is not None else np.zeros(0)

    # Per-basis stress contribution: set w_k=1, others=0; subtract baseline.
    cols_th, cols_z = [], []
    for k in range(K):
        vals = [0.0] * K; vals[k] = 1.0
        model.set_disc_weights(vals)
        sth_k, sz_k = _stack_branch_stresses(model, lam_th_t, lam_z_t, device)
        cols_th.append((sth_k - sth_base) if n_th else np.zeros(0))
        cols_z.append((sz_k  - sz_base ) if n_z  else np.zeros(0))

    # Stack design matrix and target with sqrt-weights baked in.
    blocks_A = []
    blocks_r = []
    if n_th:
        A_th = np.stack(cols_th, axis=1) * sqrt_w_th  # (n_th, K)
        r_th = (yth_arr - sth_base) * sqrt_w_th
        blocks_A.append(A_th); blocks_r.append(r_th)
    if n_z:
        A_z = np.stack(cols_z, axis=1) * sqrt_w_z     # (n_z, K)
        r_z = (yz_arr - sz_base) * sqrt_w_z
        blocks_A.append(A_z); blocks_r.append(r_z)
    A = np.vstack(blocks_A)
    r = np.concatenate(blocks_r)

    # Column-normalise A so `ridge` is dimensionless across animals/scales.
    col_norms = np.linalg.norm(A, axis=0)
    safe_norms = np.where(col_norms > 1e-30, col_norms, 1.0)
    A_tilde = A / safe_norms  # (N, K)

    # Tikhonov-stabilised normal equations on the scaled system.
    AtA = A_tilde.T @ A_tilde + float(ridge) * np.eye(K)
    Atr = A_tilde.T @ r
    try:
        w_tilde = np.linalg.solve(AtA, Atr)
    except np.linalg.LinAlgError:
        w_tilde, *_ = np.linalg.lstsq(A_tilde, r, rcond=None)
    # Unscale: zero out terms whose basis was numerically empty.
    w_hat = np.where(col_norms > 1e-30, np.asarray(w_tilde).ravel() / safe_norms, 0.0)

    # Apply solved weights and compute fit quality.
    model.set_disc_weights(w_hat.tolist())
    sth_fit, sz_fit = _stack_branch_stresses(model, lam_th_t, lam_z_t, device)
    r2_th = _r2_score_np(yth_arr, sth_fit) if n_th else float("nan")
    r2_z  = _r2_score_np(yz_arr,  sz_fit ) if n_z  else float("nan")
    residual_norm = float(np.linalg.norm(r - A @ w_hat))

    return dict(weights=w_hat.tolist(), residual_norm=residual_norm,
                r2_th=r2_th, r2_z=r2_z, ridge=float(ridge),
                column_norms=col_norms.tolist())


def _save_vein_discovered_params(model, age, outdir, *, label="variant_h",
                                  linreg_info: Optional[dict] = None):
    """Write a single human-readable txt with all base params + the 10 weights."""
    if not hasattr(model, "DISC_WEIGHTS"):
        return
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{age}_{label}_params.txt")
    state = model.state_dict()
    with open(path, "w") as f:
        f.write(f"VeinSEDFDiscovered parameters — {label} — Age {age}\n")
        f.write("=" * 72 + "\n\n")

        f.write("Cross-term weights (solved by weighted linear regression):\n")
        f.write("-" * 72 + "\n")
        for (attr, hname) in model.DISC_WEIGHTS:
            v = float(getattr(model, attr).detach().item())
            f.write(f"  {attr:6s}  {hname:24s}  {v: .8e}\n")
        if linreg_info is not None:
            f.write("\nLinear-regression diagnostics:\n")
            f.write(f"  ridge (column-normalised)      : {linreg_info.get('ridge', float('nan')):.3e}\n")
            f.write(f"  weighted residual ||A w - r||₂ : {linreg_info['residual_norm']:.6e}\n")
            f.write(f"  R²(circumferential)            : {linreg_info['r2_th']:.6f}\n")
            f.write(f"  R²(axial)                      : {linreg_info['r2_z']:.6f}\n")
            cn = linreg_info.get("column_norms")
            if cn is not None:
                f.write("  column norms ‖A_:,k‖           :\n")
                for (attr, hname), n in zip(model.DISC_WEIGHTS, cn):
                    f.write(f"    {attr:6s} {hname:24s} {n:.4e}\n")

        f.write("\nBase parameters (state_dict, gradient-trained):\n")
        f.write("-" * 72 + "\n")
        for key, tensor in state.items():
            if key.startswith("_w"):
                continue  # already printed above
            arr = tensor.detach().cpu().numpy()
            if arr.ndim == 0 or arr.size == 1:
                f.write(f"  {key}: {float(arr):.10g}\n")
            else:
                f.write(f"  {key}: {np.array2string(arr.ravel(), precision=8, separator=', ')}\n")
    print(f"Parameters saved to: {path}")


def _split_params_for_tweaked(model):
    """
    Split parameters for models with base + residual/extra components.
    Handles models with base + tweak components.
    """
    names, params = zip(*list(model.named_parameters()))
    base_params  = []
    resid_params = []
    extra_params = []
    for n, p in zip(names, params):
        if n.startswith("base."):
            base_params.append(p)
        elif n.startswith("tweak."):
            extra_params.append(p)
        else:
            resid_params.append(p)
    return base_params, resid_params, extra_params

def _split_params_for_polyonly(model):
    names, params = zip(*list(model.named_parameters()))
    base_params  = []
    extra_params = []
    for n, p in zip(names, params):
        if n.startswith("base."):
            base_params.append(p)
        elif n.startswith("tweak."):
            extra_params.append(p)
    return base_params, extra_params

def _lam_all_from_X(Xth, Xz, device):
    """Build lam_all tensor from Xth and Xz for statistics computation."""
    lam_all = []
    if Xth.size:
        lam_all.append(torch.tensor(
            np.column_stack([Xth[:,0], Xth[:,1], 1.0/(Xth[:,0]*Xth[:,1])]),
            dtype=torch.float64, device=device
        ))
    if Xz.size:
        lam_all.append(torch.tensor(
            np.column_stack([Xz[:,0], Xz[:,1], 1.0/(Xz[:,0]*Xz[:,1])]),
            dtype=torch.float64, device=device
        ))
    return torch.cat(lam_all, 0) if lam_all else None

def sweep_tweak_bases(base_model, Xth, yth, Xz, yz, wth, wz, *,
                     elastin_type: str,
                     n_basis: int,
                     epochs_tweak: int,
                     lr_tweak: float,
                     wtheta_boost: float,
                     robust_kind: str,
                     robust_delta: float,
                     batch_size: Optional[int],
                     compile_model: bool,
                     seed: int,
                     max_pairs: Optional[int] = None):
    """
    Sweep over single basis terms and 2-term combinations.
    Fits tweak weights only (base is frozen).
    Returns best result and all results.
    """
    from itertools import combinations
    
    device = next(base_model.parameters()).device
    
    # Build once for stats
    lam_all = _lam_all_from_X(Xth, Xz, device=device)
    
    results = []  # list of dicts
    
    # Subsets: singles + pairs
    subsets = [(i,) for i in range(n_basis)]
    pair_list = list(combinations(range(n_basis), 2))
    if max_pairs is not None:
        pair_list = pair_list[:max_pairs]
    subsets += pair_list
    
    for k, idxs in enumerate(subsets, 1):
        name = f"sweep_{'-'.join(map(str, idxs))}"
        
         
        # Copy base
        with torch.no_grad():
            m.base.load_state_dict(base_model.state_dict(), strict=False)
        # Copy non-state attrs
        m.base.col.dist_type = base_model.col.dist_type
        
        # Freeze base, train tweak only
        for p in m.base.parameters():
            p.requires_grad_(False)
        for p in m.tweak.parameters():
            p.requires_grad_(True)
        
        # Stats for this model (normalizes the same raw features, so stats independent of idxs)
        with torch.no_grad():
            m.set_stats(lam_all)
        
        loss_val, aic, bic, m = fit_model(
            m, Xth, yth, Xz, yz,
            epochs=epochs_tweak,
            name=name,
            lr=lr_tweak,
            lr_resid=lr_tweak,
            wtheta_boost=wtheta_boost,
            jitter=0.0, jitter_warm_epochs=0,
            seed=seed,
            weights_th=wth, weights_z=wz,
            robust_kind=robust_kind,
            robust_delta=robust_delta,
            batch_size=(batch_size or None),
            compile_model=compile_model
        )
        
        results.append(dict(idxs=idxs, loss=loss_val, aic=aic, bic=bic))
        # Optional: print a short line
        if k % 10 == 0 or k == 1:
            best_so_far = min(results, key=lambda r: r["loss"])
            print(f"[sweep] {k}/{len(subsets)} done. Best so far idxs={best_so_far['idxs']} loss={best_so_far['loss']:.4e}")
    
    best = min(results, key=lambda r: r["loss"])
    return best, results

def _robust_weighted_loss(residuals: torch.Tensor,
                          weights: Optional[torch.Tensor] = None,
                          kind: str = "huber",
                          delta: float = 0.25) -> torch.Tensor:
    if kind == "huber":
        a = residuals.abs()
        out = torch.where(a<delta, 0.5*a*a, delta*(a-0.5*delta))
    elif kind == "charbonnier":
        eps = 1e-3; out = torch.sqrt(residuals*residuals + eps*eps) - eps
    else:
        out = residuals*residuals
    if weights is None: return out.mean()
    w = weights / (weights.mean() + 1e-12)
    return (w * out).mean()

# ============================== Training ==============================
def fit_model(model: nn.Module,
              Xth, yth, Xz, yz,
              epochs: int,
              *,
              name: str,
              lr: float,
              wtheta_boost: float = 1.0,
              jitter: float = 2e-3,
              jitter_warm_epochs: int = 300,
              seed: Optional[int] = None,
              weights_th=None, weights_z=None,
              robust_kind: str = "huber", robust_delta: float = 0.25,
              batch_size: Optional[int] = None,
              compile_model: bool = False,
              gate_lambda: float = 0.0,
              anchor_lambda: float = 0.0,
              anchor_params: Optional[dict] = None,
              ):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if compile_model and hasattr(torch, "compile"):
        try: model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception: pass

    # Prepare data
    lam_th = torch.tensor(_stack_lams(Xth), device=device) if Xth.size else None
    lam_z  = torch.tensor(_stack_lams(Xz),  device=device) if Xz.size else None
    yθ     = torch.tensor(yth, device=device) if yth.size else None
    yz_t   = torch.tensor(yz, device=device) if yz.size else None
    wth_i  = torch.tensor(weights_th, device=device) if (weights_th is not None and np.size(weights_th)) else None
    wz_i   = torch.tensor(weights_z,  device=device) if (weights_z  is not None and np.size(weights_z))  else None

    lam_th_train = lam_th
    yth_train = yθ
    wth_train = wth_i
    lam_z_train = lam_z
    yz_train = yz_t
    wz_train = wz_i

    # Compute total sample size for AIC/BIC
    n_total = max((len(yth_train) if yth_train is not None else 0) + 
                  (len(yz_train) if yz_train is not None else 0), 1)

    # Keep the best full-data MSE checkpoint so later epochs cannot drift away
    # from the metric used in the comparison plots.
    best_state = copy.deepcopy(model.state_dict())
    best_full_loss = _full_plain_mse(model, lam_th, yθ, lam_z, yz_t)
    
    # Adaptive weights for balancing θ and z losses
    w_th = wtheta_boost * _inv_std_safe(yth_train)
    w_z = _inv_std_safe(yz_train)
    n_th = len(yth_train) if yth_train is not None else 0
    n_z = len(yz_train) if yz_train is not None else 0
    if n_th and n_z:
        w_th *= (n_th+n_z)/(2*n_th+1e-12)
        w_z *= (n_th+n_z)/(2*n_z+1e-12)

   
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    max_clip = 100.0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    # Training loop
    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        
        loss = torch.tensor(0.0, device=device)
        jitter_now = 0.0 if ep <= jitter_warm_epochs else jitter

        # σθ training loss
        if lam_th_train is not None and yth_train is not None:
            if batch_size and lam_th_train.shape[0] > batch_size:
                idx = torch.randint(0, lam_th_train.shape[0], (batch_size,), device=device)
                x = lam_th_train[idx].detach().clone()
                yb = yth_train[idx]
                wb = wth_train[idx] if wth_train is not None else None
            else:
                x = lam_th_train.detach().clone()
                yb = yth_train
                wb = wth_train
            
            if jitter_now>0:
                x[..., :2] *= (1.0 + jitter_now*torch.randn_like(x[..., :2]))
                x[..., 2] = 1.0/(x[...,0]*x[...,1])
            x = x.requires_grad_(True)
            _, sθ, _ = model_forward_branch(model, x, branch="theta", create_graph=True)
            loss = loss + w_th * _robust_weighted_loss(sθ - yb, wb, robust_kind, robust_delta)

        # σz training loss
        if lam_z_train is not None and yz_train is not None:
            if batch_size and lam_z_train.shape[0] > batch_size:
                idx = torch.randint(0, lam_z_train.shape[0], (batch_size,), device=device)
                x = lam_z_train[idx].detach().clone()
                yb = yz_train[idx]
                wb = wz_train[idx] if wz_train is not None else None
            else:
                x = lam_z_train.detach().clone()
                yb = yz_train
                wb = wz_train
            
            if jitter_now>0:
                x[..., :2] *= (1.0 + jitter_now*torch.randn_like(x[..., :2]))
                x[..., 2] = 1.0/(x[...,0]*x[...,1])
            x = x.requires_grad_(True)
            _, _, sz = model_forward_branch(model, x, branch="z", create_graph=True)
            loss = loss + w_z * _robust_weighted_loss(sz - yb, wb, robust_kind, robust_delta)


        
        # Gate sparsity penalty (for GatedSEDF / GatedTweakLinear)
        if gate_lambda > 0.0 and hasattr(model, "tweak") and hasattr(model.tweak, "l1_gate_penalty"):
            loss = loss + gate_lambda * model.tweak.l1_gate_penalty()
        
        # Anchor regularization: penalize moving away from reference parameters
        if anchor_lambda > 0.0 and anchor_params is not None:
            for name, p in model.named_parameters():
                if name in anchor_params:
                    loss = loss + anchor_lambda * (p - anchor_params[name]).pow(2).mean()
        

        loss = 55.0 * loss  # simple scaling for composite/legacy models (MATLAB-like)
        
        # Backward pass
        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=max_clip)
        opt.step()
        scheduler.step()

        if ep % 50 == 0:
            full_loss_now = _full_plain_mse(model, lam_th, yθ, lam_z, yz_t)
            if full_loss_now < best_full_loss:
                best_full_loss = full_loss_now
                best_state = copy.deepcopy(model.state_dict())

        # Periodic logging
        if ep % max(epochs//10, 200) == 0:
            aic, bic = _aic_bic(float(loss.item()), n_total, model)
            print(f"    [{name}] ep{ep}: loss={float(loss.item()):.4e}, full_mse={full_loss_now:.4e}, AIC={aic:.1f}, BIC={bic:.1f}, LR={opt.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state)

    model.eval()
    loss_val = 0.0
    with torch.enable_grad():
        if lam_th is not None and yθ is not None:
            lam_eval = lam_th.detach().clone().requires_grad_(True)
            _, sθ, _ = model_forward_branch(model, lam_eval, branch="theta", create_graph=False)
            loss_val += ((sθ.detach() - yθ)**2).mean().item()
        if lam_z is not None and yz_t is not None:
            lam_eval = lam_z.detach().clone().requires_grad_(True)
            _, _, sz = model_forward_branch(model, lam_eval, branch="z", create_graph=False)
            loss_val += ((sz.detach() - yz_t)**2).mean().item()

    aic, bic = _aic_bic(loss_val, n_total, model)
    print(f"    [{name}] Final (full): loss={loss_val:.4e}, AIC={aic:.1f}, BIC={bic:.1f}")
    if best_full_loss + 1e-12 < loss_val:
        print(f"    [{name}] Restored best full-data MSE checkpoint: {best_full_loss:.4e}")
    
    return loss_val, aic, bic, model

def diagnose_residuals(model, Xth, yth, Xz, yz, save_path=None):
    """Analyze where the base model fails"""
    device = next(model.parameters()).device
    
    resid_th = None
    resid_z = None
    
    # Compute residuals by strain region
    print("\n=== Circumferential (θ) Direction ===")
    if Xth.size:
        lam_th = torch.tensor(_stack_lams(Xth), device=device)
        with torch.enable_grad():
            _, sθ, _ = model_forward_branch(model, lam_th.requires_grad_(True), branch="theta", create_graph=True)
        resid_th = (sθ.detach().cpu().numpy() - yth)
        
        print(f"Overall RMSE: {np.sqrt((resid_th**2).mean()):.3f} kPa")
        print(f"Mean error: {resid_th.mean():.3f} kPa")
        print(f"Std error: {resid_th.std():.3f} kPa")
        print(f"Max |error|: {np.abs(resid_th).max():.3f} kPa")
        
        # Bin by stretch levels
        lth_bins = np.digitize(Xth[:,0], bins=np.linspace(1.2, 2.1, 11))
        print("\nError by λθ range:")
        for b in np.unique(lth_bins):
            mask = lth_bins == b
            if mask.sum() > 0:
                rmse = np.sqrt((resid_th[mask]**2).mean())
                mean_err = resid_th[mask].mean()
                print(f"  λθ={Xth[mask,0].min():.2f}-{Xth[mask,0].max():.2f} (n={mask.sum():3d}): "
                      f"RMSE={rmse:.3f}, mean={mean_err:+.3f}")
    
    # Similar for axial
    print("\n=== Axial (z) Direction ===")
    if Xz.size:
        lam_z = torch.tensor(_stack_lams(Xz), device=device)
        with torch.enable_grad():
            _, _, sz = model_forward_branch(model, lam_z.requires_grad_(True), branch="z", create_graph=True)
        resid_z = (sz.detach().cpu().numpy() - yz)
        
        print(f"Overall RMSE: {np.sqrt((resid_z**2).mean()):.3f} kPa")
        print(f"Mean error: {resid_z.mean():.3f} kPa")
        print(f"Std error: {resid_z.std():.3f} kPa")
        print(f"Max |error|: {np.abs(resid_z).max():.3f} kPa")
        
        # Bin by stretch levels  
        lz_bins = np.digitize(Xz[:,1], bins=np.linspace(1.1, 1.7, 11))
        print("\nError by λz range:")
        for b in np.unique(lz_bins):
            mask = lz_bins == b
            if mask.sum() > 0:
                rmse = np.sqrt((resid_z[mask]**2).mean())
                mean_err = resid_z[mask].mean()
                print(f"  λz={Xz[mask,1].min():.2f}-{Xz[mask,1].max():.2f} (n={mask.sum():3d}): "
                      f"RMSE={rmse:.3f}, mean={mean_err:+.3f}")
    
    # Plot residual patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top row: residuals vs stretch
    if Xth.size:
        axes[0, 0].scatter(Xth[:,0], resid_th, alpha=0.4, s=15, c='blue', edgecolors='none')
        axes[0, 0].axhline(0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('λθ', fontsize=12)
        axes[0, 0].set_ylabel('Residual σθ [kPa]', fontsize=12)
        axes[0, 0].set_title('Circumferential Residuals', fontsize=13, weight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add running mean
        sort_idx = np.argsort(Xth[:,0])
        window = max(len(Xth)//20, 10)
        running_mean = np.convolve(resid_th[sort_idx], np.ones(window)/window, mode='valid')
        axes[0, 0].plot(Xth[sort_idx,0][window//2:-window//2+1], running_mean, 
                       'orange', linewidth=2, label='Running mean')
        axes[0, 0].legend()
    
    if Xz.size:
        axes[0, 1].scatter(Xz[:,1], resid_z, alpha=0.4, s=15, c='green', edgecolors='none')
        axes[0, 1].axhline(0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('λz', fontsize=12)
        axes[0, 1].set_ylabel('Residual σz [kPa]', fontsize=12)
        axes[0, 1].set_title('Axial Residuals', fontsize=13, weight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add running mean
        sort_idx = np.argsort(Xz[:,1])
        window = max(len(Xz)//20, 10)
        running_mean = np.convolve(resid_z[sort_idx], np.ones(window)/window, mode='valid')
        axes[0, 1].plot(Xz[sort_idx,1][window//2:-window//2+1], running_mean, 
                       'orange', linewidth=2, label='Running mean')
        axes[0, 1].legend()
    
    # Bottom row: histograms
    if Xth.size:
        axes[1, 0].hist(resid_th, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residual σθ [kPa]', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Distribution of θ Residuals', fontsize=13, weight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    if Xz.size:
        axes[1, 1].hist(resid_z, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residual σz [kPa]', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Distribution of z Residuals', fontsize=13, weight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nDiagnostic plot saved to: {save_path}")
    
    plt.close(fig)  # Close figure to free memory
    
    # Additional analysis: identify problematic regions
    print("\n=== High-Error Regions ===")
    if Xth.size:
        high_err_th = np.abs(resid_th) > 2.0 * np.std(resid_th)
        if high_err_th.sum() > 0:
            print(f"Circumferential: {high_err_th.sum()} points ({100*high_err_th.mean():.1f}%) with |error| > 2σ")
            print(f"  λθ range: [{Xth[high_err_th,0].min():.3f}, {Xth[high_err_th,0].max():.3f}]")
            print(f"  λz range: [{Xth[high_err_th,1].min():.3f}, {Xth[high_err_th,1].max():.3f}]")
    
    if Xz.size:
        high_err_z = np.abs(resid_z) > 2.0 * np.std(resid_z)
        if high_err_z.sum() > 0:
            print(f"Axial: {high_err_z.sum()} points ({100*high_err_z.mean():.1f}%) with |error| > 2σ")
            print(f"  λθ range: [{Xz[high_err_z,0].min():.3f}, {Xz[high_err_z,0].max():.3f}]")
            print(f"  λz range: [{Xz[high_err_z,1].min():.3f}, {Xz[high_err_z,1].max():.3f}]")
    
    return resid_th, resid_z

# =============================== Plotting =============================
PAPER_DPI = 600
PAPER_COLORS = {
    "a": "#0072B2",  # blue
    "b": "#E69F00",  # orange
    "c": "#009E73",  # green
    "d": "#56B4E9",  # sky blue
    "e": "#CC79A7",  # reddish purple
    "f": "#000000",  # black
    "g": "#D55E00",  # vermillion
}


def _paper_rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "figure.dpi": 150,
        "savefig.dpi": PAPER_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.default": "regular",
        "text.usetex": False,
    }


def _style_paper_axes(ax, *, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=3.0, width=0.8, pad=2)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, linestyle="-", linewidth=0.4, alpha=0.18)
        ax.set_axisbelow(True)


def _save_paper_figure(fig, png_path):
    fig.savefig(png_path, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.03)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    return png_path, pdf_path


def _group_and_plot(ax_left, ax_right, Xth, yth, Xz, yz, model, color, ls, lbl_th, lbl_z,
                    round_step=0.01, device=None, ids_th=None, ids_z=None):
    from matplotlib.ticker import MaxNLocator
    try:
        from scipy.ndimage import gaussian_filter1d as _gauss_smooth
    except ImportError:
        def _gauss_smooth(y, sigma, mode='nearest'):
            w = max(3, int(2 * sigma + 1) | 1)
            return np.convolve(y, np.ones(w) / w, mode='same')
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Xth.size:
        lam_theta, lam_z, sig_theta = Xth[:,0], Xth[:,1], yth
        ax_left.scatter(lam_theta, sig_theta, s=12, c="#333333", alpha=0.45,
                        label="Experimental", edgecolors='none', zorder=1,
                        rasterized=True)
        if model is not None:
            groups = {}
            if ids_th is not None:
                for i, gid in enumerate(np.asarray(ids_th)):
                    groups.setdefault(str(gid), []).append(i)
            else:
                for i,lz in enumerate(lam_z):
                    key = float(np.round(lz / round_step) * round_step)
                    groups.setdefault(key, []).append(i)
            for j,(_,idxs) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
                ii = np.array(idxs)[np.argsort(lam_theta[idxs])]
                lam_group = torch.tensor(_stack_lams(Xth[ii]), device=device)
                with torch.enable_grad():
                    _, sθ, _ = model_forward_branch(model, lam_group, branch="theta", create_graph=False)
                x_plot = Xth[ii,0]
                y_pred = sθ.detach().cpu().numpy()
                bin_size = 0.005
                x_binned = np.round(x_plot / bin_size) * bin_size
                x_unique, inv = np.unique(x_binned, return_inverse=True)
                y_smooth = np.bincount(inv, weights=y_pred) / np.bincount(inv)
                ax_left.plot(x_unique, y_smooth, color=color, lw=2.0, alpha=1.0, ls=ls,
                             label=(lbl_th if j==0 else None), zorder=2)
        ax_left.set_xlabel("λθ"); ax_left.set_ylabel("σθ [kPa]")
        ax_left.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_left.yaxis.set_major_locator(MaxNLocator(nbins=3))
        _style_paper_axes(ax_left)
    if Xz.size:
        lam_theta, lam_z, sig_z = Xz[:,0], Xz[:,1], yz
        ax_right.scatter(lam_z, sig_z, s=12, c="#333333", alpha=0.45,
                         label="Experimental", edgecolors='none', zorder=1,
                         rasterized=True)
        if model is not None:
            groups = {}
            if ids_z is not None:
                for i, gid in enumerate(np.asarray(ids_z)):
                    groups.setdefault(str(gid), []).append(i)
            else:
                round_step_z = max(round_step * 1.5, 0.015)
                for i,lth in enumerate(lam_theta):
                    key = float(np.round(lth / round_step_z) * round_step_z)
                    groups.setdefault(key, []).append(i)
            for j,(_,idxs) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
                ii = np.array(idxs)[np.argsort(lam_z[idxs])]
                lam_group = torch.tensor(_stack_lams(Xz[ii]), device=device)
                with torch.enable_grad():
                    _, _, sz = model_forward_branch(model, lam_group, branch="z", create_graph=False)
                x_plot = Xz[ii,1]
                y_pred = sz.detach().cpu().numpy()
                bin_size = 0.01
                x_binned = np.round(x_plot / bin_size) * bin_size
                x_unique, inv = np.unique(x_binned, return_inverse=True)
                y_smooth = np.bincount(inv, weights=y_pred) / np.bincount(inv)
                y_smooth = np.maximum(y_smooth, 0.0)
                if len(y_smooth) > 5:
                    y_smooth = _gauss_smooth(y_smooth.astype(float), 0.8)
                    y_smooth = np.maximum(y_smooth, 0.0)
                ax_right.plot(x_unique, y_smooth, color=color, lw=2.0, alpha=1.0, ls=ls,
                              label=(lbl_z if j==0 else None), zorder=2)
        ax_right.set_xlabel("λz"); ax_right.set_ylabel("σz [kPa]")
        ax_right.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_right.yaxis.set_major_locator(MaxNLocator(nbins=3))
        _style_paper_axes(ax_right)

def plot_results(Xth, yth, Xz, yz, base_model, tweaked_model, age, outdir, round_step=0.01, ids_th=None, ids_z=None):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.rcParams.update(_paper_rc())

    fig1, axs1 = plt.subplots(1,2, figsize=(7.2, 3.2), constrained_layout=True)
    _group_and_plot(axs1[0], axs1[1], Xth, yth, Xz, yz, base_model, PAPER_COLORS["g"], "--",
                    r"$\Psi_{base}$", r"$\Psi_{base}$", round_step, device, ids_th=ids_th, ids_z=ids_z)
    _group_and_plot(axs1[0], axs1[1], Xth, yth, Xz, yz, tweaked_model, PAPER_COLORS["a"], "--",
                    r"$\Psi_{base} + \Psi_{cross}$", r"$\Psi_{base} + \Psi_{cross}$", round_step, device, ids_th=ids_th, ids_z=ids_z)
    fig1.suptitle(f"Age {age}: Experimental vs Base vs Poly", weight="bold")
    _save_paper_figure(fig1, os.path.join(outdir, f"{age}_fits.png"))
    plt.close(fig1)

def create_equation_comparison_plots(
    model,
    Xth,
    yth,
    Xz,
    yz,
    base_type,
    age,
    out_root,
    *,
    base_model_override=None,
    include_sigma_in_simplified: bool = False,
    ids_th=None,
    ids_z=None,
):
    """Create comparison plots for all fitted variants."""
    
    print("Creating equation comparison plots...")
        
    # Filter out None models
    models_to_compare = []
    base_for_plot = base_model_override if base_model_override is not None else getattr(model, "base", None)
    if base_for_plot is not None:
        models_to_compare.append((base_for_plot, "Base Model", PAPER_COLORS["d"]))
    if model is not None:
        models_to_compare.append((model, "SEDF + poly", PAPER_COLORS["g"]))

        
    if not models_to_compare:
        print("No valid models found for comparison")
        return
    plt.rcParams.update(_paper_rc())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    fig.suptitle(f'Equation Comparison: {age} - {base_type} Base', fontweight='bold')
    
    # Plot 1: Circumferential stress comparison
    ax1 = axes[0, 0]
    plot_stress_comparison(ax1, models_to_compare, Xth, yth, "Circumferential", "θ", ids=ids_th)
    
    # Plot 2: Axial stress comparison  
    ax2 = axes[0, 1]
    plot_stress_comparison(ax2, models_to_compare, Xz, yz, "Axial", "z", ids=ids_z)
    
    # Plot 3: Residual analysis
    ax3 = axes[1, 0]
    plot_residual_comparison(ax3, models_to_compare, Xth, yth, Xz, yz)
    
    # Plot 4: Weight significance analysis
    ax4 = axes[1, 1]
    plot_weight_significance(ax4, model)
    
    plt.tight_layout()
    
    # Save the plot
    comparison_file = os.path.join(out_root, f"{age}_equation_comparison.png")
    _save_paper_figure(fig, comparison_file)
    plt.close()
    
    print(f"Equation comparison plots saved to: {comparison_file}")


def plot_stress_comparison(ax, models, X, y, direction, coord, round_step=0.01, ids=None):
    """
    Plot stress comparison by grouping curves:
      - θ plot: x = (λθ-1), group by λz (rounded)
      - z plot: x = (λz-1), group by λθ (rounded)
    Assumes model(lam) -> (W, σθ, σz)
    """
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    import torch

    # numpy views for grouping / scatter
    X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

    lam_th_all = X_np[:, 0]
    lam_z_all  = X_np[:, 1]

    # choose x-axis and grouping key
    if coord == "θ":
        x_all = lam_th_all - 1.0
        group_by = lam_z_all
        x_label = "Circumferential Strain"
        y_label = "Circumferential Stress (kPa)"
        title   = "Circumferential Stress-Strain Comparison"
    else:
        x_all = lam_z_all - 1.0
        group_by = lam_th_all
        x_label = "Axial Strain"
        y_label = "Axial Stress (kPa)"
        title   = "Axial Stress-Strain Comparison"

    # experimental scatter (no connecting lines)
    ax.scatter(
        x_all, y_np, alpha=0.5, s=12, color="#333333",
        label="Experimental", edgecolors="none", linewidths=0.0, zorder=1,
        rasterized=True
    )

    # build groups
    if ids is not None:
        ids = np.asarray(ids)
        uniq = [(u, np.where(ids == u)[0]) for u in np.unique(ids)]
    else:
        keys = np.round(group_by / round_step) * round_step
        uniq_keys = np.unique(keys)
        uniq = [(k, np.where(keys == k)[0]) for k in uniq_keys]

    device = next(models[0][0].parameters()).device

    for model, label, color in models:
        first = True
        for _, idxs in uniq:
            if idxs.size < 2:
                continue

            # sort within curve by the plotted x-axis
            if coord == "θ":
                order = np.argsort(lam_th_all[idxs])
            else:
                order = np.argsort(lam_z_all[idxs])
            ii = idxs[order]

            lam_theta = torch.tensor(lam_th_all[ii], dtype=torch.float64, device=device)
            lam_z     = torch.tensor(lam_z_all[ii],  dtype=torch.float64, device=device)
            lam_r     = 1.0 / (lam_theta * lam_z)
            lam = torch.stack([lam_theta, lam_z, lam_r], dim=-1).requires_grad_(True)

            branch = "theta" if coord == "θ" else "z"
            _, sθ, sz = model_forward_branch(model, lam, branch=branch, create_graph=False)
            sigma = sθ if coord == "θ" else sz
            x = (lam_theta - 1.0).detach().cpu().numpy() if coord == "θ" else (lam_z - 1.0).detach().cpu().numpy()
            sig_np = sigma.detach().cpu().numpy()

            # line + markers per curve (no cross-curve connections)
            ax.plot(x, sig_np, color=color, lw=1.6, alpha=0.95,
                    label=(label if first else None), zorder=2)
            ax.scatter(x, sig_np, color=color, s=10, alpha=0.9,
                       edgecolors="white", linewidths=0.25, zorder=3)
            first = False

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    _style_paper_axes(ax)

def plot_residual_comparison(ax, models, Xth, yth, Xz, yz):
    """
    Boxplot residuals using models' (W, σθ, σz) outputs.
    """

    def _to_lam(X):
        if isinstance(X, np.ndarray):
            lam_theta = torch.tensor(X[:,0], dtype=torch.float64)
            lam_z     = torch.tensor(X[:,1], dtype=torch.float64)
        else:
            lam_theta = X[:,0]; lam_z = X[:,1]
        lam_r = 1.0/(lam_theta*lam_z)
        return torch.stack([lam_theta, lam_z, lam_r], dim=-1).requires_grad_(True)

    lam_th = _to_lam(Xth)
    lam_z  = _to_lam(Xz)

    residuals, labels = [], []
    for model, label, _ in models:
        _, sθ, _ = model_forward_branch(model, lam_th, branch="theta", create_graph=False)
        _, _, sz = model_forward_branch(model, lam_z,  branch="z", create_graph=False)
        r_th = (sθ.detach().cpu().numpy() - (yth.detach().cpu().numpy() if isinstance(yth, torch.Tensor) else yth))
        r_z  = (sz.detach().cpu().numpy() - (yz.detach().cpu().numpy()  if isinstance(yz,  torch.Tensor) else yz))
        residuals.append(np.concatenate([r_th, r_z]))
        labels.append(label)

    ax.boxplot(residuals, labels=labels)
    ax.set_ylabel('Residual Stress (kPa)')
    ax.set_title('Residual Analysis Comparison')
    x0, x1 = ax.get_xlim()
    xs = np.linspace(x0, x1, 30)
    ax.plot(xs, np.zeros_like(xs), linestyle='None', marker='o', markersize=3, color='red', alpha=0.7)
    _style_paper_axes(ax)

# ========================== Per-animal config loader ==========================

def load_fit_config(data_root: str, age: str) -> dict | None:
    """
    Load data/{age}/fit_config.json if it exists.
    Returns the parsed dict (keys: hyperparameters, initial_params) or None.
    """
    cfg_path = os.path.join(data_root, age, "fit_config.json")
    if not os.path.isfile(cfg_path):
        return None
    import json as _json
    with open(cfg_path) as f:
        cfg = _json.load(f)
    print(f"  → Loaded fit config from: {cfg_path}")
    return cfg


def apply_fit_config_hparams(cfg: dict, args) -> None:
    """
    Override args in-place from the hyperparameters section of fit_config.
    Only overrides keys that are not already set on the command line
    (i.e. that still hold their argparse default).  Since argparse does not
    expose which args were explicitly set, we apply all values unconditionally —
    the JSON was generated from the per-animal tuned defaults.
    """
    hp = cfg.get("hyperparameters", {})
    for key, val in hp.items():
        if hasattr(args, key):
            setattr(args, key, val)


def apply_fit_config_params(model, cfg: dict, device) -> None:
    """
    Warm-start a DecomposedSEDFBase (or any model with named parameters) from
    the initial_params section of fit_config.  Only base.* parameters are
    applied; tweak.* are silently ignored.
    """
    import json as _json
    init_params = cfg.get("initial_params", {})
    state = model.state_dict()
    for raw_key, val in init_params.items():
        # raw_key is like "base.iso._bth" — strip leading "base." to get
        # the parameter name relative to the model's own namespace.
        # For DecomposedSEDFBase the keys are like "iso._bth", "aniso._k1", etc.
        if raw_key.startswith("base."):
            key = raw_key[len("base."):]
        else:
            key = raw_key
        if key not in state:
            continue
        try:
            t = state[key]
            if isinstance(val, list):
                import torch as _torch
                new_val = _torch.tensor(val, dtype=t.dtype, device=device)
            else:
                import torch as _torch
                new_val = _torch.tensor(float(val), dtype=t.dtype, device=device)
            if new_val.shape == t.shape:
                state[key] = new_val
            else:
                print(f"    [config] shape mismatch for {key}: "
                      f"config {list(new_val.shape)} vs model {list(t.shape)}, skipping")
        except Exception as e:
            print(f"    [config] could not apply {key}: {e}")
    model.load_state_dict(state, strict=False)


# ========================== Model Variant Comparison ==========================




def _fit_variant(variant_key, args, Xth, yth, Xz, yz, wth, wz, in_vivo_lamz, device,
                 fit_cfg=None):
    """
    Fit one model variant; return (model, loss, aic, bic, label).

    fit_cfg: optional dict from load_fit_config() — if provided, warm-starts the
             base model from previously learned parameters before Stage-1 fitting.
    """
    cfg = VARIANT_CONFIGS[variant_key]
    label = cfg["label"]
    set_seed(args.seed)

    # All variants (a–e, h): single-stage fit
    if variant_key == "h":
        base = VeinSEDFDiscovered(
            iso_model="poly", aniso_model="4fiber",
            use_recruitment=cfg["recruit"], use_dispersion=cfg["disp"],
        ).to(device)
    else:
        base = DecomposedSEDFBase(
            iso_model=cfg["iso"], aniso_model="4fiber",
            use_recruitment=cfg["recruit"], use_dispersion=cfg["disp"],
        ).to(device)

    if fit_cfg is not None:
        apply_fit_config_params(base, fit_cfg, device)
    aniso = getattr(base, "aniso", None)
    if aniso is not None:
        if hasattr(aniso, "_recruit_ref_stretch"):
            configure_recruitment_reference_stretch(aniso, in_vivo_lamz)
        if hasattr(aniso, "_lambda_lb_raw") and cfg["recruit"]:
            configure_recruitment_start(aniso, args.recruit_start_init, args.recruit_start_mode)

    # For variant h: hold the 10 cross-term weights at zero during gradient
    # training so the base fits like an iso+4-fiber model on its own. The
    # weights are then solved analytically by weighted linear regression.
    if variant_key == "h":
        base.set_disc_weights([0.0] * len(base.DISC_WEIGHTS))
        for p in base.disc_weight_params():
            p.requires_grad_(False)

    loss, aic, bic, model = fit_model(
        base, Xth, yth, Xz, yz,
        epochs=args.epochs_base,
        name=f"Variant-{variant_key.upper()}",
        lr=args.lr_base,
        wtheta_boost=args.wtheta_boost,
        weights_th=wth, weights_z=wz,
        robust_kind=args.robust,
        robust_delta=args.robust_delta,
        batch_size=(args.batch_size or None),
        compile_model=args.compile,
        seed=args.seed,
    )

    if variant_key == "h":
        # Re-enable grad on the weights so AIC/BIC bookkeeping counts them, then
        # solve them in closed form via weighted linear regression on the
        # residual stress.
        for p in model.disc_weight_params():
            p.requires_grad_(True)
        info = linreg_vein_weights(
            model, Xth, yth, Xz, yz,
            wtheta_boost=args.wtheta_boost, device=device,
            ridge=getattr(args, "lr_ridge", 1e-3),
        )
        # Stash diagnostics on the model so the caller can write them to txt.
        model._linreg_info = info
        print(f"  [linreg] R²_circ={info['r2_th']:.4f}  R²_axial={info['r2_z']:.4f}  "
              f"||r||={info['residual_norm']:.3e}")

    return model, loss, aic, bic, label


def _save_comparison_metrics(rows, age, outdir):
    """Save unified 6-variant comparison table (CSV + txt)."""
    os.makedirs(outdir, exist_ok=True)
    columns = ["model", "r2_circumferential", "r2_axial",
               "mse_circumferential", "mse_axial", "loss", "aic", "bic"]
    df = pd.DataFrame(rows, columns=columns)
    csv_path = os.path.join(outdir, f"{age}_comparison_metrics.csv")
    txt_path = os.path.join(outdir, f"{age}_comparison_metrics.txt")
    df.to_csv(csv_path, index=False)

    W = 50
    with open(txt_path, "w") as f:
        f.write(f"Model comparison for Age {age}\n")
        f.write("=" * (W + 80) + "\n")
        f.write(f"{'Model':<{W}} {'R2 circ':>10} {'R2 axial':>10} "
                f"{'MSE circ':>12} {'MSE axial':>12} {'Loss':>12} {'AIC':>10} {'BIC':>10}\n")
        f.write("-" * (W + 80) + "\n")
        for row in rows:
            f.write(
                f"{row['model']:<{W}} "
                f"{row['r2_circumferential']:10.4f} "
                f"{row['r2_axial']:10.4f} "
                f"{row['mse_circumferential']:12.4e} "
                f"{row['mse_axial']:12.4e} "
                f"{row['loss']:12.4e} "
                f"{row['aic']:10.2f} "
                f"{row['bic']:10.2f}\n"
            )
    print(f"Comparison metrics saved to: {txt_path}")
    print(f"Comparison metrics CSV saved to: {csv_path}")


def _save_combined_equation(model, variant_key, age, outdir):
    """Save the combined-equation fitted weights for variant f or g."""
    os.makedirs(outdir, exist_ok=True)
    eq_path = os.path.join(outdir, f"{age}_variant_{variant_key}_equation.txt")
    cfg = VARIANT_CONFIGS[variant_key]
    rec_str  = "rec"    if cfg["recruit"] else "no rec"
    disp_str = "disp"   if cfg["disp"]    else "no disp"
    active_idx = model.tweak.active_idx.cpu().numpy().tolist()
    w_vals = model.tweak.w.detach().cpu().numpy()
    names = get_basis_names(model, fallback_n=model.n_basis)
    with open(eq_path, "w") as f:
        f.write(f"Combined Equation Fitted Weights — Variant {variant_key.upper()} — Age {age}\n")
        f.write("=" * 60 + "\n\n")
        f.write("W_total = W_base + W_cross_fixed\n\n")
        f.write(f"  W_base = b_θ·Eθ² + b_z·Ez² + b_θz·<Eθ><Ez>  (quadratic iso)\n")
        f.write(f"         + Σ_m (k1_m/4k2_m)·expm1(k2_m·ξ_m²)   (4-fiber, {rec_str}, {disp_str})\n\n")
        f.write("  W_cross_fixed = Σ_i w_i · ψ_i  (fixed terms from cross-age discovery)\n\n")
        f.write("Fitted weights:\n")
        f.write("-" * 45 + "\n")
        for idx, w in zip(active_idx, w_vals):
            name = names[idx] if idx < len(names) else f"term_{idx}"
            f.write(f"  [idx {idx:2d}] {name:25s}: w = {w:10.6f}\n")
        f.write(f"\nFixed term set (COMBINED_FIXED_TERMS):\n  {COMBINED_FIXED_TERMS}\n")
    print(f"Variant-{variant_key} equation saved to: {eq_path}")


def _plot_comparison(age, variant_models, Xth, yth, Xz, yz, outdir,
                     ids_th=None, ids_z=None, round_step=0.01):
    """Save a 6×2 comparison figure (one row per variant, circ + axial)."""
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update(_paper_rc())
    n_variants = len(variant_models)
    fig, axes = plt.subplots(n_variants, 2, figsize=(7.2, 2.0 * n_variants),
                             constrained_layout=True)
    if n_variants == 1:
        axes = axes[np.newaxis, :]
    device = next(next(iter(variant_models.values()))[0].parameters()).device
    for row_i, (vkey, (model, label)) in enumerate(variant_models.items()):
        ax_l, ax_r = axes[row_i, 0], axes[row_i, 1]
        color = PAPER_COLORS.get(vkey, "#0072B2")
        _group_and_plot(ax_l, ax_r, Xth, yth, Xz, yz, model,
                        color=color, ls="-",
                        lbl_th=label, lbl_z=label,
                        round_step=round_step,
                        device=device, ids_th=ids_th, ids_z=ids_z)
        ax_l.set_title(f"{label}\nCircumferential", weight="bold")
        ax_r.set_title(f"{label}\nAxial", weight="bold")
        ax_l.legend(frameon=False, fontsize=7)
        ax_r.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Model Comparison — Age {age}", weight="bold")
    plot_path = os.path.join(outdir, f"{age}_comparison_fits.png")
    _, pdf_path = _save_paper_figure(fig, plot_path)
    plt.close(fig)
    print(f"Comparison plot saved to: {plot_path} and {pdf_path}")


def _plot_each_variant(age, variant_models, Xth, yth, Xz, yz, outdir,
                       ids_th=None, ids_z=None, round_step=0.01):
    """Save one circumferential/axial fit figure per variant under plots/."""
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.rcParams.update(_paper_rc())
    device = next(next(iter(variant_models.values()))[0].parameters()).device

    for vkey, (model, label) in variant_models.items():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
        _group_and_plot(
            axes[0], axes[1], Xth, yth, Xz, yz, model,
            color=PAPER_COLORS.get(vkey, "#0072B2"),
            ls="-",
            lbl_th=label,
            lbl_z=label,
            round_step=round_step,
            device=device,
            ids_th=ids_th,
            ids_z=ids_z,
        )
        axes[0].set_title(f"Variant {vkey.upper()} - Circumferential", weight="bold")
        axes[1].set_title(f"Variant {vkey.upper()} - Axial", weight="bold")
        axes[0].legend(frameon=False)
        axes[1].legend(frameon=False)
        fig.suptitle(f"Age {age}: {label}", weight="bold")
        plot_path = os.path.join(plots_dir, f"{age}_variant_{vkey}_fit.png")
        _, pdf_path = _save_paper_figure(fig, plot_path)
        plt.close(fig)
        print(f"Variant {vkey.upper()} plot saved to: {plot_path} and {pdf_path}")


def _save_metric_bar_plots(rows, age, outdir):
    """Save one bar chart per numeric metric under bar plots/."""
    bar_dir = os.path.join(outdir, "bar plots")
    os.makedirs(bar_dir, exist_ok=True)
    plt.rcParams.update(_paper_rc())
    df = pd.DataFrame(rows)
    if df.empty or "model" not in df.columns:
        print("No metric rows available for bar plots")
        return

    metric_names = [
        "r2_circumferential",
        "r2_axial",
        "mse_circumferential",
        "mse_axial",
        "loss",
        "aic",
        "bic",
    ]
    model_labels = df["model"].astype(str).str.extract(r"^([a-g])", expand=False)
    model_labels = model_labels.fillna(df["model"].astype(str))
    colors = [PAPER_COLORS.get(str(v), "#0072B2") for v in model_labels]

    for metric in metric_names:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.isna().all():
            continue

        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
        ax.bar(model_labels, values, color=colors[:len(values)], alpha=0.9,
               edgecolor="black", linewidth=0.35)
        ax.set_xlabel("Model variant")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Age {age}: {metric.replace('_', ' ')}", fontweight="bold")
        _style_paper_axes(ax, grid_axis="y")
        if metric.startswith("r2"):
            finite = values[np.isfinite(values)]
            if len(finite):
                ymin = min(0.0, float(finite.min()) - 0.05)
                ymax = min(1.05, max(1.0, float(finite.max()) + 0.05))
                ax.set_ylim(ymin, ymax)

        safe_metric = re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_")
        plot_path = os.path.join(bar_dir, f"{age}_{safe_metric}_bar.png")
        _, pdf_path = _save_paper_figure(fig, plot_path)
        plt.close(fig)
        print(f"Metric bar plot saved to: {plot_path} and {pdf_path}")


def run_comparison(args, age, Xth, yth, Xz, yz, wth, wz, ids_th, ids_z,
                   in_vivo_lamz, device):
    """Fit all 6 variants (a–e, h) and write unified comparison outputs."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON MODE — Age {age}")
    print(f"{'='*60}")

    # Load per-animal config (hyperparameters + warm-start initial conditions)
    fit_cfg = load_fit_config(args.data_root, age)
    if fit_cfg is not None:
        apply_fit_config_hparams(fit_cfg, args)
        print(f"  → Applied per-animal hyperparameters for {age}")

    metric_rows = []
    variant_models = {}   # ordered dict preserving insertion order

    for vkey in list("abcdeh"):
        cfg = VARIANT_CONFIGS[vkey]
        print(f"\n--- Variant {vkey.upper()}: {cfg['label']} ---")
        model, loss, aic, bic, label = _fit_variant(
            vkey, args, Xth, yth, Xz, yz, wth, wz, in_vivo_lamz, device,
            fit_cfg=fit_cfg,
        )
        row = compute_fit_metrics(model, Xth, yth, Xz, yz,
                                   label=label, loss=loss, aic=aic, bic=bic)
        metric_rows.append(row)
        variant_models[vkey] = (model, label)
        print(f"    R²_circ={row['r2_circumferential']:.4f}  "
              f"R²_axial={row['r2_axial']:.4f}  "
              f"Loss={row['loss']:.4e}  AIC={row['aic']:.1f}")

    _save_comparison_metrics(metric_rows, age, args.out_root)
    _plot_comparison(age, variant_models, Xth, yth, Xz, yz, args.out_root,
                     ids_th=ids_th, ids_z=ids_z, round_step=args.round_step)
    _plot_each_variant(age, variant_models, Xth, yth, Xz, yz, args.out_root,
                       ids_th=ids_th, ids_z=ids_z, round_step=args.round_step)
    _save_metric_bar_plots(metric_rows, age, args.out_root)

    # Variant h carries solved cross-term weights; persist the full parameter set.
    if "h" in variant_models:
        h_model = variant_models["h"][0]
        _save_vein_discovered_params(
            h_model, age, args.out_root,
            label="variant_h",
            linreg_info=getattr(h_model, "_linreg_info", None),
        )

    print(f"\n[comparison done] outputs in: {args.out_root}")


# ================================= Main ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_root",  default="runs_sef")

    # Epochs
    ap.add_argument("--epochs_base",   type=int, default=4000)

    # LRs
    ap.add_argument("--lr_base",        type=float, default=0.5e-1)
    
    # Model choices
    ap.add_argument("--iso", choices=["poly","gent","polezero","neohookean"], default="poly",
                help="Isotropic model")
    ap.add_argument("--aniso_model", choices=["2fiber","4fiber"], default="4fiber",
                help="Anisotropic model")
    ap.add_argument("--recruit", choices=["on","off"], default="on", help="Enable anisotropic fiber recruitment")
    ap.add_argument("--dispersion", choices=["on","off"], default="on",
                help="Enable dispersion in the 4-fiber anisotropic model")
    ap.add_argument("--recruit_start_mode", choices=["learn","fixed"], default="learn",
                help="Treat the shared recruitment start λ_lb as a learned parameter or hold it fixed")
    ap.add_argument("--recruit_start_init", type=float, default=1.6,
                help="Initial guess or fixed value for the shared recruitment start λ_lb")
    ap.add_argument("--recruit_start_min", type=float, default=1.3,
                help="Lower bound for the learned recruitment start λ_lb")
    ap.add_argument("--recruit_start_max", type=float, default=2.5,
                help="Upper bound for the learned recruitment start λ_lb")
    ap.add_argument("--recruit_end_min", type=float, default=2.2,
                help="Lower bound for the recruitment upper stretch λ_ub")
    ap.add_argument("--recruit_end_max", type=float, default=2.40,
                help="Upper bound for the recruitment upper stretch λ_ub")
    ap.add_argument("--dist", choices=["beta","lognormal","halfnormal"], default="lognormal") # fiber distribution
    ap.add_argument("--tweak_gate", choices=["on","off"], default="on", help="Enable gate on tweak term")
    ap.add_argument("--gate_lambda", type=float, default=0.001, help="L1 penalty on gate values for sparsity")
    ap.add_argument("--lr_ridge", type=float, default=1e-4,
                help="Ridge (Tikhonov) regularisation for the variant-h linear "
                     "regression of cross-term weights. Applied AFTER column-"
                     "normalisation so the value is dimensionless: 0 = plain LS, "
                     "1e-3 mild (default), 1e-1 strong, 1.0 nearly zero weights. "
                     "Increase if predictions oscillate between data points.")

    ap.add_argument("--robust", choices=["huber","charbonnier","mse"], default="huber")
    ap.add_argument("--robust_delta", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wtheta_boost", type=float, default=5.5)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--jitter_warm_epochs", type=int, default=800)
    ap.add_argument("--seed", type=int, default=125)
    ap.add_argument("--round_step", type=float, default=0.01)

    # Variant / comparison mode
    ap.add_argument(
        "--variant",
        choices=list("abcdeh") + ["comparison"],
        default="h",
        help=(
            "comparison (default): fit all 6 variants (a-e, h) and save a unified table. "
            "a-e: base-only variants (no cross terms). "
            "h: VeinDiscovered (Quad+4Fib+rec+disp+9 cross terms, single-stage fit)."
        ),
    )

    args = ap.parse_args()
    set_seed(args.seed)
    set_recruitment_lb_bounds(args.recruit_start_min, args.recruit_start_max)
    set_recruitment_ub_bounds(args.recruit_end_min, args.recruit_end_max)

    ages = find_ages(args.data_root)
    print("Found ages:", ages)

    for age in ages:
        print(f"\n=== Age {age} ===")
        blob = parse_csvs_for_age(os.path.join(args.data_root, age))
        Xth, yth, Xz, yz, wth, wz, ids_th, ids_z = build_dataset_with_weights(blob)
        if Xth.shape[0]==0 and Xz.shape[0]==0:
            print("  (no data found)"); continue
        used = [d["_file"] for d in (blob["pd"]+blob["fl"])]
        print("  using:", ", ".join(sorted(used)))
        in_vivo_lamz = infer_in_vivo_axial_stretch(blob)
        print(f"  inferred in-vivo axial stretch from PD curves: λz,ref={in_vivo_lamz:.3f}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Dispatch based on --variant flag ---
        if args.variant == "comparison":
            run_comparison(args, age, Xth, yth, Xz, yz, wth, wz, ids_th, ids_z,
                           in_vivo_lamz, device)
            continue

        if args.variant in list("abcdeh"):
            fit_cfg = load_fit_config(args.data_root, age)
            if fit_cfg is not None:
                apply_fit_config_hparams(fit_cfg, args)
                print(f"  → Applied per-animal hyperparameters for {age}")
            model, loss, aic, bic, label = _fit_variant(
                args.variant, args, Xth, yth, Xz, yz, wth, wz, in_vivo_lamz, device,
                fit_cfg=fit_cfg,
            )
            row = compute_fit_metrics(model, Xth, yth, Xz, yz,
                                       label=label, loss=loss, aic=aic, bic=bic)
            save_fit_metrics_table([row], age, args.out_root)
            plot_results(Xth, yth, Xz, yz, model, model, age, args.out_root,
                         round_step=args.round_step, ids_th=ids_th, ids_z=ids_z)
            if args.variant == "h":
                _save_vein_discovered_params(
                    model, age, args.out_root,
                    label="variant_h",
                    linreg_info=getattr(model, "_linreg_info", None),
                )
            print(f"  R²_circ={row['r2_circumferential']:.4f}  "
                  f"R²_axial={row['r2_axial']:.4f}  Loss={row['loss']:.4e}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
