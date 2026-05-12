#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Juvenile SEDF model-variant comparison pipeline.

Fits 7 fixed model variants (a-g) to biaxial mechanical test data and saves
a unified comparison table (CSV + txt) and comparison plot.

Variants:
  a: NeoHookean + 4-fiber (no recruitment, no dispersion)
  b: Quadratic iso + 4-fiber (no recruitment, no dispersion)
  c: Quadratic iso + 4-fiber (recruitment, no dispersion)
  d: Quadratic iso + 4-fiber (no recruitment, dispersion)
  e: Quadratic iso + 4-fiber (recruitment, dispersion)
  f: Combined Equation (Quad + 4-fiber + recruitment + dispersion + 8 fixed cross terms)
  g: Combined Equation (Quad + 4-fiber + no recruitment + no dispersion + 8 fixed cross terms)

Usage
-----
python Juvenile_SEDF_final.py --data_root data --out_root runs_sef --variant comparison
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
EXP_ARG_MAX = 400.0
ZERO_TRANSITION_EPS = 1e-2
RECRUIT_LB_MIN = 0.30
RECRUIT_LB_MAX = 2.25
RECRUIT_UB_MIN = 1.15
RECRUIT_UB_MAX = 2.40

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
            k2  = torch.clamp(softplus_pos(self._k2[m]), max=60.0)
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
    Stage-1 base model: additive isotropic + anisotropic decomposition with no mixture weights.
    """
    def __init__(self, iso_model="poly", aniso_model="4fiber", *, use_recruitment=True,
                 use_dispersion=True):
        super().__init__()
        self.iso_model = iso_model
        self.aniso_model = aniso_model
        self.iso = build_iso_module(iso_model)
        self.aniso = build_aniso_module(aniso_model, use_recruitment=use_recruitment, use_dispersion=use_dispersion)

    def energy_parts(self, lam: torch.Tensor, branch: str = "theta"):
        W_iso = self.iso.energy(lam)
        W_aniso = self.aniso.energy(lam, branch=branch)
        return W_iso + W_aniso, W_iso, W_aniso

    def forward(self, lam: torch.Tensor, create_graph: bool = True, branch: str = "theta"):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        W, _, _ = self.energy_parts(lam, branch=branch)
        σθ, σz = cauchy_from_W(W, lam, create_graph=create_graph)
        return W, σθ, σz

class TweakLinear(nn.Module):
    """
    Masked linear tweak head that only uses a subset of basis features.
    Useful for feature selection sweeps.
    """
    def __init__(self, n_basis, active_idx):
        super().__init__()
        self.n_basis = n_basis
        self.active_idx = torch.tensor(active_idx, dtype=torch.long)
        self._w_raw = nn.Parameter(torch.zeros(len(active_idx), dtype=torch.float64))

    @property
    def w(self):
        return softplus_pos(self._w_raw)

    def forward(self, basis_feats):
        """
        basis_feats: (..., n_basis) tensor of normalized basis features
        Returns: (...,) tensor of tweak energy contribution
        """
        z = basis_feats.index_select(-1, self.active_idx)
        return (z * self.w).sum(dim=-1)

class SymbolicInvariantLayer(nn.Module):
    """Layer 1: map F -> incompressible invariants and Green strains."""
    def __init__(self, aniso_module):
        super().__init__()
        self.aniso_module = aniso_module

    def forward(self, lam):
        C = C_from_lambdas(lam)
        E_th = 0.5 * (lam[...,0]**2 - 1.0)
        E_z  = 0.5 * (lam[...,1]**2 - 1.0)
        I1v  = I1(C) - 3.0
        I2v  = I2(C) - 3.0

        dir_map = self.aniso_module.symbolic_dirs()
        I4th = I4(C, dir_map["theta"]) - 1.0
        I4zv = I4(C, dir_map["z"]) - 1.0
        zero = torch.zeros_like(I1v)
        I4D1 = I4(C, dir_map["d1"]) - 1.0 if dir_map["d1"] is not None else zero
        I4D2 = I4(C, dir_map["d2"]) - 1.0 if dir_map["d2"] is not None else zero
        I5D1 = I5(C, dir_map["d1"]) - 1.0 if dir_map["d1"] is not None else zero
        I5D2 = I5(C, dir_map["d2"]) - 1.0 if dir_map["d2"] is not None else zero
        I8D1D2 = (
            I8(C, dir_map["d1"], dir_map["d2"])
            if (dir_map["d1"] is not None and dir_map["d2"] is not None)
            else zero
        )

        return {
            "I₁-3": I1v,
            "I₂-3": I2v,
            "I₄θ-1": I4th,
            "I₄z-1": I4zv,
            "I₄D1-1": I4D1,
            "I₄D2-1": I4D2,
            "I₅D1-1": I5D1,
            "I₅D2-1": I5D2,
            "I₈D1D2": I8D1D2,
            "E_z": E_z,
            "E_θ": E_th,
        }

class SymbolicPreActivationLayer(nn.Module):
    """
    Paper-inspired zeroth layer:
      identity, Macauley bracket, and absolute value.
    """
    def __init__(self, eps: float = ZERO_TRANSITION_EPS):
        super().__init__()
        self.eps = eps

    def forward(self, inv):
        feats = []
        base_keys = [
            "I₁-3", "I₂-3", "I₄θ-1", "I₄z-1", "I₄D1-1", "I₄D2-1",
            "I₅D1-1", "I₅D2-1", "I₈D1D2",
            "E_z", "E_θ",
        ]
        for key in base_keys:
            val = inv[key]
            feats.append((key, val))
            feats.append((f"<{key}>", smooth_relu_zero(val, eps=self.eps)))
            feats.append((f"|{key}|", smooth_abs_zero(val, eps=self.eps)))
        return feats

class SymbolicOperatorLayer(nn.Module):
    """Paper-inspired first hidden layer: powers and selected cross terms."""
    def forward(self, h0_feats):
        feat_map = {name: value for name, value in h0_feats}
        feats = []
        linear_keys = [
            "I₁-3", "I₂-3", "I₄θ-1", "I₄z-1", "I₄D1-1", "I₄D2-1",
            "I₅D1-1", "I₅D2-1", "I₈D1D2", "E_z", "E_θ",
            "<I₁-3>", "<I₂-3>", "<I₄θ-1>", "<I₄z-1>", "<I₄D1-1>", "<I₄D2-1>",
            "<I₅D1-1>", "<I₅D2-1>", "<I₈D1D2>", "<E_z>", "<E_θ>",
        ]
        for key in linear_keys:
            if key in feat_map:
                feats.append((key, feat_map[key]))

        square_keys = [
            "<I₁-3>", "<I₂-3>",
            "<I₄θ-1>", "<I₄z-1>", "<I₄D1-1>", "<I₄D2-1>",
            "<I₅D1-1>", "<I₅D2-1>",
            "<E_z>", "<E_θ>",
        ]
        for key in square_keys:
            if key in feat_map:
                feats.append((f"({key})²", feat_map[key] ** 2))

        cube_keys = ["<E_θ>", "<E_z>", "<I₄θ-1>", "<I₄z-1>", "<I₄D1-1>", "<I₄D2-1>"]
        for key in cube_keys:
            if key in feat_map:
                feats.append((f"({key})³", feat_map[key] ** 3))

        cross_pairs = [
            ("<I₄D1-1>", "<E_θ>"),
            ("<I₄D2-1>", "<E_θ>"),
            ("<I₄θ-1>", "<E_θ>"),
            ("<I₄z-1>", "<E_z>"),
            ("<E_θ>", "<E_z>"),
            ("<I₁-3>", "<E_θ>"),
            ("<I₁-3>", "<E_z>"),
            ("<I₂-3>", "<E_θ>"),
            ("<I₂-3>", "<E_z>"),
        ]
        for a, b in cross_pairs:
            if a in feat_map and b in feat_map:
                feats.append((f"{a}·{b}", feat_map[a] * feat_map[b]))

        fiber_cross_pairs = [
            ("<I₄θ-1>", "<I₄z-1>"),
            ("<I₄D1-1>", "<I₄D2-1>"),
            ("<I₄θ-1>", "<I₄D1-1>"),
            ("<I₄z-1>", "<I₄D2-1>"),
            ("<I₅D1-1>", "<E_θ>"),
            ("<I₅D2-1>", "<E_θ>"),
        ]
        for a, b in fiber_cross_pairs:
            if a in feat_map and b in feat_map:
                feats.append((f"{a}·{b}", feat_map[a] * feat_map[b]))
        return feats

class SymbolicActivationLayer(nn.Module):
    """Second hidden layer: identity, exponential, and x log(1+x) maps."""
    def __init__(self, exp_keys=None, xlog_keys=None):
        super().__init__()
        self.exp_keys = set(exp_keys or [
            "E_θ", "E_z", "<E_θ>", "<E_z>",
            "<I₄θ-1>", "<I₄z-1>", "<I₄D1-1>", "<I₄D2-1>",
        ])
        self.xlog_keys = set(xlog_keys or [
            "<I₁-3>", "<I₂-3>",
            "<I₄θ-1>", "<I₄z-1>", "<I₄D1-1>", "<I₄D2-1>",
            "<I₅D1-1>", "<I₅D2-1>",
            "<E_θ>", "<E_z>",
        ])

    def forward(self, operator_feats):
        feats = []
        for name, value in operator_feats:
            feats.append((name, value))
            if name in self.exp_keys:
                feats.append((f"exp({name})-1", expm1_clamped(2.0 * value)))
            if name in self.xlog_keys:
                feats.append((f"{name}ln(1+{name})", smooth_xlog1p_pos(value)))
        return feats

class SymbolicCrossMixin:
    """
    Explicit slide-style symbolic library:
      layer 1 invariants -> layer 2 operators -> layer 3 activations -> layer 4 regression
    """
    def init_symbolic_layers(self):
        self.invariant_layer = SymbolicInvariantLayer(self.base.aniso)
        self.pre_activation_layer = SymbolicPreActivationLayer()
        self.operator_layer = SymbolicOperatorLayer()
        self.activation_layer = SymbolicActivationLayer()

    def symbolic_library_raw(self, lam):
        inv = self.invariant_layer(lam)
        h0_feats = self.pre_activation_layer(inv)
        op_feats = self.operator_layer(h0_feats)
        feat_map = {name: value for name, value in self.activation_layer(op_feats)}
        bucket_terms = []

        iso_terms = [
            ("I₁-3", "iso"),
            ("I₂-3", "iso"),
            ("<E_θ>", "iso"),
            ("<E_z>", "iso"),
            ("<I₂-3>ln(1+<I₂-3>)", "iso"),
        ]
        aniso_terms = [
            ("<I₄θ-1>", "aniso"),
            ("<I₄z-1>", "aniso"),
            ("<I₄D1-1>", "aniso"),
            ("<I₄D2-1>", "aniso"),
            ("(<I₄θ-1>)²", "aniso"),
            ("(<I₄z-1>)²", "aniso"),
            ("(<I₄D1-1>)²", "aniso"),
            ("(<I₄D2-1>)²", "aniso"),
            ("(<I₄θ-1>)³", "aniso"),
            ("(<I₄z-1>)³", "aniso"),
            ("(<I₄D1-1>)³", "aniso"),
            ("(<I₄D2-1>)³", "aniso"),
            ("<I₄D1-1>ln(1+<I₄D1-1>)", "aniso"),
            ("<I₄D2-1>ln(1+<I₄D2-1>)", "aniso"),
            ("<I₄θ-1>ln(1+<I₄θ-1>)", "aniso"),
            ("<I₄z-1>ln(1+<I₄z-1>)", "aniso"),
            ("exp(<I₄θ-1>)-1", "aniso"),
            ("exp(<I₄z-1>)-1", "aniso"),
            ("exp(<I₄D1-1>)-1", "aniso"),
            ("exp(<I₄D2-1>)-1", "aniso"),
            ("<I₅D1-1>", "aniso"),
            ("<I₅D2-1>", "aniso"),
            ("<I₅D1-1>ln(1+<I₅D1-1>)", "aniso"),
            ("<I₅D2-1>ln(1+<I₅D2-1>)", "aniso"),
        ]
        cross_terms = [
            ("<I₄θ-1>·<E_θ>", "cross"),
            ("<I₄z-1>·<E_z>", "cross"),
            ("<E_θ>·<E_z>", "cross"),
            ("<I₁-3>·<E_θ>", "cross"),
            ("<I₁-3>·<E_z>", "cross"),
            ("<I₂-3>·<E_θ>", "cross"),
            ("<I₂-3>·<E_z>", "cross"),
            ("<I₄D1-1>·<E_θ>", "cross"),
            ("<I₄D2-1>·<E_θ>", "cross"),
            ("<I₄θ-1>·<I₄z-1>", "cross"),
            ("<I₄D1-1>·<I₄D2-1>", "cross"),
            ("<I₄θ-1>·<I₄D1-1>", "cross"),
            ("<I₄z-1>·<I₄D2-1>", "cross"),
            ("I₈D1D2", "cross"),
            ("<I₅D1-1>·<E_θ>", "cross"),
            ("<I₅D2-1>·<E_θ>", "cross"),
        ]
        for name, bucket in (iso_terms + aniso_terms + cross_terms):
            if name in feat_map:
                bucket_terms.append((name, feat_map[name], bucket))
        return bucket_terms

    def symbolic_layer_outputs(self, lam):
        inv = self.invariant_layer(lam)
        h0_feats = self.pre_activation_layer(inv)
        op_feats = self.operator_layer(h0_feats)
        act_feats = self.activation_layer(op_feats)
        return inv, h0_feats, op_feats, act_feats

    def basis_names(self):
        return [name for name, _, _ in self.symbolic_library_raw(self._basis_probe())[:self.n_basis]]

    def basis_buckets(self):
        return [bucket for _, _, bucket in self.symbolic_library_raw(self._basis_probe())[:self.n_basis]]

    def _basis_probe(self):
        device = next(self.parameters()).device
        return torch.ones((1, 3), dtype=torch.float64, device=device)

    def basis_feats_raw(self, lam):
        feats = [tensor for _, tensor, _ in self.symbolic_library_raw(lam)]
        if self.n_basis > len(feats):
            raise ValueError(f"Requested n_basis={self.n_basis}, but only {len(feats)} symbolic terms are defined")
        return torch.stack(feats[:self.n_basis], dim=-1)

def get_basis_names(model, fallback_n: int = 19):
    if hasattr(model, "basis_names"):
        return model.basis_names()
    n_basis = getattr(model, "n_basis", fallback_n)
    return [f"term_{i}" for i in range(n_basis)]

# ==================== Combined Cross-Age Equation ====================

COMBINED_FIXED_TERMS = [
    "I₁-3",                    # iso: 3/12 specimens (Pablo, Rosie, Dodge)
    "I₂-3",                    # iso: 4/12 specimens (Mojito, Pablo, Rosie, Dodge)
    "<I₂-3>ln(1+<I₂-3>)",     # iso: 3/12 specimens (Mojito, Rosie, Dodge)
    "<I₄z-1>",                  # aniso: 5/12 specimens
    "<I₄D1-1>",                 # aniso: 4/12 specimens
    "<I₄D2-1>",                 # aniso: 4/12 specimens
    "<I₂-3>·<E_z>",             # cross: 3/12 specimens (Mercedes, Mojito, Sagittarius)
    "I₈D1D2",                   # cross: 12/12 specimens (universal)
]

# V2: 10-term v1 plus 3 age-rare-but-mechanically-strong terms identified from
# the per-animal runsfinal/*_final_equation.txt outputs. Used by the unified
# cross-age workflow where a single fixed equation form is refit per animal
# and the weights are then summarized as functions of age.
COMBINED_FIXED_TERMS_V2 = [
    "I₁-3", "I₂-3", "<I₂-3>ln(1+<I₂-3>)",
    "(<I₄θ-1>)³",               # aniso: 2/12 (Rosie 3w, Sagittarius 16w; coeff_raw≈39 in Sagittarius)
    "<I₂-3>·<E_z>", "I₈D1D2",
]

TERM_SETS = {
    "v1": COMBINED_FIXED_TERMS,
    "v2": COMBINED_FIXED_TERMS_V2,
}

class CombinedEquationSEDF(SymbolicCrossMixin, nn.Module):
    """
    Combined equation from symbolic regression across all juvenile ages.

    Base : quadratic isotropic (poly) + 4-fiber
    Cross: fixed 10 symbolic terms consistently selected across specimens (≥3/12)
             iso   : I1-3, I2-3, <E_theta>, <E_z>, <I2-3>ln(1+<I2-3>)
             aniso : <I4z-1>, <I4D1-1>, <I4D2-1>
             cross : <I2-3>·<E_z>, I8D1D2

    Pre-determined active_idx corresponds to the 10 fixed cross-age terms [0,1,2,3,4,6,7,8,35,42].
    use_recruitment / use_dispersion control the base fiber model (variant f = both on; variant g = both off).
    """
    def __init__(self, n_basis: int = 45,
                 use_recruitment: bool = True,
                 use_dispersion: bool = True,
                 term_list: Optional[List[str]] = None):
        super().__init__()
        self.base = DecomposedSEDFBase(
            iso_model="poly", aniso_model="4fiber",
            use_recruitment=use_recruitment, use_dispersion=use_dispersion,
        )
        self.n_basis = n_basis
        self.init_symbolic_layers()

        terms = list(term_list) if term_list is not None else COMBINED_FIXED_TERMS
        probe = torch.ones((1, 3), dtype=torch.float64)
        lib_names = [name for name, _, _ in self.symbolic_library_raw(probe)]
        active_idx = [lib_names.index(t) for t in terms if t in lib_names]
        missing = [t for t in terms if t not in lib_names]
        if missing:
            print(f"  [CombinedEquationSEDF] WARN: terms not found in library: {missing}")
        if not active_idx:
            raise RuntimeError("None of the requested fixed terms found in symbolic library.")
        self.fixed_term_names = [lib_names[i] for i in active_idx]
        self.tweak = TweakLinear(n_basis, active_idx)
        # Start all cross-term weights near-zero (softplus_pos(-4) ≈ 0.018) so the
        # optimizer selectively grows only the terms that actually help each specimen,
        # mirroring GatedSEDF's low initial_gate=0.2 behavior.
        with torch.no_grad():
            self.tweak._w_raw.fill_(-4.0)

        self.register_buffer('b_mu',    torch.zeros(n_basis, dtype=torch.float64))
        self.register_buffer('b_sigma', torch.ones(n_basis,  dtype=torch.float64))

    def basis_feats(self, lam: torch.Tensor) -> torch.Tensor:
        feats = self.basis_feats_raw(lam)
        return (feats - self.b_mu) / self.b_sigma

    @torch.no_grad()
    def set_stats(self, lam_all: Optional[torch.Tensor]):
        if lam_all is None or lam_all.numel() == 0:
            self.b_mu[:] = 0.0
            self.b_sigma[:] = 1.0
            return
        feats = self.basis_feats_raw(lam_all)
        self.b_mu.copy_(feats.mean(0))
        self.b_sigma.copy_(feats.std(0).clamp_min(1e-6))

    def forward(self, lam: torch.Tensor, create_graph: bool = True, branch: str = "theta"):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        W_mix, *_ = self.base.energy_parts(lam, branch=branch)
        z = self.basis_feats(lam)
        W_cross = self.tweak(z)
        W = W_mix + W_cross
        σθ, σz = cauchy_from_W(W, lam, create_graph=create_graph)
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


def configure_recruitment_reference_stretch(fiber_like, ref_stretch: float):
    """
    Set the in-vivo axial stretch used to shift λ_lb for different circumferential curves.
    """
    ref = max(float(ref_stretch), 1e-6)
    with torch.no_grad():
        fiber_like._recruit_ref_stretch.copy_(torch.tensor(ref, dtype=torch.float64, device=fiber_like._recruit_ref_stretch.device))
    return float(fiber_like.recruit_ref_stretch().detach().cpu().item())

def estimate_lamz_iv_from_FL(blob, n_interp: int = 1000) -> Optional[float]:
    """
    Estimate the in-vivo axial stretch via the classical FL-curve intersection
    method (port of `lambdaz_iv.m`). At λ_z = λ_z_iv, axial force (and the wall
    axial Cauchy stress σ_z used here as a proxy) is approximately independent
    of luminal pressure, so the FL curves at different pressures intersect near
    a common λ_z. We average the intersection λ_z found between each consecutive
    pair (cyclic) of pressure protocols.

    Returns None if fewer than two FL protocols are available or the curves do
    not share an overlapping λ_z range.
    """
    fls = list(blob.get("fl", []))
    if len(fls) < 2:
        return None

    # Sort by protocol pressure parsed from the filename ("fl60", "fl100", …).
    pat = re.compile(r"fl(\d+)", re.I)
    def _level(s):
        m = pat.search(s.get("_file", ""))
        return int(m.group(1)) if m else 0
    fls = sorted(fls, key=_level)

    arrays = []
    for s in fls:
        lz = np.asarray(s["lamz"], dtype=float).ravel()
        sz = np.asarray(s["sig"],  dtype=float).ravel()
        finite = np.isfinite(lz) & np.isfinite(sz)
        lz, sz = lz[finite], sz[finite]
        if lz.size < 2:
            continue
        order = np.argsort(lz)
        lz, sz = lz[order], sz[order]
        _, uniq = np.unique(lz, return_index=True)  # de-duplicate λ_z
        arrays.append((lz[uniq], sz[uniq]))
    if len(arrays) < 2:
        return None

    lo = max(a[0][0]  for a in arrays)
    hi = min(a[0][-1] for a in arrays)
    if hi <= lo:
        return None
    grid = np.linspace(hi, lo, n_interp)  # high→low to match the MATLAB convention
    interps = [np.interp(grid, lz, sz) for (lz, sz) in arrays]

    n = len(interps)
    crosses = []
    for k in range(n):
        a = interps[k]
        b = interps[(k + 1) % n]
        diff = np.abs(a - b)
        if not np.all(np.isfinite(diff)):
            continue
        crosses.append(grid[int(np.argmin(diff))])
    if not crosses:
        return None
    return float(np.mean(crosses))


def infer_in_vivo_axial_stretch(blob) -> float:
    """
    Best-available estimate of the in-vivo axial stretch.

    Priority:
      1. FL-curve intersection method (Weizsäcker / Ferruzzi `lambdaz_iv.m`).
         Requires ≥2 FL protocols at different pressures with overlapping λ_z.
      2. PD100 curve median λ_z (often degenerate to 1.0 if the smoothed PD
         CSVs lack a real `lambda_z` column — falls back to filename scaling).
      3. Median across all PD protocols.
      4. 1.0 if nothing usable.
    """
    val = estimate_lamz_iv_from_FL(blob)
    if val is not None and 0.5 < val < 2.5:
        return val
    for s in blob.get("pd", []):
        if re.search(r"pd100", s.get("_file", ""), flags=re.I):
            return float(np.median(np.asarray(s["lamz"], dtype=float)))
    if blob.get("pd"):
        meds = [float(np.median(np.asarray(s["lamz"], dtype=float))) for s in blob["pd"]]
        meds = sorted(meds)
        return float(meds[len(meds)//2])
    return 1.0

def infer_in_vivo_circ_stretch(blob) -> float:
    """
    Use the PD100 curve as the in-vivo circumferential reference when available
    (PD100 is the pressure-diameter test at 100% of in-vivo axial stretch, so its
    median λ_θ is the reasonable in-vivo value). Fall back to the median across
    all PD curves, then 1.0 if no PD data exist.
    """
    for s in blob.get("pd", []):
        if re.search(r"pd100", s.get("_file", ""), flags=re.I):
            return float(np.median(np.asarray(s["lamθ"], dtype=float)))
    if blob.get("pd"):
        meds = [float(np.median(np.asarray(s["lamθ"], dtype=float))) for s in blob["pd"]]
        meds = sorted(meds)
        return float(meds[len(meds)//2])
    return 1.0


# ===================== Small-on-large linearised stiffness =====================
# Reference: Baek, Gleason, Rajagopal, Humphrey (2007) "Theory of small on large:
# potential utility in computations of fluid-solid interactions in arteries."
#
# We compute the spatial 4th-order elasticity tensor C^SoL at the in-vivo loaded
# state, restricted to the diagonal (no-shear) block plus the pre-stress
# contribution to shear components. The full shear material stiffness K_θzθz
# would require evaluating W with off-diagonal C entries, which the existing
# energy(lam) API does not support.
#
# Baek's formula (per data point, Cartesian principal coords, J=1):
#   C_{ijkl} = t^extra_{il} δ_{jk} + t^extra_{lj} δ_{ik}                        (pre-stress)
#            + (1/J) F_{iA} F_{jB} F_{kP} F_{lQ} K_{ABPQ}                       (material)
# where K_{ABPQ} = ∂²W / (∂E_{AB} ∂E_{PQ}) is the Lagrangian elasticity tensor.
# Chain rule for principal-axis perturbations (E_AA = (λ_A² − 1)/2):
#   K_AABB     = (1/(λ_A λ_B)) ∂²W/∂λ_A∂λ_B                       (A ≠ B)
#   K_AAAA     = (1/λ_A²) ∂²W/∂λ_A² − (1/λ_A³) ∂W/∂λ_A
def compute_small_on_large(model, lam_th_iv: float, lam_z_iv: float,
                            *, branch: str = "theta", device=None):
    """
    Evaluate the diagonal block of the Baek 2007 small-on-large stiffness tensor
    plus the pre-stress contribution to the in-plane and out-of-plane shears.

    Returns a dict with:
        F                : (3,)  in-vivo principal stretches [λ_r, λ_θ, λ_z]
        t_extra          : (3,)  diagonal extra Cauchy stress [t_rr, t_θθ, t_zz] (kPa)
        t_total          : (3,)  σ_θ, σ_z from cauchy_from_W; σ_r=0 enforced (kPa)
        pressure         : float Lagrange multiplier p (kPa)
        K_lagrangian     : (3,3) K_AABB block (kPa) in (r, θ, z) ordering
        C_diag           : (3,3) C_iiii / C_iijj block (kPa) in (r, θ, z) ordering
        C_shear_prestress: dict with C_{θzθz}, C_{rθrθ}, C_{rzrz} pre-stress only
        K_FSG            : (5,5) computational stiffness matrix (MPa)
        notes            : str   what is / isn't covered
    """
    if device is None:
        device = next(model.parameters()).device
    lam_r_iv = 1.0 / (lam_th_iv * lam_z_iv)
    lam = torch.tensor([lam_th_iv, lam_z_iv, lam_r_iv],
                       dtype=torch.float64, device=device).requires_grad_(True)

    # forward to get W, σ_θ, σ_z (with Lagrange-p baked in)
    model.eval()
    with torch.enable_grad():
        out = model_forward_branch(model, lam.unsqueeze(0), branch=branch, create_graph=True)
        # Some models return (W, σθ, σz); some return just (σθ, σz)
        if len(out) == 3:
            W_val, sθ, sz = out
        else:
            sθ, sz = out
            W_val = model_forward_branch(model, lam.unsqueeze(0), branch=branch, create_graph=True)[0]

        Wsum = W_val.sum() if hasattr(W_val, "sum") else W_val
        # 1st derivatives ∂W/∂λ → extra (no Lagrange p) Cauchy stress = λ_A · ∂W/∂λ_A
        (dW_dlam,) = torch.autograd.grad(Wsum, lam, create_graph=True)
        # 2nd derivatives — full 3×3 Hessian via per-row autograd
        H = torch.zeros(3, 3, dtype=torch.float64, device=device)
        for a in range(3):
            (row,) = torch.autograd.grad(dW_dlam[a], lam, create_graph=False, retain_graph=True)
            H[a, :] = row

    dW = dW_dlam.detach().cpu().numpy()                       # [θ, z, r]
    Hn = H.detach().cpu().numpy()
    sigma_th = float(sθ.detach().cpu().numpy().ravel()[0])
    sigma_z  = float(sz.detach().cpu().numpy().ravel()[0])

    # Reorder λ-derivative arrays from forward order (θ, z, r) to (r, θ, z)
    perm = np.array([2, 0, 1])
    F = np.array([lam_r_iv, lam_th_iv, lam_z_iv])             # (r, θ, z)
    dW_p = dW[perm]                                           # (r, θ, z)
    H_p  = Hn[np.ix_(perm, perm)]                             # (3,3) in (r, θ, z)

    # Extra Cauchy stress diag (no Lagrange p): t_extra_AA = λ_A · ∂W/∂λ_A
    t_extra = F * dW_p                                        # (3,)
    p_lag = t_extra[0]                                        # σ_r = t_rr_extra − p = 0 → p = t_rr_extra
    t_total = t_extra - p_lag                                 # σ_r = 0, σ_θ, σ_z

    # Lagrangian K_AABB (kPa)
    K_lag = np.zeros((3, 3))
    for A in range(3):
        for B in range(3):
            if A == B:
                K_lag[A, B] = (H_p[A, A] / (F[A] ** 2)
                               - dW_p[A] / (F[A] ** 3))
            else:
                K_lag[A, B] = H_p[A, B] / (F[A] * F[B])

    # Spatial diagonal block C_iiii / C_iijj using Baek (J = 1, principal coords)
    J = 1.0
    C_diag = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            pre = 0.0
            if i == j:
                pre = 2.0 * t_extra[i]                        # t_extra_il δ_jk + t_extra_lj δ_ik with i=j=k=l
            mat = (1.0 / J) * (F[i] ** 2) * (F[j] ** 2) * K_lag[i, j]
            C_diag[i, j] = pre + mat

    # Pre-stress contributions to the three relevant shear stiffnesses
    # (θzθz, rθrθ, rzrz). Material part requires off-diagonal C perturbation.
    C_shear_prestress = {
        "C_θzθz": float(t_extra[2]),     # t_extra_zz
        "C_rθrθ": float(t_extra[1]),     # t_extra_θθ (uses δ_rr in pre-stress)
        "C_rzrz": float(t_extra[2]),     # t_extra_zz
    }

    # Build the 5×5 K_FSG matrix in MPa, mirroring the MATLAB layout.
    # Index ordering: 1=θ, 2=z, 3=θz, 4=rθ, 5=rz
    Cqqqq = C_diag[1, 1]
    Czzzz = C_diag[2, 2]
    Cqqzz = C_diag[1, 2]
    Czzqq = C_diag[2, 1]
    Cqzqz = C_shear_prestress["C_θzθz"]    # pre-stress only
    Crqrq = C_shear_prestress["C_rθrθ"]    # pre-stress only
    Crzrz = C_shear_prestress["C_rzrz"]    # pre-stress only

    K_FSG_kPa = np.zeros((5, 5))
    K_FSG_kPa[0, 0] = Cqqqq
    K_FSG_kPa[0, 1] = Cqqzz
    K_FSG_kPa[1, 0] = Czzqq
    K_FSG_kPa[1, 1] = Czzzz
    K_FSG_kPa[2, 2] = Cqzqz
    K_FSG_kPa[3, 3] = Crqrq
    K_FSG_kPa[4, 4] = Crzrz
    # Off-diagonal coupling rows 1↔3 / 2↔3 require K_θzθθ etc., which need shear K
    K_FSG = K_FSG_kPa * 1e-3                                  # → MPa, matches MATLAB

    return dict(
        F=F, t_extra=t_extra, t_total=t_total, pressure=float(p_lag),
        K_lagrangian=K_lag, C_diag=C_diag,
        C_shear_prestress=C_shear_prestress, K_FSG=K_FSG,
        sigma_th=sigma_th, sigma_z=sigma_z,
        lam_th_iv=lam_th_iv, lam_z_iv=lam_z_iv,
        notes=("Diagonal block (Cθθθθ, Czzzz, Cθθzz, Czzθθ) is computed from the "
               "λ-Hessian of W via principal-axis chain rule; values are exact for "
               "the model. C_θzθz / C_rθrθ / C_rzrz include the Baek pre-stress "
               "term only — material K_θzθz etc. needs off-diagonal C perturbation, "
               "which the energy(lam) API does not currently expose."),
    )


def _save_small_on_large_txt(info: dict, age: str, outdir: str, *, label: str = "variant_f"):
    """Write a human-readable Baek 2007 small-on-large summary."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{age}_{label}_small_on_large.txt")
    with open(path, "w") as f:
        f.write(f"Small-on-large linearised stiffness — {label} — Age {age}\n")
        f.write("Reference: Baek, Gleason, Rajagopal, Humphrey (2007)\n")
        f.write("=" * 72 + "\n\n")

        f.write("In-vivo configuration (assumed equilibrium):\n")
        f.write(f"  λ_θ_iv     : {info['lam_th_iv']:.6f}\n")
        f.write(f"  λ_z_iv     : {info['lam_z_iv']:.6f}\n")
        f.write(f"  F = diag   : {np.array2string(info['F'], precision=6)}\n")
        f.write(f"  σ_θ_total  : {info['sigma_th']:.6e} kPa\n")
        f.write(f"  σ_z_total  : {info['sigma_z']:.6e} kPa\n")
        f.write(f"  Lagrange p : {info['pressure']:.6e} kPa\n\n")

        f.write("Extra Cauchy stress t_extra_AA (no Lagrange p) [kPa]:\n")
        names = ["rr", "θθ", "zz"]
        for n, v in zip(names, info["t_extra"]):
            f.write(f"  t_extra_{n} : {v:.6e}\n")
        f.write("\n")

        f.write("Lagrangian material stiffness K_AABB (kPa)  ordering (r, θ, z):\n")
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                f.write(f"  K_{ni}{nj}   : {info['K_lagrangian'][i, j]:+.6e}\n")
        f.write("\n")

        f.write("Spatial elasticity tensor C diagonal block (kPa)  ordering (r, θ, z):\n")
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                f.write(f"  C_{ni}{nj}{ni}{nj}: {info['C_diag'][i, j]:+.6e}\n")
        f.write("\n")

        f.write("Shear stiffness pre-stress contribution only (kPa):\n")
        for k, v in info["C_shear_prestress"].items():
            f.write(f"  {k:8s} : {v:+.6e}\n")
        f.write("\n")

        f.write("Computational K_FSG (5×5, MPa)  rows/cols = [θ, z, θz, rθ, rz]:\n")
        f.write(np.array2string(info["K_FSG"], precision=4, suppress_small=True) + "\n\n")

        f.write("C_out summary (MPa) [Cθθθθ, Czzzz, Cθθzz, Cθzθz_pre]:\n")
        Cqqqq = info["C_diag"][1, 1] * 1e-3
        Czzzz = info["C_diag"][2, 2] * 1e-3
        Cqqzz = info["C_diag"][1, 2] * 1e-3
        Cqzqz = info["C_shear_prestress"]["C_θzθz"] * 1e-3
        f.write(f"  [{Cqqqq:.6f}, {Czzzz:.6f}, {Cqqzz:.6f}, {Cqzqz:.6f}]\n\n")

        f.write("Notes:\n  " + info["notes"].replace("\n", "\n  ") + "\n")
    print(f"Small-on-large saved to: {path}")


def is_decomposed_model(model) -> bool:
    base = getattr(model, "base", model)
    return isinstance(base, DecomposedSEDFBase)

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

def _full_weighted_huber(model: nn.Module, lam_th, yth, lam_z, yz_t,
                          w_th: float, w_z: float,
                          robust_kind: str = "huber", robust_delta: float = 0.5) -> float:
    """Weighted Huber loss on the full dataset — same formula as the training objective."""
    model.eval()
    loss_val = 0.0
    with torch.enable_grad():
        if lam_th is not None and yth is not None:
            lam_eval = lam_th.detach().clone().requires_grad_(True)
            _, sθ, _ = model_forward_branch(model, lam_eval, branch="theta", create_graph=False)
            loss_val += w_th * _robust_weighted_loss(sθ.detach() - yth, None, robust_kind, robust_delta).item()
        if lam_z is not None and yz_t is not None:
            lam_eval = lam_z.detach().clone().requires_grad_(True)
            _, _, sz = model_forward_branch(model, lam_eval, branch="z", create_graph=False)
            loss_val += w_z * _robust_weighted_loss(sz.detach() - yz_t, None, robust_kind, robust_delta).item()
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

def _r2_huber_np(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 0.5) -> float:
    """Huber pseudo-R²: 1 - Σhuber(y-ŷ,δ) / Σhuber(y-ȳ,δ).  More robust to outliers."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if y.size == 0 or y.size != p.size:
        return float("nan")
    def _hsum(r):
        a = np.abs(r)
        return float(np.where(a < delta, 0.5*a*a, delta*(a - 0.5*delta)).sum())
    ss_tot = _hsum(y - y.mean())
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - _hsum(y - p) / ss_tot

def compute_fit_metrics(model: nn.Module, Xth, yth, Xz, yz, *,
                        label: str, loss: float = np.nan,
                        aic: float = np.nan, bic: float = np.nan,
                        robust_delta: float = 0.5) -> dict:
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
            row["r2_circumferential"] = _r2_huber_np(yth, pred, robust_delta)
            row["mse_circumferential"] = float(np.mean((np.asarray(yth) - pred) ** 2))
            mse_parts.append(row["mse_circumferential"])
        if Xz.size and yz.size:
            lam = torch.tensor(_stack_lams(Xz), dtype=torch.float64, device=device).requires_grad_(True)
            _, _, sz = model_forward_branch(model, lam, branch="z", create_graph=False)
            pred = sz.detach().cpu().numpy()
            row["r2_axial"] = _r2_huber_np(yz, pred, robust_delta)
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
              lr_base: float = 0.0,
              lr_resid: float = 0.0,
              wtheta_boost: float = 1.0,
              jitter: float = 2e-3,
              jitter_warm_epochs: int = 300,
              reg_gain: float = 5e-5,
              reg_basis: float = 5e-5,
              seed: Optional[int] = None,
              weights_th=None, weights_z=None,
              robust_kind: str = "huber", robust_delta: float = 0.25,
              batch_size: Optional[int] = None,
              compile_model: bool = False,
              val_split: float = 0,
              early_stop_patience: int = 500,
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

    use_validation = False
    best_val_loss = float('inf')
    patience_counter = 0
    
    if use_validation:
        # Split theta data
        if lam_th is not None and yθ is not None:
            n_th = len(yθ)
            n_val_th = int(n_th * val_split)
            idx = np.random.permutation(n_th)
            val_idx_th = idx[:n_val_th]
            train_idx_th = idx[n_val_th:]
            
            lam_th_train = lam_th[train_idx_th]
            lam_th_val = lam_th[val_idx_th]
            yth_train = yθ[train_idx_th]
            yth_val = yθ[val_idx_th]
            wth_train = wth_i[train_idx_th] if wth_i is not None else None
            wth_val = wth_i[val_idx_th] if wth_i is not None else None
        else:
            lam_th_train = lam_th
            lam_th_val = None
            yth_train = yθ
            yth_val = None
            wth_train = wth_i
            wth_val = None
        
        # Split z data
        if lam_z is not None and yz_t is not None:
            n_z = len(yz_t)
            n_val_z = int(n_z * val_split)
            idx = np.random.permutation(n_z)
            val_idx_z = idx[:n_val_z]
            train_idx_z = idx[n_val_z:]
            
            lam_z_train = lam_z[train_idx_z]
            lam_z_val = lam_z[val_idx_z]
            yz_train = yz_t[train_idx_z]
            yz_val = yz_t[val_idx_z]
            wz_train = wz_i[train_idx_z] if wz_i is not None else None
            wz_val = wz_i[val_idx_z] if wz_i is not None else None
        else:
            lam_z_train = lam_z
            lam_z_val = None
            yz_train = yz_t
            yz_val = None
            wz_train = wz_i
            wz_val = None
        
        print(f"    [{name}] Using validation split: {val_split*100:.0f}% for early stopping")
        if lam_th is not None:
            print(f"      θ: {len(train_idx_th)} train, {len(val_idx_th)} val")
        if lam_z is not None:
            print(f"      z: {len(train_idx_z)} train, {len(val_idx_z)} val")
    else:
        # No validation split for base models
        lam_th_train = lam_th
        lam_th_val = None
        yth_train = yθ
        yth_val = None
        wth_train = wth_i
        wth_val = None
        
        lam_z_train = lam_z
        lam_z_val = None
        yz_train = yz_t
        yz_val = None
        wz_train = wz_i
        wz_val = None

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

    # Check if decomposed symbolic model
    is_composite = is_decomposed_model(model)
    
    # Setup optimizer
    if is_composite:
        # Decomposed symbolic models: split base and tweak if present
        base_params = []
        tweak_params = []
        for name, p in model.named_parameters():
            if name.startswith("base."):
                base_params.append(p)
            elif name.startswith("tweak."):
                tweak_params.append(p)
            else:
                base_params.append(p)
        for p in base_params:
            p.requires_grad_(lr_base > 0.0 if lr_base != 0.0 else True)
        for p in tweak_params:
            p.requires_grad_(True)
        groups = []
        if base_params:
            lr_base_eff = lr_base if lr_base > 0.0 else lr
            groups.append({"params": base_params, "lr": lr_base_eff, "weight_decay": 1e-5})
        if tweak_params:
            lr_tweak = lr_resid if lr_resid > 0.0 else lr
            groups.append({"params": tweak_params, "lr": lr_tweak, "weight_decay": 2e-5})
        opt = torch.optim.Adam(groups if groups else model.parameters(), lr=lr, weight_decay=1e-5)
        max_clip = 10.0
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        max_clip = 100.0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    # Training loop
    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        
        loss = torch.tensor(0.0, device=device)
        jitter_now = 0.0 if ep <= jitter_warm_epochs else jitter
        
        # Optional: anneal gate temperature (if tweak module exposes .temp)
        if hasattr(model, "tweak") and hasattr(model.tweak, "temp"):
            model.tweak.temp = max(0.2, 1.0 * (0.97 ** (ep / 50)))

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

        # Anchor regularization: penalize moving away from reference parameters
        if anchor_lambda > 0.0 and anchor_params is not None:
            for name, p in model.named_parameters():
                if name in anchor_params:
                    loss = loss + anchor_lambda * (p - anchor_params[name]).pow(2).mean()

        loss = 55.0 * loss  # simple scaling (MATLAB-like)
        
        # Backward pass
        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=max_clip)
        opt.step()
        scheduler.step()

        # Validation and early stopping
        if use_validation and ep % 50 == 0:
            model.eval()
            val_loss = 0.0
            with torch.enable_grad():

                # Validate on θ
                if lam_th_val is not None and yth_val is not None:
                    lam_val = lam_th_val.detach().clone().requires_grad_(True)
                    _, sθ_val, _ = model_forward_branch(model, lam_val, branch="theta", create_graph=False)
                    val_loss += ((sθ_val.detach() - yth_val)**2).mean().item()
                
                # Validate on z
                if lam_z_val is not None and yz_val is not None:
                    lam_val = lam_z_val.detach().clone().requires_grad_(True)
                    _, _, sz_val = model_forward_branch(model, lam_val, branch="z", create_graph=False)
                    val_loss += ((sz_val.detach() - yz_val)**2).mean().item()
            
            # Early stopping check
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 50
            
            if patience_counter >= early_stop_patience:
                print(f"    [{name}] Early stopping at ep{ep}: validation loss not improving (best={best_val_loss:.4e})")
                break
            
            model.train()

        if ep % 50 == 0:
            full_loss_now = _full_plain_mse(model, lam_th, yθ, lam_z, yz_t)
            if full_loss_now < best_full_loss:
                best_full_loss = full_loss_now
                best_state = copy.deepcopy(model.state_dict())

        # Periodic logging
        if ep % max(epochs//10, 200) == 0:
            aic, bic = _aic_bic(float(loss.item()), n_total, model)
            val_str = f", val_loss={best_val_loss:.4e}" if use_validation else ""
            print(f"    [{name}] ep{ep}: loss={float(loss.item()):.4e}, full_mse={full_loss_now:.4e}, AIC={aic:.1f}, BIC={bic:.1f}, LR={opt.param_groups[0]['lr']:.2e}{val_str}")

    # Restore the best checkpoint measured on the same full-data MSE used by
    # the comparison plots and the "Final (full)" report below.
    model.load_state_dict(best_state)

    # Final evaluation on full dataset (or validation set for ICNN)
    model.eval()
    
    loss_val = 0.0
    eval_th = lam_th_val if use_validation and lam_th_val is not None else lam_th
    eval_yth = yth_val if use_validation and yth_val is not None else yθ
    eval_z = lam_z_val if use_validation and lam_z_val is not None else lam_z
    eval_yz = yz_val if use_validation and yz_val is not None else yz_t

    with torch.enable_grad():
        if eval_th is not None and eval_yth is not None:
            lam_eval = eval_th.detach().clone().requires_grad_(True)
            _, sθ, _ = model_forward_branch(model, lam_eval, branch="theta", create_graph=False)
            loss_val += ((sθ.detach() - eval_yth)**2).mean().item()
        if eval_z is not None and eval_yz is not None:
            lam_eval = eval_z.detach().clone().requires_grad_(True)
            _, _, sz = model_forward_branch(model, lam_eval, branch="z", create_graph=False)
            loss_val += ((sz.detach() - eval_yz)**2).mean().item()

    aic, bic = _aic_bic(loss_val, n_total, model)
    eval_set = "validation" if use_validation else "full"
    print(f"    [{name}] Final ({eval_set}): loss={loss_val:.4e}, AIC={aic:.1f}, BIC={bic:.1f}")
    if best_full_loss + 1e-12 < loss_val:
        print(f"    [{name}] Restored best full-data MSE checkpoint: {best_full_loss:.4e}")
    
    if is_composite:
        base_module = model.base if hasattr(model, 'base') else model
        print(f"    [{name}] Base decomposition: iso={base_module.iso_model}, aniso={base_module.aniso_model}")
    
    return loss_val, aic, bic, model

# =============================== Plotting =============================
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
        ax_left.scatter(lam_theta, sig_theta, s=25, c="k", alpha=0.5, label="Experimental", edgecolors='none', zorder=1)
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
                ax_left.plot(x_unique, y_smooth, color=color, lw=4, alpha=1.0, ls=ls,
                             label=(lbl_th if j==0 else None), zorder=2)
        ax_left.set_xlabel("λθ", fontsize=12, fontweight='bold'); ax_left.set_ylabel("σθ [kPa]", fontsize=12, fontweight='bold')
        # Set only 3 ticks on each axis with larger labels
        ax_left.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_left.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_left.tick_params(axis='both', which='major', labelsize=14)
    if Xz.size:
        lam_theta, lam_z, sig_z = Xz[:,0], Xz[:,1], yz
        ax_right.scatter(lam_z, sig_z, s=25, c="k", alpha=0.5, label="Experimental", edgecolors='none', zorder=1)
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
                ax_right.plot(x_unique, y_smooth, color=color, lw=4, alpha=1.0, ls=ls,
                              label=(lbl_z if j==0 else None), zorder=2)
        ax_right.set_xlabel("λz", fontsize=12, fontweight='bold'); ax_right.set_ylabel("σz [kPa]", fontsize=12, fontweight='bold')
        # Set only 3 ticks on each axis with larger labels
        ax_right.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_right.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_right.tick_params(axis='both', which='major', labelsize=14)

def plot_results(Xth, yth, Xz, yz, base_model, tweaked_model, age, outdir, round_step=0.01, ids_th=None, ids_z=None):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = False  # Use matplotlib's built-in LaTeX-like rendering
    plt.rcParams['mathtext.default'] = 'regular'
    
    fig1, axs1 = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)
    _group_and_plot(axs1[0], axs1[1], Xth, yth, Xz, yz, base_model, "#DC143C", "--",
                    r"$\Psi_{base}$", r"$\Psi_{base}$", round_step, device, ids_th=ids_th, ids_z=ids_z)
    _group_and_plot(axs1[0], axs1[1], Xth, yth, Xz, yz, tweaked_model, "#0066CC", "--",
                    r"$\Psi_{base} + \Psi_{cross}$", r"$\Psi_{base} + \Psi_{cross}$", round_step, device, ids_th=ids_th, ids_z=ids_z)
    fig1.suptitle(f"Age {age}: Experimental vs Base vs Poly", fontsize=18, weight="bold")
    fig1.savefig(os.path.join(outdir, f"{age}_fits.png"), dpi=300); plt.close(fig1)  # Higher DPI for better quality

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
        x_all, y_np, alpha=0.7, s=35, color="black",
        label="Experimental", edgecolors="gray", linewidths=0.5, zorder=1
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
            ax.plot(x, sig_np, color=color, lw=2.5, alpha=0.9,
                    label=(label if first else None), zorder=2)
            ax.scatter(x, sig_np, color=color, s=28, alpha=0.9,
                       edgecolors="white", linewidths=0.4, zorder=3)
            first = False

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="both", which="major", labelsize=14)

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
    "f": dict(iso="poly",       recruit=True,  disp=True,
              label="f: Combined Eq (Quad+4Fib+rec+disp+cross)"),
    "g": dict(iso="poly",       recruit=False, disp=False,
              label="g: Combined Eq (Quad+4Fib+no rec+no disp+cross)"),
}


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

    if variant_key in list("abcde"):
        # Stage 1 only: pure base model, no symbolic cross terms.
        # No warm-start — the old Base_Poly pipeline also fitted from scratch,
        # and params from the discovery run (which co-optimised with cross terms)
        # lead to a worse basin for a base-only fit.
        base = DecomposedSEDFBase(
            iso_model=cfg["iso"], aniso_model="4fiber",
            use_recruitment=cfg["recruit"], use_dispersion=cfg["disp"],
        ).to(device)
        if hasattr(base.aniso, "dist_type"):
            base.aniso.dist_type = "lognormal"
        if hasattr(base.aniso, "_recruit_ref_stretch"):
            configure_recruitment_reference_stretch(base.aniso, in_vivo_lamz)
        if hasattr(base.aniso, "_lambda_lb_raw") and cfg["recruit"]:
            configure_recruitment_start(base.aniso, args.recruit_start_init, args.recruit_start_mode)
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
            jitter=0.0, jitter_warm_epochs=0,
        )
        return model, loss, aic, bic, label

    else:
        # Variants f and g: Stage 1 base then Stage 2 fixed combined-equation cross terms
        # f: recruit=True,  disp=True
        # g: recruit=False, disp=False

        use_rec  = cfg["recruit"]
        use_disp = cfg["disp"]
        vname    = variant_key.upper()

        # Stage 1 ---------------------------------------------------
        base = DecomposedSEDFBase(
            iso_model="poly", aniso_model="4fiber",
            use_recruitment=use_rec, use_dispersion=use_disp,
        ).to(device)
        if hasattr(base.aniso, "dist_type"):
            base.aniso.dist_type = "lognormal"
        if fit_cfg is not None:
            apply_fit_config_params(base, fit_cfg, device)
        if hasattr(base.aniso, "_recruit_ref_stretch"):
            configure_recruitment_reference_stretch(base.aniso, in_vivo_lamz)
        if hasattr(base.aniso, "_lambda_lb_raw") and use_rec:
            configure_recruitment_start(base.aniso, args.recruit_start_init, args.recruit_start_mode)
        _, _, _, base = fit_model(
            base, Xth, yth, Xz, yz,
            epochs=args.epochs_base,
            name=f"Variant-{vname}-Stage1",
            lr=args.lr_base,
            wtheta_boost=args.wtheta_boost,
            weights_th=wth, weights_z=wz,
            robust_kind=args.robust,
            robust_delta=args.robust_delta,
            batch_size=(args.batch_size or None),
            compile_model=args.compile,
            seed=args.seed,
            jitter=0.0, jitter_warm_epochs=0,
        )

        # Stage 2: freeze base, learn fixed cross-term correction ---
        combined = CombinedEquationSEDF(
            n_basis=args.n_basis,
            use_recruitment=use_rec,
            use_dispersion=use_disp,
            term_list=TERM_SETS[getattr(args, "term_set", "v1")],
        ).to(device)
        with torch.no_grad():
            combined.base.load_state_dict(base.state_dict(), strict=False)
        if hasattr(combined.base.aniso, "dist_type"):
            combined.base.aniso.dist_type = "lognormal"
        if hasattr(combined.base.aniso, "_recruit_ref_stretch"):
            configure_recruitment_reference_stretch(combined.base.aniso, in_vivo_lamz)

        lam_all = _lam_all_from_X(Xth, Xz, device=device)
        with torch.no_grad():
            combined.set_stats(lam_all)
            for p in combined.base.parameters():
                p.requires_grad_(False)
            for p in combined.tweak.parameters():
                p.requires_grad_(True)

        loss, aic, bic, combined = fit_model(
            combined, Xth, yth, Xz, yz,
            epochs=args.epochs_warmup,
            name=f"Variant-{vname}-Stage2",
            lr=args.lr_resid,
            wtheta_boost=args.wtheta_boost,
            weights_th=wth, weights_z=wz,
            robust_kind=args.robust,
            robust_delta=args.robust_delta,
            batch_size=(args.batch_size or None),
            compile_model=args.compile,
            seed=args.seed,
        )

        # Stage 3 (optional): Joint fine-tune — unfreeze base + tweak together.
        # The base was optimized before cross terms existed; a joint pass lets it
        # re-adapt to complement the fixed symbolic correction.
        if args.epochs_joint > 0:
            for p in combined.parameters():
                p.requires_grad_(True)
            loss, aic, bic, combined = fit_model(
                combined, Xth, yth, Xz, yz,
                epochs=args.epochs_joint,
                name=f"Variant-{vname}-Stage3-Joint",
                lr=args.lr_resid,
                lr_base=args.lr_joint_base,
                lr_resid=args.lr_resid,
                wtheta_boost=args.wtheta_boost,
                weights_th=wth, weights_z=wz,
                robust_kind=args.robust,
                robust_delta=args.robust_delta,
                batch_size=(args.batch_size or None),
                compile_model=args.compile,
                seed=args.seed,
            )

        return combined, loss, aic, bic, label


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


def _format_state_dict_lines(state, skip_prefix=()):
    """Return a list of "  key: value" strings for a flattened state_dict."""
    lines = []
    for key, tensor in state.items():
        if any(key.startswith(p) for p in skip_prefix):
            continue
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 0 or arr.size == 1:
            lines.append(f"  {key}: {float(arr):.10g}")
        else:
            lines.append(
                f"  {key}: {np.array2string(arr.ravel(), precision=8, separator=', ')}"
            )
    return lines


def _save_base_only_params(model, variant_key, age, outdir):
    """Single-txt parameter dump for the base-only variants (a–e)."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{age}_variant_{variant_key}_params.txt")
    cfg = VARIANT_CONFIGS[variant_key]
    with open(path, "w") as f:
        f.write(f"Base parameters — Variant {variant_key.upper()} — Age {age}\n")
        f.write(f"  {cfg['label']}\n")
        f.write("=" * 72 + "\n\n")
        f.write("Model: DecomposedSEDFBase  (no symbolic cross terms)\n")
        f.write(f"  isotropic   : {cfg['iso']}\n")
        f.write(f"  anisotropic : 4fiber  (recruit={cfg['recruit']}, disp={cfg['disp']})\n\n")
        f.write("State dict (gradient-trained):\n")
        f.write("-" * 72 + "\n")
        for line in _format_state_dict_lines(model.state_dict()):
            f.write(line + "\n")
    print(f"Variant-{variant_key} params saved to: {path}")


def _save_combined_equation(model, variant_key, age, outdir, term_set_name="v1"):
    """Save the combined-equation fitted weights for variant f or g.

    Writes four files:
      {age}_variant_{vk}_equation.txt   — short human-readable summary (weights only)
      {age}_variant_{vk}_params.txt     — full single-txt: weights + every base param
      {age}_variant_{vk}_weights.csv    — tidy cross-term weights (Stage B input)
      {age}_variant_{vk}_base_params.csv — flattened base parameters (Stage B input)
    """
    os.makedirs(outdir, exist_ok=True)
    eq_path     = os.path.join(outdir, f"{age}_variant_{variant_key}_equation.txt")
    params_path = os.path.join(outdir, f"{age}_variant_{variant_key}_params.txt")
    w_csv_path  = os.path.join(outdir, f"{age}_variant_{variant_key}_weights.csv")
    bp_csv_path = os.path.join(outdir, f"{age}_variant_{variant_key}_base_params.csv")
    cfg = VARIANT_CONFIGS[variant_key]
    rec_str  = "rec"    if cfg["recruit"] else "no rec"
    disp_str = "disp"   if cfg["disp"]    else "no disp"
    active_idx = model.tweak.active_idx.cpu().numpy().tolist()
    w_vals = model.tweak.w.detach().cpu().numpy()
    names = get_basis_names(model, fallback_n=model.n_basis)
    term_list_used = TERM_SETS.get(term_set_name, COMBINED_FIXED_TERMS)

    with open(eq_path, "w") as f:
        f.write(f"Combined Equation Fitted Weights — Variant {variant_key.upper()} — Age {age}\n")
        f.write(f"Term set: {term_set_name}  ({len(term_list_used)} terms)\n")
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
        f.write(f"\nFixed term set ({term_set_name}):\n  {term_list_used}\n")
    print(f"Variant-{variant_key} equation saved to: {eq_path}")

    # Comprehensive single-txt: cross weights + every base parameter
    with open(params_path, "w") as f:
        f.write(f"Combined Equation parameters — Variant {variant_key.upper()} — Age {age}\n")
        f.write(f"  {cfg['label']}\n")
        f.write(f"  Term set: {term_set_name}  ({len(term_list_used)} terms)\n")
        f.write("=" * 72 + "\n\n")
        f.write("Cross-term weights (W_cross = Σ w_i · ψ_i):\n")
        f.write("-" * 72 + "\n")
        for idx, w in zip(active_idx, w_vals):
            name = names[idx] if idx < len(names) else f"term_{idx}"
            f.write(f"  [idx {idx:2d}] {name:25s}: w = {float(w): .8e}\n")
        f.write("\nBase parameters (state_dict, gradient-trained):\n")
        f.write(f"  isotropic   : poly\n")
        f.write(f"  anisotropic : 4fiber  (recruit={cfg['recruit']}, disp={cfg['disp']})\n")
        f.write("-" * 72 + "\n")
        for line in _format_state_dict_lines(model.base.state_dict()):
            f.write(line + "\n")
    print(f"Variant-{variant_key} params saved to: {params_path}")

    # tidy weights CSV
    with open(w_csv_path, "w") as f:
        f.write("term_idx,term_name,w\n")
        for idx, w in zip(active_idx, w_vals):
            name = names[idx] if idx < len(names) else f"term_{idx}"
            f.write(f"{idx},{name},{float(w):.10g}\n")
    print(f"Variant-{variant_key} weights CSV saved to: {w_csv_path}")

    # tidy base params CSV (flatten tensors element-wise)
    rows = []
    base_state = model.base.state_dict()
    for key, tensor in base_state.items():
        arr = tensor.detach().cpu().numpy().reshape(-1)
        if arr.size == 1:
            rows.append((f"base.{key}", "", float(arr[0])))
        else:
            for i, v in enumerate(arr):
                rows.append((f"base.{key}", str(i), float(v)))
    with open(bp_csv_path, "w") as f:
        f.write("param,index,value\n")
        for name, idx, val in rows:
            f.write(f"{name},{idx},{val:.10g}\n")
    print(f"Variant-{variant_key} base params CSV saved to: {bp_csv_path}")


def _plot_comparison(age, variant_models, Xth, yth, Xz, yz, outdir,
                     ids_th=None, ids_z=None, round_step=0.01):
    """Save a 6×2 comparison figure (one row per variant, circ + axial)."""
    os.makedirs(outdir, exist_ok=True)
    n_variants = len(variant_models)
    fig, axes = plt.subplots(n_variants, 2, figsize=(14, 4 * n_variants),
                             constrained_layout=True)
    if n_variants == 1:
        axes = axes[np.newaxis, :]
    device = next(next(iter(variant_models.values()))[0].parameters()).device
    colors = ["#DC143C", "#FF8C00", "#228B22", "#0066CC", "#8B008B", "#000000", "#008B8B"]
    for row_i, (vkey, (model, label)) in enumerate(variant_models.items()):
        ax_l, ax_r = axes[row_i, 0], axes[row_i, 1]
        color = colors[row_i % len(colors)]
        _group_and_plot(ax_l, ax_r, Xth, yth, Xz, yz, model,
                        color=color, ls="-",
                        lbl_th=label, lbl_z=label,
                        round_step=round_step,
                        device=device, ids_th=ids_th, ids_z=ids_z)
        ax_l.set_title(f"{label}\nCircumferential", fontsize=9, weight="bold")
        ax_r.set_title(f"{label}\nAxial", fontsize=9, weight="bold")
        ax_l.legend(fontsize=7)
        ax_r.legend(fontsize=7)
    fig.suptitle(f"Model Comparison — Age {age}", fontsize=14, weight="bold")
    plot_path = os.path.join(outdir, f"{age}_comparison_fits.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to: {plot_path}")


def run_comparison(args, age, Xth, yth, Xz, yz, wth, wz, ids_th, ids_z,
                   in_vivo_lamz, device):
    """Fit all 7 variants (a–g) and write unified comparison outputs."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON MODE — Age {age}")
    print(f"{'='*60}")

    # Load per-animal config (hyperparameters + warm-start initial conditions)
    fit_cfg = load_fit_config(args.data_root, age)
    if fit_cfg is not None:
        apply_fit_config_hparams(fit_cfg, args)
        print(f"  → Applied per-animal hyperparameters for {age}")

    # Pre-compute reporting weights (same formula as fit_model's adaptive weighting)
    # Used only for the reported loss in the comparison table, not for training.
    _yth_t = torch.tensor(yth, dtype=torch.float64, device=device)
    _yz_t  = torch.tensor(yz,  dtype=torch.float64, device=device)
    _lam_th_r = torch.tensor(_stack_lams(Xth), dtype=torch.float64, device=device) if Xth.size else None
    _lam_z_r  = torch.tensor(_stack_lams(Xz),  dtype=torch.float64, device=device) if Xz.size else None
    _w_th_r = args.wtheta_boost * _inv_std_safe(_yth_t)
    _w_z_r  = _inv_std_safe(_yz_t)
    _n_th_r, _n_z_r = len(yth), len(yz)
    if _n_th_r and _n_z_r:
        _w_th_r *= (_n_th_r + _n_z_r) / (2 * _n_th_r + 1e-12)
        _w_z_r  *= (_n_th_r + _n_z_r) / (2 * _n_z_r  + 1e-12)

    metric_rows = []
    variant_models = {}   # ordered dict preserving insertion order

    for vkey in list("abcdefg"):
        cfg = VARIANT_CONFIGS[vkey]
        print(f"\n--- Variant {vkey.upper()}: {cfg['label']} ---")
        model, _plain_loss, aic, bic, label = _fit_variant(
            vkey, args, Xth, yth, Xz, yz, wth, wz, in_vivo_lamz, device,
            fit_cfg=fit_cfg,
        )
        # Report the weighted Huber loss (same objective as training) and Huber R².
        w_loss = _full_weighted_huber(model, _lam_th_r, _yth_t, _lam_z_r, _yz_t,
                                      _w_th_r, _w_z_r, args.robust, args.robust_delta)
        row = compute_fit_metrics(model, Xth, yth, Xz, yz,
                                   label=label, loss=w_loss, aic=aic, bic=bic,
                                   robust_delta=args.robust_delta)
        metric_rows.append(row)
        variant_models[vkey] = (model, label)
        print(f"    R²_circ={row['r2_circumferential']:.4f}  "
              f"R²_axial={row['r2_axial']:.4f}  "
              f"Loss={row['loss']:.4e}  AIC={row['aic']:.1f}")

    _save_comparison_metrics(metric_rows, age, args.out_root)
    _plot_comparison(age, variant_models, Xth, yth, Xz, yz, args.out_root,
                     ids_th=ids_th, ids_z=ids_z, round_step=args.round_step)
    # Per-variant params + small-on-large outputs
    lam_th_iv = getattr(args, "lam_th_iv", None)
    if lam_th_iv is None:
        # Median λ_θ across all PD samples ≈ in-vivo state when PD100 dominates the dataset.
        lam_th_iv = float(np.median(Xth[:, 0])) if Xth.size else 1.0
    lam_z_iv = getattr(args, "lam_z_iv", None) or in_vivo_lamz
    for vkey, (vmodel, _vlabel) in variant_models.items():
        if vkey in ("f", "g"):
            _save_combined_equation(vmodel, vkey, age, args.out_root,
                                    term_set_name=getattr(args, "term_set", "v1"))
        else:
            _save_base_only_params(vmodel, vkey, age, args.out_root)
        try:
            sol_info = compute_small_on_large(vmodel, lam_th_iv, lam_z_iv, device=device)
            _save_small_on_large_txt(sol_info, age, args.out_root, label=f"variant_{vkey}")
        except Exception as e:
            print(f"  [SoL] variant {vkey} failed: {type(e).__name__}: {e}")

    print(f"\n[comparison done] outputs in: {args.out_root}")


# ================================= Main ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_root",  default="runs_sef")

    # Epochs
    ap.add_argument("--epochs_base",   type=int, default=4000)
    ap.add_argument("--epochs_warmup", type=int, default=7000)
    ap.add_argument("--epochs_joint",  type=int, default=3000,
                    help="Stage 3 joint fine-tune epochs for variants f/g (base+tweak co-trained). Default 0 (off).")

    # LRs
    ap.add_argument("--lr_base",        type=float, default=5e-2)
    ap.add_argument("--lr_resid",       type=float, default=1e-3)
    ap.add_argument("--lr_joint_base",  type=float, default=1e-3,
                    help="Base LR for Stage 3 joint fine-tune. Default 1e-3.")

    # Regularizers
    ap.add_argument("--reg_gain",  type=float, default=1e-5)
    ap.add_argument("--reg_basis", type=float, default=1e-8)

    # Model choices
    ap.add_argument("--recruit_start_mode", choices=["learn","fixed"], default="learn",
                help="Treat the shared recruitment start λ_lb as a learned parameter or hold it fixed")
    ap.add_argument("--recruit_start_init", type=float, default=0.75,
                help="Initial guess or fixed value for the shared recruitment start λ_lb")
    ap.add_argument("--recruit_start_min", type=float, default=0.30,
                help="Lower bound for the learned recruitment start λ_lb")
    ap.add_argument("--recruit_start_max", type=float, default=2.55,
                help="Upper bound for the learned recruitment start λ_lb")
    ap.add_argument("--recruit_end_min", type=float, default=1.15,
                help="Lower bound for the recruitment upper stretch λ_ub")
    ap.add_argument("--recruit_end_max", type=float, default=2.80,
                help="Upper bound for the recruitment upper stretch λ_ub")
    ap.add_argument("--robust", choices=["huber","charbonnier","mse"], default="huber")
    ap.add_argument("--robust_delta", type=float, default=1.5)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wtheta_boost", type=float, default=4.5)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--jitter_warm_epochs", type=int, default=800)
    ap.add_argument("--seed", type=int, default=125)
    ap.add_argument("--round_step", type=float, default=0.01)

    # Small-on-large (Baek 2007) overrides; both default to data-derived values.
    ap.add_argument("--lam_th_iv", type=float, default=None,
                help="In-vivo circumferential stretch for Baek 2007 small-on-large. "
                     "Default: median λ_θ from the PD100 curve.")
    ap.add_argument("--lam_z_iv", type=float, default=None,
                help="In-vivo axial stretch override for small-on-large. "
                     "Default: same as the recruitment reference (PD100 median λ_z).")

    # Basis options
    ap.add_argument("--n_basis", type=int, default=45,
                help="Number of symbolic candidate terms in the fixed combined-equation library")
    ap.add_argument("--term_set", choices=list(TERM_SETS.keys()), default="v1",
                help="Which fixed cross-term set the combined equation uses. "
                     "v1 = original 10 terms (≥3/12 frequency). "
                     "v2 = 13 terms (v1 + <I₄θ-1>, (<I₄θ-1>)³, exp(<I₄z-1>)-1) for "
                     "the unified cross-age workflow.")
    # Variant / comparison mode
    ap.add_argument(
        "--variant",
        choices=list("abcdefg") + ["comparison"],
        default="f",
        help=(
            "comparison (default): fit all 7 variants (a-g) and save a unified table. "
            "a-e: base-only variants (no cross terms). "
            "f: Combined Eq (rec+disp+cross). "
            "g: Combined Eq (no rec, no disp, +cross)."
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

        if args.variant in list("abcdefg"):
            fit_cfg = load_fit_config(args.data_root, age)
            if fit_cfg is not None:
                apply_fit_config_hparams(fit_cfg, args)
                print(f"  → Applied per-animal hyperparameters for {age}")
            model, _plain_loss, aic, bic, label = _fit_variant(
                args.variant, args, Xth, yth, Xz, yz, wth, wz, in_vivo_lamz, device,
                fit_cfg=fit_cfg,
            )
            # Compute weighted Huber loss for reporting
            _yth_t = torch.tensor(yth, dtype=torch.float64, device=device)
            _yz_t  = torch.tensor(yz,  dtype=torch.float64, device=device)
            _lam_th_r = torch.tensor(_stack_lams(Xth), dtype=torch.float64, device=device) if Xth.size else None
            _lam_z_r  = torch.tensor(_stack_lams(Xz),  dtype=torch.float64, device=device) if Xz.size else None
            _w_th_r = args.wtheta_boost * _inv_std_safe(_yth_t)
            _w_z_r  = _inv_std_safe(_yz_t)
            _n_th_r, _n_z_r = len(yth), len(yz)
            if _n_th_r and _n_z_r:
                _w_th_r *= (_n_th_r + _n_z_r) / (2 * _n_th_r + 1e-12)
                _w_z_r  *= (_n_th_r + _n_z_r) / (2 * _n_z_r  + 1e-12)
            w_loss = _full_weighted_huber(model, _lam_th_r, _yth_t, _lam_z_r, _yz_t,
                                          _w_th_r, _w_z_r, args.robust, args.robust_delta)
            row = compute_fit_metrics(model, Xth, yth, Xz, yz,
                                       label=label, loss=w_loss, aic=aic, bic=bic,
                                       robust_delta=args.robust_delta)
            save_fit_metrics_table([row], age, args.out_root)
            plot_results(Xth, yth, Xz, yz, model, model, age, args.out_root,
                         round_step=args.round_step, ids_th=ids_th, ids_z=ids_z)
            if args.variant in ("f", "g"):
                _save_combined_equation(model, args.variant, age, args.out_root,
                                        term_set_name=getattr(args, "term_set", "v1"))
            else:
                _save_base_only_params(model, args.variant, age, args.out_root)
            # Small-on-large (Baek 2007) at the in-vivo state
            lam_th_iv = (args.lam_th_iv if args.lam_th_iv is not None
                         else (float(np.median(Xth[:, 0])) if Xth.size else 1.0))
            lam_z_iv = args.lam_z_iv if args.lam_z_iv is not None else in_vivo_lamz
            try:
                sol_info = compute_small_on_large(model, lam_th_iv, lam_z_iv, device=device)
                _save_small_on_large_txt(sol_info, age, args.out_root,
                                         label=f"variant_{args.variant}")
            except Exception as e:
                print(f"  [SoL] failed: {type(e).__name__}: {e}")
            print(f"  R²_circ={row['r2_circumferential']:.4f}  "
                  f"R²_axial={row['r2_axial']:.4f}  Loss={row['loss']:.4e}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
