#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eq9 (+Gent or PoleZero) + ICNN-MoE residual with fiber recruitment (half-normal over slack stretches).
- Stable stresses via autograd (compute σ from W with gradients).
- Robust weighting; equal weight per physical test (original + resampled balanced).
- Optional fiber dispersion (GOH κ) and per-family knee width τ.
- NEW: Half-normal recruitment Γ_s(λ_s) on λ_s ∈ (1, λ_ub] with parameter σ_s, integrated by quadrature.
- Residuals: tiny input-convex ΔΨ (MoE) + tiny stress-head correction.

Examples
--------
python run_sef.py --iso gent --recruit on --epochs_base 4000 --epochs_warmup 1500 --epochs_joint 3000 \
  --lr_base 4e-3 --lr_resid 3e-3 --lr_joint_base 1e-5 --lr_joint_resid 1e-4 --compile
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
EXP_ARG_MAX = 100.0
ZERO_TRANSITION_EPS = 1e-2
RECRUIT_LB_MIN = 0.90
RECRUIT_LB_MAX = 1.25
RECRUIT_UB_MIN = 1.15
RECRUIT_UB_MAX = 1.40

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

# ================= Residuals: tiny ICNN + tiny stress head =============
class OneTermTweak(nn.Module):
    def __init__(self, use_gate=True):
        super().__init__()
        self._w = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))  # starts at 0
        self.use_gate = use_gate
        self._center = nn.Parameter(torch.tensor(1.8, dtype=torch.float64))
        self._width_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        C = C_from_lambdas(lam)
        I1v = torch.clamp(I1(C) - 3.0, min=0.0)
        base = I1v*I1v  # (I1-3)^2

        if not self.use_gate:
            g = 1.0
        else:
            lam_max, _ = lam[..., :2].max(dim=-1)
            width = softplus_pos(self._width_raw) + 1e-6
            g = torch.sigmoid((lam_max - self._center)/width)

        w = self._w  # allow signed; if you want strictly positive, wrap softplus
        return w * g * base
class CompositeWithTweak(nn.Module):
    def __init__(self, elastin_type="neohookean", tweak_gate=True, use_recruitment=True, phi_constraint="sum1"):
        super().__init__()
        self.base = CompositeSEDFBase(
            elastin_type=elastin_type,
            use_recruitment=use_recruitment,
            phi_constraint=phi_constraint,
        )
        self.tweak = OneTermTweak(use_gate=tweak_gate)

    def forward(self, lam, create_graph=True):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        W_mix, *_ = self.base.energy_parts(lam)
        W = W_mix + self.tweak(lam)
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

class GatedTweakLinear(nn.Module):
    """
    W_cross = sum_i (g_i * w_i * psi_i)
    g_i in (0,1) via sigmoid(logit_i / temp)
    Sparsity via L1 penalty on g.
    """
    def __init__(self, n_basis: int, init_gate: float = 0.2, temp: float = 1.0, dropout_p: float = 0.0):
        super().__init__()
        self.n_basis = n_basis
        self._w_raw = nn.Parameter(torch.zeros(n_basis, dtype=torch.float64))
        
        # gate logits initialized so sigmoid(logit)=init_gate
        init_gate = float(np.clip(init_gate, 1e-4, 1-1e-4))
        init_logit = math.log(init_gate/(1-init_gate))
        self.logits = nn.Parameter(torch.full((n_basis,), init_logit, dtype=torch.float64))
        
        self.temp = temp  # you can anneal this during training
        self.dropout_p = float(np.clip(dropout_p, 0.0, 0.95))
    
    def gates(self):
        return torch.sigmoid(self.logits / self.temp)  # (n_basis,)

    @property
    def w(self):
        return softplus_pos(self._w_raw)
    
    def forward(self, basis_feats):  # (..., n_basis)
        g = self.gates()
        eff_w = g * self.w
        if self.training and self.dropout_p > 0.0:
            keep_p = 1.0 - self.dropout_p
            mask = torch.bernoulli(torch.full_like(eff_w, keep_p))
            if torch.all(mask == 0):
                mask = torch.ones_like(mask)
            eff_w = eff_w * (mask / keep_p)
        return (basis_feats * eff_w).sum(dim=-1)
    
    def l1_gate_penalty(self):
        # since g>=0, abs(g)=g
        return self.gates().sum()
    
    @torch.no_grad()
    def active_idx(self, thr: float = 0.5):
        g = self.gates().detach().cpu().numpy()
        return np.where(g > thr)[0].tolist()

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
        # Higher invariants — I5 is new info only for off-axis (helical) families;
        # I8 captures the signed difference between how the two helical families are loaded.
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

        # Cubic terms: strain and all fiber invariants (for steep heel-region stiffening)
        cube_keys = ["<E_θ>", "<E_z>", "<I₄θ-1>", "<I₄z-1>", "<I₄D1-1>", "<I₄D2-1>"]
        for key in cube_keys:
            if key in feat_map:
                feats.append((f"({key})³", feat_map[key] ** 3))

        # Matrix × fiber cross terms
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

        # Fiber–fiber cross terms (coupling between fiber families)
        fiber_cross_pairs = [
            ("<I₄θ-1>", "<I₄z-1>"),        # circumferential × axial fiber
            ("<I₄D1-1>", "<I₄D2-1>"),       # helical family interaction
            ("<I₄θ-1>", "<I₄D1-1>"),        # circumferential × helical
            ("<I₄z-1>", "<I₄D2-1>"),        # axial × helical
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
        # Exp now applied to fiber invariants too — gives Fung-type exponential fiber terms
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

        # Small, structured candidate library:
        #   ΔΨ_iso   : smooth additions to the isotropic baseline
        #   ΔΨ_aniso : smooth additions to the anisotropic baseline
        #   Ψ_cross  : explicit circumferential-axial coupling
        iso_terms = [
            ("I₁-3", "iso"),
            ("I₂-3", "iso"),
            ("<E_θ>", "iso"),
            ("<E_z>", "iso"),
            ("<I₂-3>ln(1+<I₂-3>)", "iso"),
        ]
        aniso_terms = [
            # Linear fiber invariants
            ("<I₄θ-1>", "aniso"),
            ("<I₄z-1>", "aniso"),
            ("<I₄D1-1>", "aniso"),
            ("<I₄D2-1>", "aniso"),
            # Quadratic fiber invariants
            ("(<I₄θ-1>)²", "aniso"),
            ("(<I₄z-1>)²", "aniso"),
            ("(<I₄D1-1>)²", "aniso"),
            ("(<I₄D2-1>)²", "aniso"),
            # Cubic fiber invariants (steep heel stiffening)
            ("(<I₄θ-1>)³", "aniso"),
            ("(<I₄z-1>)³", "aniso"),
            ("(<I₄D1-1>)³", "aniso"),
            ("(<I₄D2-1>)³", "aniso"),
            # xlog fiber invariants
            ("<I₄D1-1>ln(1+<I₄D1-1>)", "aniso"),
            ("<I₄D2-1>ln(1+<I₄D2-1>)", "aniso"),
            ("<I₄θ-1>ln(1+<I₄θ-1>)", "aniso"),
            ("<I₄z-1>ln(1+<I₄z-1>)", "aniso"),
            # Exponential fiber terms (Fung-type response discovered via library)
            ("exp(<I₄θ-1>)-1", "aniso"),
            ("exp(<I₄z-1>)-1", "aniso"),
            ("exp(<I₄D1-1>)-1", "aniso"),
            ("exp(<I₄D2-1>)-1", "aniso"),
            # Higher invariants I5 for helical families (genuinely new vs I4^2)
            ("<I₅D1-1>", "aniso"),
            ("<I₅D2-1>", "aniso"),
            ("<I₅D1-1>ln(1+<I₅D1-1>)", "aniso"),
            ("<I₅D2-1>ln(1+<I₅D2-1>)", "aniso"),
        ]
        cross_terms = [
            # Matrix × fiber
            ("<I₄θ-1>·<E_θ>", "cross"),
            ("<I₄z-1>·<E_z>", "cross"),
            ("<E_θ>·<E_z>", "cross"),
            ("<I₁-3>·<E_θ>", "cross"),
            ("<I₁-3>·<E_z>", "cross"),
            ("<I₂-3>·<E_θ>", "cross"),
            ("<I₂-3>·<E_z>", "cross"),
            ("<I₄D1-1>·<E_θ>", "cross"),
            ("<I₄D2-1>·<E_θ>", "cross"),
            # Fiber–fiber coupling
            ("<I₄θ-1>·<I₄z-1>", "cross"),      # circumferential × axial
            ("<I₄D1-1>·<I₄D2-1>", "cross"),     # helical family interaction
            ("<I₄θ-1>·<I₄D1-1>", "cross"),      # circumferential × helical
            ("<I₄z-1>·<I₄D2-1>", "cross"),      # axial × helical
            # I8: signed circ-vs-axial loading difference (new invariant)
            ("I₈D1D2", "cross"),
            # I5 × strain coupling
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

def select_active_by_bucket(
    scores,
    bucket_map,
    *,
    max_iso: int = 2,
    max_aniso: int = 2,
    max_cross: int = 3,
    thr: float = 0.5,
    force_bucket_fill: bool = False,
):
    caps = {"iso": max_iso, "aniso": max_aniso, "cross": max_cross}
    scores = np.asarray(scores, dtype=float)
    active = []
    for bucket in ("iso", "aniso", "cross"):
        idxs = [i for i, b in enumerate(bucket_map) if b == bucket]
        if not idxs:
            continue
        passing = [i for i in idxs if scores[i] > thr]
        ranked_source = passing if passing else (idxs if force_bucket_fill else [])
        ranked = sorted(ranked_source, key=lambda i: scores[i], reverse=True)
        active.extend(ranked[:caps[bucket]])
    return sorted(set(active))

def report_active_gated_indices(model, eff_vals=None, weight_thresh: float = 0.01):
    """Use selected gated terms, falling back to significant effective weights for reporting."""
    selected = list(getattr(model, "selected_active_idx", []))
    if selected:
        return selected, False
    if eff_vals is None and hasattr(model, "tweak"):
        eff_vals = (model.tweak.gates() * model.tweak.w).detach().cpu().numpy()
    if eff_vals is None:
        return [], False
    active = np.where(np.abs(np.asarray(eff_vals, dtype=float)) > weight_thresh)[0].tolist()
    return active, bool(active)

class SimpleSEDF(SymbolicCrossMixin, nn.Module):
    """
    Simple SEDF model: base mixture (isotropic/anisotropic) + masked linear tweak.
    Designed for fast sweeps over different basis feature subsets.
    """
    def __init__(self, iso_model="poly", aniso_model="4fiber", use_recruitment=True, use_dispersion=True,
                 n_basis=45, active_idx=None):
        super().__init__()
        self.base = DecomposedSEDFBase(
            iso_model=iso_model,
            aniso_model=aniso_model,
            use_recruitment=use_recruitment,
            use_dispersion=use_dispersion,
        )
        self.n_basis = n_basis
        self.init_symbolic_layers()

        # Default to all features if active_idx not specified
        if active_idx is None:
            active_idx = list(range(n_basis))
        self.tweak = TweakLinear(n_basis, active_idx)
        
        # Statistics buffers for basis feature normalization
        self.register_buffer('b_mu', torch.zeros(n_basis, dtype=torch.float64))
        self.register_buffer('b_sigma', torch.ones(n_basis, dtype=torch.float64))
    
    def basis_feats(self, lam):
        """
        Compute normalized basis features.
        """
        feats = self.basis_feats_raw(lam)
        return (feats - self.b_mu) / self.b_sigma
    
    @torch.no_grad()
    def set_stats(self, lam_all: Optional[torch.Tensor]):
        """Set normalization statistics for basis features."""
        if lam_all is None or lam_all.numel() == 0:
            self.b_mu[:] = 0.0
            self.b_sigma[:] = 1.0
            return
        feats = self.basis_feats_raw(lam_all)
        self.b_mu.copy_(feats.mean(0))
        self.b_sigma.copy_(feats.std(0).clamp_min(1e-6))
    
    def forward_parts(self, lam, branch: str = "theta"):
        """Return slide-21 framework parts for inspection."""
        W_base, W_iso, W_aniso = self.base.energy_parts(lam, branch=branch)
        z = self.basis_feats(lam)
        W_cross = self.tweak(z)
        inv, h0_feats, op_feats, act_feats = self.symbolic_layer_outputs(lam)
        return W_base, W_cross, W_iso, W_aniso, inv, h0_feats, op_feats, act_feats
    
    def forward(self, lam, create_graph=True, branch: str = "theta"):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        
        W_mix, *_ = self.base.energy_parts(lam, branch=branch)
        z = self.basis_feats(lam)
        W_cross = self.tweak(z)
        W = W_mix + W_cross
        
        σθ, σz = cauchy_from_W(W, lam, create_graph=create_graph)
        return W, σθ, σz

class GatedSEDF(SymbolicCrossMixin, nn.Module):
    """
    CompositeSEDFBase + gated symbolic cross term
    """
    def __init__(self, iso_model="poly", aniso_model="4fiber", use_recruitment=True, use_dispersion=True, n_basis=45,
                 init_gate=0.2, gate_temp=1.0, dropout_p: float = 0.0):
        super().__init__()
        self.base = DecomposedSEDFBase(
            iso_model=iso_model,
            aniso_model=aniso_model,
            use_recruitment=use_recruitment,
            use_dispersion=use_dispersion,
        )
        self.n_basis = n_basis
        self.init_symbolic_layers()

        self.tweak = GatedTweakLinear(n_basis=n_basis, init_gate=init_gate, temp=gate_temp, dropout_p=dropout_p)
        
        self.register_buffer('b_mu', torch.zeros(n_basis, dtype=torch.float64))
        self.register_buffer('b_sigma', torch.ones(n_basis, dtype=torch.float64))
    
    def basis_feats(self, lam):
        feats = self.basis_feats_raw(lam)
        return (feats - self.b_mu) / self.b_sigma
    
    @torch.no_grad()
    def set_stats(self, lam_all):
        if lam_all is None or lam_all.numel() == 0:
            self.b_mu[:] = 0.0
            self.b_sigma[:] = 1.0
            return
        feats = self.basis_feats_raw(lam_all)
        self.b_mu.copy_(feats.mean(0))
        self.b_sigma.copy_(feats.std(0).clamp_min(1e-6))
    
    def forward(self, lam, create_graph=True, branch: str = "theta"):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        
        W_mix, *_ = self.base.energy_parts(lam, branch=branch)
        z = self.basis_feats(lam)
        W_cross = self.tweak(z)
        W = W_mix + W_cross
        
        σθ, σz = cauchy_from_W(W, lam, create_graph=create_graph)
        return W, σθ, σz

class SigmaResidualHead(nn.Module):
    def __init__(self, in_dim=5, hidden=12, scale=0.12, cap=0.9):
        super().__init__()
        self.cap = cap
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2, bias=True)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1e-3, 1e-3); nn.init.zeros_(m.bias)
        self.scale = scale
    def forward(self, feats):
        d = self.scale * self.net(feats)
        return torch.clamp(d, -self.cap, self.cap)

class ICNNResidual(nn.Module):
    def __init__(self, in_dim=5, hidden=32, n_layers=3, scale=0.08, cap=0.6):
        super().__init__()
        self.scale, self.cap = scale, cap
        self.n_layers = n_layers
        
        # First layer
        self.W1 = nn.Parameter(torch.randn(hidden, in_dim) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        
        # Hidden layers - store as ParameterList, not ModuleList
        self.W_hidden = nn.ParameterList([
            nn.Parameter(torch.randn(hidden, hidden) * 0.05) 
            for _ in range(n_layers - 1)
        ])
        self.W_skip = nn.ParameterList([
            nn.Parameter(torch.randn(hidden, in_dim) * 0.05)
            for _ in range(n_layers - 1)
        ])
        self.b_hidden = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden))
            for _ in range(n_layers - 1)
        ])
        
        # Output layer
        self.w_out = nn.Parameter(torch.randn(1, hidden) * 0.05)
        self.b_out = nn.Parameter(torch.zeros(1))
        
    def forward(self, feats):
        # Ensure non-negativity of passthrough weights for convexity
        W1_pos = torch.relu(self.W1)
        
        # First activation
        z = torch.nn.functional.softplus(feats @ W1_pos.T + self.b1)
        
        # Subsequent layers with skip connections from input
        for W, W_skip, b in zip(self.W_hidden, self.W_skip, self.b_hidden):
            W_pos = torch.relu(W)  # Positive weights for convexity
            # Skip connection can have any sign
            z = torch.nn.functional.softplus(z @ W_pos.T + feats @ W_skip.T + b)
        
        # Output
        raw = (z @ torch.relu(self.w_out).T + self.b_out).squeeze(-1)
        dPsi = self.scale * torch.tanh(raw / self.scale)  # Better scaling
        return torch.clamp(dPsi, -self.cap, self.cap)
    
class AnchoredICNNMoE(nn.Module):
    """
    Wrap a base (Eq9/Eq9_Gent/PoleZero) with:
      - gains γ_iso, γ_fib
      - tiny physics basis head
      - anchored ΔΨ from 2 ICNN experts via logistic gate (optional)
      - tiny σ residual head
    """
    def __init__(self, base_cls, hidden_icnn=16, n_layers=2, n_basis=18, scale=6e-4, cap=3e-3, use_gate=True,
                 poly_gate_center: float | None = None,
                 poly_gate_width: float | None = None):
        super().__init__()
        self.base = base_cls()
        self.n_basis = n_basis  # STORE THIS
        self.use_gate = use_gate  # honor constructor flag
        self._w_gain_iso = nn.Parameter(torch.tensor(0.45))
        self._w_gain_fib = nn.Parameter(torch.tensor(0.45))
         # λθ0 ~ where extra stiffening kicks in
        self._gate_lamth0 = nn.Parameter(torch.tensor(1.15))  # initial guess
        # sharpness k > 0 controlling transition width
        self._gate_sharp  = nn.Parameter(torch.tensor(8.0))   # moderate slope

        self.sigma_res = SigmaResidualHead(in_dim=5, hidden=12, scale=0.12, cap=0.9)
        self.use_sigma_residual = True
        self._gain_eps   = 1e-8
        # Gated polynomial heads (tiny)
        self.poly_head_low  = nn.Linear(n_basis, 1, bias=False)
        self.poly_head_high = nn.Linear(n_basis, 1, bias=False)
        nn.init.zeros_(self.poly_head_low.weight); nn.init.zeros_(self.poly_head_high.weight)
        # Keep legacy single head for compatibility with older reporting utils (may be unused)
        self.extra_head  = nn.Linear(n_basis, 1, bias=False)
        nn.init.zeros_(self.extra_head.weight)
        
        if use_gate:
            # Gated ICNN: two networks with gate
            self.res_low  = ICNNResidual(in_dim=5, hidden=hidden_icnn, n_layers=n_layers, scale=scale, cap=cap)
            self.res_high = ICNNResidual(in_dim=5, hidden=hidden_icnn, n_layers=n_layers, scale=scale, cap=cap)
            
            self.gate_lin = nn.Linear(2, 1, bias=True)
            self._gate_temp = nn.Parameter(torch.tensor(10.0))
            with torch.no_grad():
                self.gate_lin.weight[:] = torch.tensor([[3.0, 0.6]], dtype=torch.float64)
                self.gate_lin.bias[:]   = torch.tensor([-5.0], dtype=torch.float64)
        else:
        #     # Single ICNN: no gating for physical equations
            self.res_single = ICNNResidual(in_dim=5, hidden=hidden_icnn, n_layers=n_layers, scale=scale, cap=cap)
            self.hoop_icnn = ICNNResidual(in_dim=1, hidden=8, n_layers=2, scale=5e-3, cap=0.15)
        self.register_buffer("f_mu",    torch.zeros(5, dtype=torch.float64))
        self.register_buffer("f_sigma", torch.ones( 5, dtype=torch.float64))
        self.register_buffer("b_mu",    torch.zeros(n_basis, dtype=torch.float64))
        self.register_buffer("b_sigma", torch.ones(  n_basis, dtype=torch.float64))
        # Optional initialization of tiny polynomial gate parameters
        if poly_gate_center is not None:
            self._poly_gate_center = nn.Parameter(torch.tensor(float(poly_gate_center), dtype=torch.float64))
        if poly_gate_width is not None:
            self._poly_gate_width = nn.Parameter(torch.tensor(float(poly_gate_width), dtype=torch.float64))
        self.register_buffer("lamth_mu",    torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("lamth_sigma", torch.tensor(0.1, dtype=torch.float64))
    # gains
    def gain_iso(self): return 1.0 + 0.5*torch.tanh(self._w_gain_iso)
    def gain_fib(self): return 1.0 + 0.5*torch.tanh(self._w_gain_fib)

    # features
    def _feats(self, lam):
        C = C_from_lambdas(lam)
        dirs = self.base.dirs()
        feats = torch.stack([I1(C), I4(C, dirs[0]), I4(C, dirs[-2]), I4(C, dirs[-1]), I4(C, dirs[1])], dim=-1)
        return (feats - self.f_mu) / self.f_sigma
    
    def basis_feats(self, lam):
        C  = C_from_lambdas(lam)
        E_th = 0.5 * (lam[...,0]**2 - 1.0)
        E_z  = 0.5 * (lam[...,1]**2 - 1.0)
        I1v  = I1(C) - 3.0
        dirs = self.base.dirs()
        I4h  = I4(C, dirs[0]) - 1.0
        I4t  = I4(C, dirs[-2]) - 1.0
        I4z  = I4(C, dirs[-1]) - 1.0
        I4hE2 = I4h * (E_th**2)
        I4zE  = I4z * E_th      # or E_z, depending on what you want
        I4zEz = I4z * E_z
        Ezth2 = (E_th * E_z) ** 2
        # Safe nonlinear transforms
        expE_th = expm1_clamped(2.0 * E_th)  # bounded expm1
        expE_z  = expm1_clamped(2.0 * E_z)
        E_th_p  = smooth_pos(E_th, 1e-3)
        E_z_p   = smooth_pos(E_z,  1e-3)
        I1v_p   = smooth_pos(I1v,  1e-3)
        lnE_th  = torch.log1p(E_th_p)
        lnE_z   = torch.log1p(E_z_p)
        lnI1    = torch.log1p(I1v_p)

        all_feats = [
            E_th**2,
            E_z**2,
            E_th*E_z,
            E_th**3,
            E_z**3,
            I1v,
            I4h**2,
            I4t**2,
            I4z**2,
            I4h*E_th,
            I4hE2,    # new
            I4zE,     # new
            I4zEz,    # new
            Ezth2,    # new
            expE_th,  # exp
            expE_z,   # exp
            lnE_th,   # ln
            lnE_z,    # ln
            lnI1      # ln
        ]

        
        # Select only n_basis features (take the first n_basis)
        feats = torch.stack(all_feats[:self.n_basis], dim=-1)
        return (feats - self.b_mu) / self.b_sigma
    def _hoop_gate(self, lam_th: torch.Tensor) -> torch.Tensor:
        """
        Smooth gate g(λθ) ≈ 0 at low stretch, ≈1 at high stretch.
        Physically: onset of an extra stiffening mechanism.
        """
        k  = softplus_pos(self._gate_sharp)         # k > 0
        l0 = self._gate_lamth0                      # threshold location
        return torch.sigmoid(k * (lam_th - l0))

    @torch.no_grad()
    def set_stats(self, lam_all: Optional[torch.Tensor]):
        if lam_all is None or lam_all.numel() == 0:
            self.f_mu[:] = 0.0; self.f_sigma[:] = 1.0
            self.b_mu[:] = 0.0; self.b_sigma[:] = 1.0
            self.lamth_mu[:] = 1.0; self.lamth_sigma[:] = 0.1
            return
        F = self._feats(lam_all); self.f_mu.copy_(F.mean(0)); self.f_sigma.copy_(F.std(0).clamp_min(1e-6))
        B = self.basis_feats(lam_all); self.b_mu.copy_(B.mean(0)); self.b_sigma.copy_(B.std(0).clamp_min(1e-6))
        lam_th_all = lam_all[..., 0]
        self.lamth_mu[...] = lam_th_all.mean()
        self.lamth_sigma[...] = lam_th_all.std().clamp_min(1e-6)
        
    def residual_energy(self, lam: torch.Tensor, detach_ref: bool = True) -> torch.Tensor:
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)

        feats = self._feats(lam)

        if self.use_gate:
            lamθ = lam[..., 0:1]
            lamz = lam[..., 1:2]
            inp = torch.cat([lamθ, lamz], -1)
            g_logits = self.gate_lin(inp)
            g = torch.sigmoid(self._gate_temp * g_logits)

            dPsi_low  = self.res_low(feats)
            dPsi_high = self.res_high(feats)
            dPsi = (1.0 - g) * dPsi_low + g * dPsi_high
        else:
            dPsi = self.res_single(feats)

        return dPsi

    def basis_energy(self, lam):
        z = self.basis_feats(lam)
        # Simple gate on max principal stretch (θ or z)
        lam_max, _ = lam[..., :2].max(dim=-1)
        # Use a gentle width to avoid discontinuities
        center = getattr(self, "_poly_gate_center", None)
        width_p = getattr(self, "_poly_gate_width", None)
        if center is None:
            self._poly_gate_center = nn.Parameter(torch.tensor(1.85, dtype=torch.float64))
            center = self._poly_gate_center
        if width_p is None:
            self._poly_gate_width = nn.Parameter(torch.tensor(0.10, dtype=torch.float64))
            width_p = self._poly_gate_width
        width = softplus_pos(width_p) + 1e-6
        g = torch.sigmoid((lam_max - center) / width)
        Wlow  = self.poly_head_low(z).squeeze(-1)
        Whigh = self.poly_head_high(z).squeeze(-1)
        return (1.0 - g) * Wlow + g * Whigh

    def forward(self, lam, create_graph=True):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)

        q_iso, q_fib = self.base.energy_parts(lam)
        W_base = self.gain_iso() * q_iso + self.gain_fib() * q_fib

        Wtot = W_base
        Wtot = Wtot + self.basis_energy(lam)        # polynomial correction
        Wtot = Wtot + self.residual_energy(lam)     # ICNN ΔΨ

        σθ, σz = cauchy_from_W(Wtot, lam, create_graph)

        # keep stress residual OFF for now
        # if self.use_sigma_residual:
        #     feats = self._feats(lam)
        #     dσ = self.sigma_res(feats)
        #     σθ = σθ + dσ[..., 0]; σz = σz + dσ[..., 1]

        return Wtot, σθ, σz


# -------------------- Polynomial-only wrapper (no ICNN residual) --------------------
class AnchoredPolyOnly(nn.Module):
    """
    Wrap a base (Eq9/Eq9_Gent/PoleZero) with:
      - gains γ_iso, γ_fib
      - tiny physics polynomial head with smooth low/high gate
    No ICNN residual is built or used.
    """
    def __init__(self, base_cls, n_basis=18,
                 use_poly_gate: bool = True,
                 poly_gate_center: float | None = None,
                 poly_gate_width: float | None = None):
        super().__init__()
        self.base = base_cls()
        self.n_basis = n_basis
        self.use_poly_gate = use_poly_gate
        self._w_gain_iso = nn.Parameter(torch.tensor(0.45))
        self._w_gain_fib = nn.Parameter(torch.tensor(0.45))
        # Gated polynomial heads
        self.poly_head_low  = nn.Linear(n_basis, 1, bias=False)
        self.poly_head_high = nn.Linear(n_basis, 1, bias=False)
        nn.init.zeros_(self.poly_head_low.weight); nn.init.zeros_(self.poly_head_high.weight)
        # Legacy single head retained for reporting utils
        self.extra_head  = nn.Linear(n_basis, 1, bias=False)
        nn.init.zeros_(self.extra_head.weight)
        # Stats for normalization
        self.register_buffer("b_mu",    torch.zeros(n_basis, dtype=torch.float64))
        self.register_buffer("b_sigma", torch.ones(  n_basis, dtype=torch.float64))
        # Optional initialization for polynomial gate parameters
        if poly_gate_center is not None:
            self._poly_gate_center = nn.Parameter(torch.tensor(float(poly_gate_center), dtype=torch.float64))
        if poly_gate_width is not None:
            self._poly_gate_width = nn.Parameter(torch.tensor(float(poly_gate_width), dtype=torch.float64))

    # gains
    def gain_iso(self): return 1.0 + 0.5*torch.tanh(self._w_gain_iso)
    def gain_fib(self): return 1.0 + 0.5*torch.tanh(self._w_gain_fib)

    def basis_feats(self, lam):
        C  = C_from_lambdas(lam)
        E_th = 0.5 * (lam[...,0]**2 - 1.0)
        E_z  = 0.5 * (lam[...,1]**2 - 1.0)
        I1v  = I1(C) - 3.0
        dirs = self.base.dirs()
        I4h  = I4(C, dirs[0]) - 1.0
        I4t  = I4(C, dirs[-2]) - 1.0
        I4z  = I4(C, dirs[-1]) - 1.0
        I4hE2 = I4h * (E_th**2)
        I4zE  = I4z * E_th
        I4zEz = I4z * E_z
        Ezth2 = (E_th * E_z) ** 2
        expE_th = expm1_clamped(2.0 * E_th)
        expE_z  = expm1_clamped(2.0 * E_z)
        E_th_p  = smooth_pos(E_th, 1e-3)
        E_z_p   = smooth_pos(E_z,  1e-3)
        I1v_p   = smooth_pos(I1v,  1e-3)
        lnE_th  = torch.log1p(E_th_p)
        lnE_z   = torch.log1p(E_z_p)
        lnI1    = torch.log1p(I1v_p)
        all_feats = [
            E_th**2, E_z**2, E_th*E_z, E_th**3, E_z**3,
            I1v, I4h**2, I4t**2, I4z**2,
            I4h*E_th, I4hE2, I4zE, I4zEz, Ezth2,
            expE_th, expE_z, lnE_th, lnE_z, lnI1
        ]
        feats = torch.stack(all_feats[:self.n_basis], dim=-1)
        return (feats - self.b_mu) / self.b_sigma

    @torch.no_grad()
    def set_stats(self, lam_all: Optional[torch.Tensor]):
        if lam_all is None or lam_all.numel() == 0:
            self.b_mu[:] = 0.0; self.b_sigma[:] = 1.0
            return
        B = self.basis_feats(lam_all)
        self.b_mu.copy_(B.mean(0))
        self.b_sigma.copy_(B.std(0).clamp_min(1e-6))

    def basis_energy(self, lam):
        z = self.basis_feats(lam)
        # Optionally use a smooth gate on max principal stretch (θ or z)
        if not getattr(self, "use_poly_gate", True):
            return self.extra_head(z).squeeze(-1)
        lam_max, _ = lam[..., :2].max(dim=-1)
        center = getattr(self, "_poly_gate_center", None)
        width_p = getattr(self, "_poly_gate_width", None)
        if center is None:
            self._poly_gate_center = nn.Parameter(torch.tensor(1.85, dtype=torch.float64))
            center = self._poly_gate_center
        if width_p is None:
            self._poly_gate_width = nn.Parameter(torch.tensor(0.10, dtype=torch.float64))
            width_p = self._poly_gate_width
        width = softplus_pos(width_p) + 1e-6
        g = torch.sigmoid((lam_max - center) / width)
        Wlow  = self.poly_head_low(z).squeeze(-1)
        Whigh = self.poly_head_high(z).squeeze(-1)
        return (1.0 - g) * Wlow + g * Whigh

    def forward(self, lam, create_graph=True):
        if not lam.requires_grad:
            lam = lam.requires_grad_(True)
        q_iso, q_fib = self.base.energy_parts(lam)
        W_base = self.gain_iso() * q_iso + self.gain_fib() * q_fib
        Wtot = W_base + self.basis_energy(lam)
        σθ, σz = cauchy_from_W(Wtot, lam, create_graph)
        return Wtot, σθ, σz
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

def _split_params_for_tweaked(model):
    """
    Split parameters for models with base + residual/extra components.
    Handles both CompositeWithTweak (base + tweak) and legacy ICNN models.
    """
    names, params = zip(*list(model.named_parameters()))
    base_params  = []
    resid_params = []
    extra_params = []
    for n,p in zip(names, params):
        if n.startswith("base."): 
            base_params.append(p)
        elif n.startswith("tweak."):
            # CompositeWithTweak: treat tweak as extra_params
            extra_params.append(p)
        elif n.startswith("extra_head.") or n.startswith("_w_gain") or n.startswith("poly_head_low") or n.startswith("poly_head_high"): 
            extra_params.append(p)
        else: 
            resid_params.append(p)
    return base_params, resid_params, extra_params

def _split_params_for_polyonly(model):
    """
    Split parameters for polynomial-only models (AnchoredPolyOnly).
    Also handles CompositeWithTweak by treating tweak as extra.
    """
    names, params = zip(*list(model.named_parameters()))
    base_params  = []
    extra_params = []
    for n,p in zip(names, params):
        if n.startswith("base."):
            base_params.append(p)
        elif n.startswith("tweak."):
            # CompositeWithTweak: treat tweak as extra_params
            extra_params.append(p)
        elif (n.startswith("extra_head.") or n.startswith("_w_gain")
              or n.startswith("poly_head_low") or n.startswith("poly_head_high")):
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
        
        # Model with this subset
        m = SimpleSEDF(
            elastin_type=elastin_type,
            use_recruitment=True,
            n_basis=n_basis,
            active_idx=list(idxs)
        ).to(device)
        
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

    # Split into train/val for ICNN training only
    use_validation = isinstance(model, AnchoredICNNMoE) and val_split > 0
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
    if isinstance(model, AnchoredICNNMoE):
        base_params, resid_params, extra_params = _split_params_for_tweaked(model)
        for p in base_params: p.requires_grad_(lr_base > 0.0)
        if lr_resid <= 0.0: lr_resid = lr
        groups = []
        if lr_base>0: groups.append({"params": base_params, "lr": lr_base, "weight_decay": 1e-6})
        groups.append({"params": resid_params, "lr": lr_resid, "weight_decay": 2e-5})
        groups.append({"params": extra_params, "lr": lr_resid, "weight_decay": 2e-5})
        opt = torch.optim.Adam(groups)
        max_clip = 1.0
    elif isinstance(model, AnchoredPolyOnly):
        base_params, extra_params = _split_params_for_polyonly(model)
        for p in base_params: p.requires_grad_(lr_base > 0.0)
        groups = []
        if lr_base>0: groups.append({"params": base_params, "lr": lr_base, "weight_decay": 1e-6})
        lr_poly = lr_resid if lr_resid > 0.0 else lr
        groups.append({"params": extra_params, "lr": lr_poly, "weight_decay": 2e-5})
        opt = torch.optim.Adam(groups)
        max_clip = 5.0
    elif is_composite:
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
        
        # Optional: anneal gate temperature (for GatedTweakLinear)
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

        # Enhanced regularization for ICNN
        if isinstance(model, AnchoredICNNMoE):
            lam_pool = []
            if lam_th_train is not None and lam_th_train.numel()>0: lam_pool.append(lam_th_train)
            if lam_z_train is not None and lam_z_train.numel()>0: lam_pool.append(lam_z_train)
            
            if lam_pool:
                lam_all = torch.cat(lam_pool, 0)
                idx = torch.randint(0, lam_all.shape[0], (min(256, lam_all.shape[0]),), device=device)
                lam_reg = lam_all[idx].detach().clone().requires_grad_(True)
                
                # 1. Energy magnitude regularization (STRONGER)
                dPsi_reg = model.residual_energy(lam_reg, detach_ref=True)
                loss = loss + 1e-3 * dPsi_reg.pow(2).mean()
                
                # 2. Gradient regularization (STRONGER)
                g = torch.autograd.grad(dPsi_reg.sum(), lam_reg, create_graph=True, retain_graph=True, allow_unused=True)[0]
                if g is not None:
                    loss = loss + 2e-4 * g.pow(2).sum(-1).mean()
                    
                    # 3. Second derivative regularization (smoothness)
                    # hess_diag = []
                    # for i in range(min(lam_reg.shape[-1],2)):
                    #     h = torch.autograd.grad(g[:, i].sum(), lam_reg, create_graph= True, retain_graph=True, allow_unused=True)[0]
                    #     if h is not None:
                    #         hess_diag.append(h[:, i])
                    # if hess_diag:
                    #     hess_diag = torch.stack(hess_diag, -1)
                    #     loss = loss + 5e-3 * hess_diag.pow(2).mean()
            
            # Regularize gains and basis
            loss = loss + reg_gain*((model.gain_iso()-1.0).pow(2) + (model.gain_fib()-1.0).pow(2))
            loss = loss + reg_basis*(model.extra_head.weight.pow(2).sum())
            
            # Stronger L2 on all ICNN weights
            l2_icnn = 0.0
            for name_p, param in model.named_parameters():
                if 'res_low' in name_p or 'res_high' in name_p or 'res_single' in name_p:
                    l2_icnn = l2_icnn + param.pow(2).sum()
            loss = loss + 1e-5 * l2_icnn
            
            # L2 on stress residual head
            l2_sigma = 0.0
            for m in model.sigma_res.net.modules():
                if isinstance(m, nn.Linear):
                    l2_sigma = l2_sigma + m.weight.pow(2).mean()
            loss = loss + 3e-4 * l2_sigma
        
        # Gate sparsity penalty (for GatedSEDF / GatedTweakLinear)
        if gate_lambda > 0.0 and hasattr(model, "tweak") and hasattr(model.tweak, "l1_gate_penalty"):
            loss = loss + gate_lambda * model.tweak.l1_gate_penalty()
        
        # Anchor regularization: penalize moving away from reference parameters
        if anchor_lambda > 0.0 and anchor_params is not None:
            for name, p in model.named_parameters():
                if name in anchor_params:
                    loss = loss + anchor_lambda * (p - anchor_params[name]).pow(2).mean()
        
        # Penalty method: only apply squaring to ICNN models (composite uses standard LSQ)
        if isinstance(model, AnchoredICNNMoE):
            loss = (loss)**2 * 55.0  # penalty method for ICNN
        else:
            loss = 55.0 * loss  # simple scaling for composite/legacy models (MATLAB-like)
        
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

def print_final_equation(model, base_type):
    """Print the final equation form with significant weights only"""
    print(f"\nBase Model: {base_type}")
    print("="*50)
    
    # Get gains
    gain_iso = model.gain_iso().item()
    gain_fib = model.gain_fib().item()
    print(f"Gains: γ_iso = {gain_iso:.3f}, γ_fib = {gain_fib:.3f}")
    
    # Get polynomial basis weights
    use_pg = getattr(model, "use_poly_gate", True)
    if use_pg:
        w_low  = model.poly_head_low.weight.detach().cpu().numpy().flatten()
        w_high = model.poly_head_high.weight.detach().cpu().numpy().flatten()
    else:
        w = model.extra_head.weight.detach().cpu().numpy().flatten()
    print(f"\nPolynomial Basis Terms (|weight| > 0.01):")
    print("-" * 40)
    
    # Define basis term names
    basis_names = [
        "E_θ²",           # 0: E_th**2
        "E_z²",           # 1: E_z**2  
        "E_θ·E_z",        # 2: E_th*E_z
        "E_θ³",           # 3: E_th**3
        "E_z³",           # 4: E_z**3
        "I₁-3",           # 5: I1v
        "I₄ₕ²",           # 6: I4h**2
        "I₄ₜ²",           # 7: I4t**2
        "I₄ᵧ²",           # 8: I4z**2
        "I₄ₕ·E_θ",        # 9: I4h*E_th
        "I₄ₕ·E_θ²",       # 10  (I4hE2)
        "I₄_z·E_θ",       # 11  (I4zE)
        "I₄_z·E_z",       # 12  (I4zEz)
        "(E_θ·E_z)²",     # 13  (Ezth2)
        "exp(E_θ)-1",     # 14
        "exp(E_z)-1",     # 15
        "ln(1+E_θ)",      # 16
        "ln(1+E_z)",      # 17
        "ln(1+I₁-3)"      # 18
    ]
    
    if use_pg:
        print(" Low-strain head:")
        any_low = False
        for i, (name, weight) in enumerate(zip(basis_names, w_low)):
            if abs(weight) > 0.01:
                any_low = True
                print(f"  {name:12s}: {weight:8.3f}")
        if not any_low:
            print("  (no |weight|>0.01)")
        
        print(" High-strain head:")
        any_high = False
        for i, (name, weight) in enumerate(zip(basis_names, w_high)):
            if abs(weight) > 0.01:
                any_high = True
                print(f"  {name:12s}: {weight:8.3f}")
        if not any_high:
            print("  (no |weight|>0.01)")
    else:
        print(" Single head:")
        any_any = False
        for i, (name, weight) in enumerate(zip(basis_names, w)):
            if abs(weight) > 0.01:
                any_any = True
                print(f"  {name:12s}: {weight:8.3f}")
        if not any_any:
            print("  (no |weight|>0.01)")
    
    # Final equation form (polynomial-only residual in this script)
    print(f"\nFINAL EQUATION FORM:")
    print("="*50)
    print(f"W_total = γ_iso·W_iso + γ_fib·W_fib + W_poly")
    print(f"where:")
    print(f"  γ_iso = {gain_iso:.3f}")
    print(f"  γ_fib = {gain_fib:.3f}")
    print(f"  W_iso = {base_type} isotropic energy")
    recruit_on = getattr(getattr(model, "base", None), "use_recruitment", False)
    fib_desc = "fiber energy with recruitment" if recruit_on else "fiber energy (no recruitment)"
    print(f"  W_fib = {fib_desc}")
    if use_pg:
        print(f"  W_poly = (1-g)·(w_low·basis) + g·(w_high·basis)  [g is a smooth gate]")
    else:
        print(f"  W_poly = w·basis  [gate off]")
    
    # Summary
    print(f"\nSUMMARY:")
    print("-" * 30)
    print(f"• Base model: {base_type} " + ("with fiber recruitment" if recruit_on else "(no recruitment)"))
    print(f"• Energy scaling: {gain_iso:.1f}x isotropic, {gain_fib:.1f}x fiber")
    print(f"• Polynomial correction: gated low/high linear basis")
    print(f"• Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def save_equation_to_file(model, base_type, age, out_root, *, weight_thresh: float = 0.01):
    """Save the final equation to a text file"""
    os.makedirs(out_root, exist_ok=True)
    equation_file = os.path.join(out_root, f"{age}_final_equation.txt")
    
    # Check if this is a decomposed stage-1/2 model
    is_composite = is_decomposed_model(model)
    is_simple_sedf = isinstance(model, SimpleSEDF)
    is_gated_sedf = isinstance(model, GatedSEDF)
    
    with open(equation_file, "w") as f:
        f.write(f"FINAL EQUATION FOR {age} - {base_type} BASE\n")
        f.write("="*60 + "\n\n")
        
        if is_composite:
            # Decomposed model: fit isotropic and anisotropic parameters directly
            base = model.base if hasattr(model, 'base') else model
            f.write("Base Decomposition:\n")
            f.write(f"  isotropic model   = {base.iso_model}\n")
            f.write(f"  anisotropic model = {base.aniso_model}\n\n")
            
            # Tweak term parameters
            if is_simple_sedf:
                # SimpleSEDF: show selected basis indices and weights
                active_idx = model.tweak.active_idx.cpu().numpy().tolist()
                w_vals = model.tweak.w.detach().cpu().numpy()
                
                basis_names = get_basis_names(model, fallback_n=len(active_idx))
                basis_buckets = model.basis_buckets() if hasattr(model, "basis_buckets") else []
                
                f.write(f"Selected Basis Terms (from sweep):\n")
                f.write(f"  Active indices: {active_idx}\n\n")
                f.write(f"Tweak Weights:\n")
                for idx, w in zip(active_idx, w_vals):
                    name = basis_names[idx] if idx < len(basis_names) else f"term_{idx}"
                    bucket = basis_buckets[idx] if idx < len(basis_buckets) else "symbolic"
                    f.write(f"  [{bucket:5s}] {name:12s} (idx {idx:2d}): {w:10.6f}\n")
                f.write(f"\n  W_cross = Σ(w_i · ψ_i) where ψ_i are the selected basis terms above\n\n")
            elif is_gated_sedf:
                gate_vals = model.tweak.gates().detach().cpu().numpy()
                eff_vals = (model.tweak.gates() * model.tweak.w).detach().cpu().numpy()
                active_idx, used_reporting_fallback = report_active_gated_indices(
                    model, eff_vals=eff_vals, weight_thresh=weight_thresh
                )
                sigma_vals = model.b_sigma.detach().cpu().numpy() if hasattr(model, "b_sigma") else np.ones_like(eff_vals)
                basis_names = get_basis_names(model, fallback_n=len(eff_vals))
                basis_buckets = model.basis_buckets() if hasattr(model, "basis_buckets") else []

                f.write(f"Selected Basis Terms (from gated discovery):\n")
                f.write(f"  Active indices: {active_idx}\n\n")
                if not active_idx:
                    f.write("  No symbolic terms survived the activity threshold in this run.\n\n")
                elif used_reporting_fallback:
                    f.write(
                        f"  Active-term selection was empty under the training threshold; reporting terms with "
                        f"|g_i·w_i| > {weight_thresh:g} so the text equation matches the plotted weights.\n\n"
                    )
                f.write("Effective Tweak Weights:\n")
                for idx in active_idx:
                    name = basis_names[idx] if idx < len(basis_names) else f"term_{idx}"
                    bucket = basis_buckets[idx] if idx < len(basis_buckets) else "symbolic"
                    gate = gate_vals[idx] if idx < len(gate_vals) else float("nan")
                    eff = eff_vals[idx] if idx < len(eff_vals) else float("nan")
                    sigma = sigma_vals[idx] if idx < len(sigma_vals) else 1.0
                    raw_coeff = eff / max(float(sigma), 1e-12)
                    f.write(
                        f"  [{bucket:5s}] {name:20s} (idx {idx:2d}): "
                        f"gate={gate:12.4e}, eff_w_norm={eff:12.4e}, coeff_raw={raw_coeff:12.4e}, sigma={sigma:12.4e}\n"
                    )
                f.write(f"\n  W_cross = Σ((g_i·w_i) · ψ_i) over the selected basis terms above\n\n")
            elif hasattr(model, 'tweak'):
                # CompositeWithTweak: OneTermTweak
                tweak = model.tweak
                w_tweak = tweak._w.item()
                f.write(f"Cross Term:\n")
                if tweak.use_gate:
                    center = tweak._center.item()
                    width = softplus_pos(tweak._width_raw).item()
                    f.write(f"  w = {w_tweak:.6f}\n")
                    f.write(f"  gate center = {center:.6f}\n")
                    f.write(f"  gate width = {width:.6f}\n")
                    f.write(f"  W_cross = w · σ((max(λθ,λz) - center)/width) · (I₁ - 3)²\n\n")
                else:
                    f.write(f"  w = {w_tweak:.6f}\n")
                    f.write(f"  W_cross = w · (I₁ - 3)²\n\n")
        else:
            # Legacy model: use gains
            gain_iso = model.gain_iso().item()
            gain_fib = model.gain_fib().item()
            f.write(f"Energy Scaling Factors:\n")
            f.write(f"  γ_iso = {gain_iso:.6f}\n")
            f.write(f"  γ_fib = {gain_fib:.6f}\n\n")
        
        # Get polynomial basis weights (only for legacy models)
        if not is_composite:
            use_pg = getattr(model, "use_poly_gate", True)
            if use_pg:
                w_low  = model.poly_head_low.weight.detach().cpu().numpy().flatten()
                w_high = model.poly_head_high.weight.detach().cpu().numpy().flatten()
            else:
                w = model.extra_head.weight.detach().cpu().numpy().flatten()
            f.write(f"Polynomial Basis Terms (|weight| > {weight_thresh:g}):\n")
            f.write("-" * 40 + "\n")
            
            basis_names = get_basis_names(model)
            
            # List significant terms
            if use_pg:
                f.write(" Low-strain head (w_low):\n")
                any_low = False
                for name, weight in zip(basis_names, w_low):
                    if abs(weight) > weight_thresh:
                        any_low = True
                        f.write(f"  {name:12s}: {weight:10.6f}\n")
                if not any_low:
                    f.write("  (no terms above threshold)\n")
                f.write("\n High-strain head (w_high):\n")
                any_high = False
                for name, weight in zip(basis_names, w_high):
                    if abs(weight) > weight_thresh:
                        any_high = True
                        f.write(f"  {name:12s}: {weight:10.6f}\n")
                if not any_high:
                    f.write("  (no terms above threshold)\n")
            else:
                f.write(" Single head (w):\n")
                any_single = False
                for name, weight in zip(basis_names, w):
                    if abs(weight) > weight_thresh:
                        any_single = True
                        f.write(f"  {name:12s}: {weight:10.6f}\n")
                if not any_single:
                    f.write("  (no terms above threshold)\n")
        
        # Final equation
        f.write(f"\nFINAL EQUATION:\n")
        f.write("="*30 + "\n")
        
        if is_composite:
            base = model.base if hasattr(model, 'base') else model
            f.write(f"W_total = W_base + W_cross\n\n")
            
            f.write(f"where:\n")
            f.write(f"  W_base = W_iso + W_aniso\n")
            f.write(f"  isotropic model   = {base.iso_model}\n")
            f.write(f"  anisotropic model = {base.aniso_model}\n")
            recruit_on = getattr(base.aniso, "use_recruitment", False)
            aniso_desc = "recruiting fiber-directional energy" if recruit_on else "fiber-directional energy"
            f.write(f"  W_aniso = {aniso_desc}\n")
            if is_simple_sedf:
                active_idx = model.tweak.active_idx.cpu().numpy().tolist()
                basis_names = get_basis_names(model, fallback_n=len(active_idx))
                basis_buckets = model.basis_buckets() if hasattr(model, "basis_buckets") else []
                term_list = ", ".join([basis_names[i] if i < len(basis_names) else f"term_{i}" for i in active_idx])
                iso_terms = [basis_names[i] for i in active_idx if i < len(basis_buckets) and basis_buckets[i] == "iso"]
                aniso_terms = [basis_names[i] for i in active_idx if i < len(basis_buckets) and basis_buckets[i] == "aniso"]
                cross_terms = [basis_names[i] for i in active_idx if i < len(basis_buckets) and basis_buckets[i] == "cross"]
                f.write(f"  ΔW_iso   uses ψ_i ∈ {{{', '.join(iso_terms) if iso_terms else '∅'}}}\n")
                f.write(f"  ΔW_aniso uses ψ_i ∈ {{{', '.join(aniso_terms) if aniso_terms else '∅'}}}\n")
                f.write(f"  W_cross  uses ψ_i ∈ {{{', '.join(cross_terms) if cross_terms else '∅'}}}\n")
                f.write(f"  Combined symbolic terms: {{{term_list}}}\n")
            elif is_gated_sedf:
                eff_vals = (model.tweak.gates() * model.tweak.w).detach().cpu().numpy()
                active_idx, _ = report_active_gated_indices(
                    model, eff_vals=eff_vals, weight_thresh=weight_thresh
                )
                basis_names = get_basis_names(model, fallback_n=len(eff_vals))
                basis_buckets = model.basis_buckets() if hasattr(model, "basis_buckets") else []
                term_list = ", ".join([basis_names[i] if i < len(basis_names) else f"term_{i}" for i in active_idx])
                iso_terms = [basis_names[i] for i in active_idx if i < len(basis_buckets) and basis_buckets[i] == "iso"]
                aniso_terms = [basis_names[i] for i in active_idx if i < len(basis_buckets) and basis_buckets[i] == "aniso"]
                cross_terms = [basis_names[i] for i in active_idx if i < len(basis_buckets) and basis_buckets[i] == "cross"]
                f.write(f"  ΔW_iso   uses ψ_i ∈ {{{', '.join(iso_terms) if iso_terms else '∅'}}}\n")
                f.write(f"  ΔW_aniso uses ψ_i ∈ {{{', '.join(aniso_terms) if aniso_terms else '∅'}}}\n")
                f.write(f"  W_cross  uses ψ_i ∈ {{{', '.join(cross_terms) if cross_terms else '∅'}}}\n")
                f.write(f"  Combined symbolic terms: {{{term_list}}}\n")
            elif hasattr(model, 'tweak'):
                if hasattr(model.tweak, 'use_gate') and model.tweak.use_gate:
                    f.write(f"  W_cross = w · σ((max(λθ,λz) - center)/width) · (I₁ - 3)²\n")
                else:
                    f.write(f"  W_cross = w · (I₁ - 3)²\n")
            else:
                f.write(f"  W_cross = 0  (no cross term)\n")
        else:
            f.write(f"W_total = γ_iso·W_iso + γ_fib·W_fib + W_poly\n\n")
            
            f.write(f"where:\n")
            f.write(f"  γ_iso = {gain_iso:.6f}\n")
            f.write(f"  γ_fib = {gain_fib:.6f}\n")
            f.write(f"  W_iso = {base_type} isotropic energy\n")
            recruit_on = getattr(getattr(model, "base", None), "use_recruitment", False)
            fib_desc = "fiber energy with recruitment" if recruit_on else "fiber energy (no recruitment)"
            f.write(f"  W_fib = {fib_desc}\n")
            
            # Summarize polynomial contribution explicitly
            use_pg = getattr(model, "use_poly_gate", True)
            if use_pg:
                f.write(f"  W_poly = (1-g)·(Σ_i w_low_i·φ_i) + g·(Σ_i w_high_i·φ_i)\n")
                f.write(f"    g = σ((max(λθ,λz) - c)/Δ), with c,Δ learned (poly gate)\n")
            else:
                f.write(f"  W_poly = Σ_i w_i·φ_i  (poly gate off)\n")
            f.write(f"    φ_i are the basis terms listed above\n")
        
        f.write(f"\nFeatures: [I₁, I₄ₕ, I₄ₜ, I₄ᵧ, I₄₋ₕ]\n")
        f.write(f"where I₄₋ₕ = I₄(helical_2)\n")
        
        # All weights for reference (only for legacy models with polynomial basis)
        if not is_composite:
            f.write(f"\nALL POLYNOMIAL WEIGHTS (for reference):\n")
            f.write("-" * 40 + "\n")
            use_pg = getattr(model, "use_poly_gate", True)
            if use_pg:
                w_low  = model.poly_head_low.weight.detach().cpu().numpy().flatten()
                w_high = model.poly_head_high.weight.detach().cpu().numpy().flatten()
                basis_weights = np.maximum(np.abs(w_low), np.abs(w_high))
            else:
                w = model.extra_head.weight.detach().cpu().numpy().flatten()
                basis_weights = np.abs(w)
            
            basis_names = [
                "E_θ²", "E_z²", "E_θ·E_z", "E_θ³", "E_z³", "I₁-3",
                "I₄ₕ²", "I₄ₜ²", "I₄ᵧ²", "I₄ₕ·E_θ", "I₄ₕ·E_θ²",
                "I₄_z·E_θ", "I₄_z·E_z", "(E_θ·E_z)²", "exp(E_θ)-1",
                "exp(E_z)-1", "ln(1+E_θ)", "ln(1+E_z)", "ln(1+I₁-3)"
            ]
            for i, (name, weight) in enumerate(zip(basis_names, basis_weights)):
                f.write(f"  {name:12s}: {weight:10.6f}\n")
    
    print(f"Equation saved to: {equation_file}")

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
    """Create comprehensive comparison plots between base, full ICNN, and simplified equation"""
    
    print("Creating equation comparison plots...")
        
    # Filter out None models
    models_to_compare = []
    base_for_plot = base_model_override if base_model_override is not None else getattr(model, "base", None)
    if base_for_plot is not None:
        models_to_compare.append((base_for_plot, "Base Model", "blue"))
    if model is not None:
        models_to_compare.append((model, "SEDF + poly", "red"))

        
    if not models_to_compare:
        print("No valid models found for comparison")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Equation Comparison: {age} - {base_type} Base', fontsize=16, fontweight='bold')
    
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
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
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

def plot_weight_significance(ax, model):
    """Plot weight significance analysis"""
    
    import numpy as np
    
    # Check if decomposed symbolic model
    is_composite = is_decomposed_model(model)
    is_simple_sedf = isinstance(model, SimpleSEDF)
    is_gated_sedf = isinstance(model, GatedSEDF)
    
    if is_composite:
        # For decomposed symbolic models, plot the symbolic weights only
        if is_gated_sedf:
            gates = model.tweak.gates().detach().cpu().numpy()
            w_vals = model.tweak.w.detach().cpu().numpy()
            gated_weights = gates * np.abs(w_vals)
            basis_names_full = get_basis_names(model, fallback_n=len(gated_weights))
            weights = gated_weights
            names = [basis_names_full[i] if i < len(basis_names_full) else f"term_{i}" for i in range(len(gated_weights))]
        elif is_simple_sedf:
            active_idx = model.tweak.active_idx.cpu().numpy().tolist()
            w_vals = model.tweak.w.detach().cpu().numpy()
            basis_names_full = get_basis_names(model, fallback_n=max(active_idx)+1 if active_idx else len(w_vals))
            weights = np.abs(w_vals)
            names = [basis_names_full[i] if i < len(basis_names_full) else f"term_{i}" for i in active_idx]
        elif hasattr(model, 'tweak'):
            if hasattr(model.tweak, '_w'):
                w_tweak = abs(model.tweak._w.item())
                weights = np.array([w_tweak])
                names = ['w_tweak']
            else:
                weights = np.array([])
                names = []
        else:
            weights = np.array([])
            names = []
        
        basis_weights = weights
        basis_names = names
    else:
        # Legacy models: Get polynomial weights
        use_pg = getattr(model, "use_poly_gate", True)
        if use_pg:
            if hasattr(model, 'poly_head_low'):
                w_low  = model.poly_head_low.weight.detach().cpu().numpy().flatten()
                w_high = model.poly_head_high.weight.detach().cpu().numpy().flatten()
                basis_weights = np.maximum(np.abs(w_low), np.abs(w_high))
            else:
                # Fallback if poly_head doesn't exist
                basis_weights = np.array([])
                basis_names = []
        else:
            if hasattr(model, 'extra_head'):
                w = model.extra_head.weight.detach().cpu().numpy().flatten()
                basis_weights = np.abs(w)
            else:
                basis_weights = np.array([])
                basis_names = []
        
        if len(basis_weights) > 0:
            basis_names = get_basis_names(model, fallback_n=len(basis_weights))
        else:
            basis_names = []
    
    if len(basis_weights) == 0:
        ax.text(0.5, 0.5, 'No weights to display', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Weight Significance')
        return
    
    # Color bars based on significance
    threshold = 0.01 if not is_composite else 0.01
    colors = ['red' if abs(w) > threshold else 'lightblue' for w in basis_weights]
    
    bars = ax.bar(range(len(basis_weights)), basis_weights, color=colors, alpha=0.7)
    
    if is_composite:
        ax.set_xlabel('Symbolic Terms')
        ax.set_ylabel('Weight Value')
        ax.set_title('Symbolic Cross-Term Weights\n(Red: |weight| > 0.01)')
    else:
        ax.set_xlabel('Polynomial Basis Terms')
        ax.set_ylabel('Weight Value')
        ax.set_title('Polynomial Weight Significance\n(Red: |weight| > 0.01)')
    
    ax.set_xticks(range(len(basis_names)))
    ax.set_xticklabels(basis_names, rotation=45, ha='right')
    x0, x1 = ax.get_xlim()
    ax.hlines([threshold, -threshold], xmin=x0, xmax=x1, linestyles="--", linewidth=1.5, colors="red", alpha=0.8, label="Significance threshold")
    ax.legend()

# ================================= Main ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_root",  default="runs_sef")

    # Epochs
    ap.add_argument("--epochs_base",   type=int, default=4000)
    ap.add_argument("--epochs_warmup", type=int, default=7000)
    ap.add_argument("--joint_ft_epochs", type=int, default=0)

    # LRs
    ap.add_argument("--lr_base",        type=float, default=5e-2)
    ap.add_argument("--lr_resid",       type=float, default=1e-3)
    ap.add_argument("--lr_joint_base",  type=float, default=1e-3)
    ap.add_argument("--joint_ft_lr_base", type=float, default=1e-4)
    ap.add_argument("--joint_ft_lr_resid", type=float, default=1e-4)
    ap.add_argument("--joint_ft_lr", type=float, default=1e-3, help="If >0, use this LR for BOTH base and polynomial during final fine‑tune")

    # Regularizers
    ap.add_argument("--reg_gain",  type=float, default=1e-5)
    ap.add_argument("--reg_basis", type=float, default=1e-8)

    # Model choices
    ap.add_argument("--iso", choices=["poly","gent","polezero","neohookean"], default="poly",
                help="Isotropic model for stage 1")
    ap.add_argument("--aniso_model", choices=["2fiber","4fiber"], default="4fiber",
                help="Anisotropic model for stage 1")
    ap.add_argument("--recruit", choices=["on","off"], default="on", help="Enable anisotropic fiber recruitment")
    ap.add_argument("--dispersion", choices=["on","off"], default="on",
                help="Enable dispersion in the 4-fiber anisotropic model")
    ap.add_argument("--recruit_start_mode", choices=["learn","fixed"], default="learn",
                help="Treat the shared recruitment start λ_lb as a learned parameter or hold it fixed")
    ap.add_argument("--recruit_start_init", type=float, default=0.95,
                help="Initial guess or fixed value for the shared recruitment start λ_lb")
    ap.add_argument("--recruit_start_min", type=float, default=0.90,
                help="Lower bound for the learned recruitment start λ_lb")
    ap.add_argument("--recruit_start_max", type=float, default=1.55,
                help="Upper bound for the learned recruitment start λ_lb")
    ap.add_argument("--recruit_end_min", type=float, default=1.15,
                help="Lower bound for the recruitment upper stretch λ_ub")
    ap.add_argument("--recruit_end_max", type=float, default=1.80,
                help="Upper bound for the recruitment upper stretch λ_ub")
    ap.add_argument("--dist", choices=["beta","lognormal","halfnormal"], default="lognormal") # fiber distribution
    ap.add_argument("--tweak_gate", choices=["on","off"], default="on", help="Enable gate on tweak term")
    ap.add_argument("--gate_lambda", type=float, default=0.0, help="L1 penalty on gate values for sparsity")
    ap.add_argument("--gate_thr", type=float, default=0.001,
                help="Threshold on effective symbolic score |gate * weight| for determining active terms")
    ap.add_argument("--anchor_lambda", type=float, default=2e-3, help="L2 penalty on base parameters to stay near reference (joint FT only)")
    ap.add_argument("--stage2_base_lr", type=float, default=0.0,
                help="Learning rate for the base during gated discovery; set to 0 to freeze the base")
    ap.add_argument("--stage2_anchor_lambda", type=float, default=5e-2,
                help="L2 anchor on base parameters during gated discovery so discovery starts from the best base and only drifts when helpful")
    ap.add_argument("--stage2_term_dropout", type=float, default=0.01,
                help="Term-level dropout probability during gated discovery to reveal which symbolic terms are robustly useful")

    ap.add_argument("--robust", choices=["huber","charbonnier","mse"], default="huber")
    ap.add_argument("--robust_delta", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wtheta_boost", type=float, default=4.5)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--jitter_warm_epochs", type=int, default=800)
    ap.add_argument("--seed", type=int, default=125)
    ap.add_argument("--round_step", type=float, default=0.01)

    # Basis options
    ap.add_argument("--n_basis", type=int, default=45,
                help="Number of symbolic candidate terms to retain from the split iso/aniso/cross library")
    ap.add_argument("--max_iso_terms", type=int, default=3,
                help="Maximum number of symbolic isotropic correction terms to retain")
    ap.add_argument("--max_aniso_terms", type=int, default=3,
                help="Maximum number of symbolic anisotropic correction terms to retain")
    ap.add_argument("--max_cross_terms", type=int, default=4,
                help="Maximum number of symbolic coupling terms to retain")

    # Reporting threshold for printing polynomial weights
    ap.add_argument("--weight_thresh", type=float, default=0.001, help="Absolute weight threshold to include basis terms in the exported equation")

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
        
        # Build additive stage-1 base model
        print(f"→ Building stage-1 base ({args.iso} isotropic + {args.aniso_model} anisotropic)")
        base_model = DecomposedSEDFBase(
            iso_model=args.iso,
            aniso_model=args.aniso_model,
            use_recruitment=(args.recruit == "on"),
            use_dispersion=(args.dispersion == "on"),
        ).to(device)

        if hasattr(base_model.aniso, "dist_type"):
            base_model.aniso.dist_type = args.dist
        if hasattr(base_model.aniso, "_recruit_ref_stretch"):
            ref_lamz = configure_recruitment_reference_stretch(base_model.aniso, in_vivo_lamz)
            print(f"→ Recruitment reference stretch set to λz,ref={ref_lamz:.3f}")
        if hasattr(base_model.aniso, "_lambda_lb_raw"):
            recruit_start = configure_recruitment_start(
                base_model.aniso, args.recruit_start_init, args.recruit_start_mode
            )
            print(f"→ Recruitment start λ_lb ({args.recruit_start_mode}) initialized to {recruit_start:.3f}")
        if hasattr(base_model.aniso, "_beta_a_raw"):
            with torch.no_grad():
                init = {0:(2.5,3.0), 1:(2.5,3.2), 2:(2.5,4.0), 3:(2.6,2.0)}
                n_fam = len(base_model.aniso._beta_a_raw)
                for fam, (a,b) in init.items():
                    if fam < n_fam:
                        base_model.aniso._beta_a_raw[fam] = _raw_from_alpha(a)
                        base_model.aniso._beta_b_raw[fam] = _raw_from_alpha(b)
        
        # 1) Fit base (CompositeSEDFBase or legacy base)
        print(f"→ Fitting base ({args.iso.upper()} + {args.aniso_model.upper()})")
        base_loss, base_aic, base_bic, base_model = fit_model(
            base_model, Xth, yth, Xz, yz,
            epochs=args.epochs_base, name=f"{args.iso.upper()}_{args.aniso_model.upper()}-base", lr=args.lr_base,
            wtheta_boost=args.wtheta_boost, jitter=0.0, jitter_warm_epochs=0, seed=args.seed,
            weights_th=wth, weights_z=wz, robust_kind=args.robust, robust_delta=args.robust_delta,
            batch_size=(args.batch_size or None), compile_model=args.compile
        )

        # 2) Stage 2: Gated discovery
        print("→ Stage 2: Gated discovery (freeze base, learn gates + weights)")
            
        gated = GatedSEDF(
            iso_model=args.iso,
            aniso_model=args.aniso_model,
            use_recruitment=(args.recruit == "on"),
            use_dispersion=(args.dispersion == "on"),
            n_basis=args.n_basis,
            init_gate=0.20,
            gate_temp=1.0,
            dropout_p=args.stage2_term_dropout,
        ).to(device)
            
        with torch.no_grad():
            gated.base.load_state_dict(base_model.state_dict(), strict=False)
        if hasattr(gated.base.aniso, "dist_type") and hasattr(base_model.aniso, "dist_type"):
            gated.base.aniso.dist_type = base_model.aniso.dist_type
        if hasattr(gated.base.aniso, "_recruit_ref_stretch") and hasattr(base_model.aniso, "_recruit_ref_stretch"):
            configure_recruitment_reference_stretch(gated.base.aniso, float(base_model.aniso.recruit_ref_stretch().detach().cpu().item()))
        if hasattr(gated.base.aniso, "_lambda_lb_raw"):
            preserve_lb = preserve_recruitment_start(gated.base.aniso, args.recruit_start_mode)
            print(f"→ Stage 2 carrying forward learned λ_lb={preserve_lb:.3f}")
            
        lam_all = _lam_all_from_X(Xth, Xz, device=device)
        stage2_base_ref = {f"base.{n}": p.detach().clone() for n, p in gated.base.named_parameters()}
        with torch.no_grad():
            gated.set_stats(lam_all)
            for p in gated.base.parameters():
                p.requires_grad_(args.stage2_base_lr > 0.0)
            for p in gated.tweak.parameters():
                p.requires_grad_(True)
        if args.stage2_base_lr > 0.0:
            print(f"→ Stage 2 co-training base from best fit (lr_base={args.stage2_base_lr:.2e}, anchor={args.stage2_anchor_lambda:.2e})")
        else:
            print("→ Stage 2 freezing base (stage2_base_lr=0)")
            
        final_loss, final_aic, final_bic, gated = fit_model(
            gated, Xth, yth, Xz, yz,
            epochs=args.epochs_warmup,
            name="GatedDiscovery",
            lr=args.lr_resid,
            lr_base=args.stage2_base_lr,
            lr_resid=args.lr_resid,
            wtheta_boost=args.wtheta_boost,
            jitter=0.0, jitter_warm_epochs=0,
            seed=args.seed,
            weights_th=wth, weights_z=wz,
            robust_kind=args.robust,
            robust_delta=args.robust_delta,
            batch_size=(args.batch_size or None),
            compile_model=args.compile,
            gate_lambda=args.gate_lambda,
            anchor_lambda=args.stage2_anchor_lambda if args.stage2_base_lr > 0.0 else 0.0,
            anchor_params=stage2_base_ref if args.stage2_base_lr > 0.0 else None,
        )
        
        # Inspect selected terms
        bucket_map = gated.basis_buckets()
        gate_vals = gated.tweak.gates().detach().cpu().numpy()
        gate_weight_scores = gate_vals * gated.tweak.w.detach().cpu().numpy()
        active = select_active_by_bucket(
            gate_weight_scores,
            bucket_map,
            max_iso=args.max_iso_terms,
            max_aniso=args.max_aniso_terms,
            max_cross=args.max_cross_terms,
            thr=args.gate_thr,
            force_bucket_fill=False,
        )
        if len(active) == 0:
            print(f"[gates] No gates passed thr={args.gate_thr}. No symbolic terms selected.")
        
        print(f"\n[gates] Active basis indices (threshold={args.gate_thr}): {active}")
        print(f"[gates] Gate values: {gate_vals}")
        print(f"[gates] Effective scores |g·w|: {gate_weight_scores}")
        print(f"[gates] Active basis buckets: {[bucket_map[i] for i in active]}")
        gated.selected_active_idx = list(active)
        gated.selected_gate_scores = gate_weight_scores.tolist()
        final_model = gated
        print("→ Using stage-2 joint discovery model as final symbolic model")

        # Optional extra joint fine-tune of the stage-2 model only
        if args.joint_ft_epochs > 0:
            print("→ Fine-tuning stage-2 model (base + tweak together)")
            base_ref = {f"base.{n}": p.detach().clone() for n, p in final_model.base.named_parameters()}
            for _, p in final_model.base.named_parameters():
                p.requires_grad_(True)
            for p in final_model.tweak.parameters():
                p.requires_grad_(True)

            lr_base_ft = args.joint_ft_lr_base if args.joint_ft_lr_base > 0 else args.stage2_base_lr
            lr_tweak_ft = args.joint_ft_lr_resid if args.joint_ft_lr_resid > 0 else args.lr_resid
            anchor_lambda = args.anchor_lambda if args.anchor_lambda > 0.0 else 0.0

            final_loss, final_aic, final_bic, final_model = fit_model(
                final_model, Xth, yth, Xz, yz,
                epochs=args.joint_ft_epochs,
                name="Stage2-JointFT",
                lr=lr_tweak_ft,
                lr_base=lr_base_ft,
                lr_resid=lr_tweak_ft,
                wtheta_boost=args.wtheta_boost,
                jitter=0.0, jitter_warm_epochs=0,
                seed=args.seed,
                weights_th=wth, weights_z=wz,
                robust_kind=args.robust,
                robust_delta=args.robust_delta,
                batch_size=(args.batch_size or None),
                compile_model=args.compile,
                anchor_lambda=anchor_lambda,
                anchor_params=base_ref,
            )

        metric_rows = [
            compute_fit_metrics(
                base_model, Xth, yth, Xz, yz,
                label="Base", loss=base_loss, aic=base_aic, bic=base_bic
            ),
            compute_fit_metrics(
                final_model, Xth, yth, Xz, yz,
                label="Base + Tweak", loss=final_loss, aic=final_aic, bic=final_bic
            ),
        ]
        save_fit_metrics_table(metric_rows, age, args.out_root)

        # Save equation to file
        save_equation_to_file(final_model, args.iso.upper(), age, args.out_root, weight_thresh=args.weight_thresh)
        
        # Create comparison plots
        print("\n" + "="*60)
        print("CREATING EQUATION COMPARISON PLOTS")
        print("="*60)
        create_equation_comparison_plots(
            model=final_model,              # final symbolic model instance
            Xth=Xth, yth=yth, Xz=Xz, yz=yz,
            base_type=args.iso.upper(),
            age=age,
            out_root=args.out_root,
            base_model_override=base_model,
            include_sigma_in_simplified=False,   # or True to match the full model
            ids_th=ids_th, ids_z=ids_z
        )

        # plots + dumps
        os.makedirs(args.out_root, exist_ok=True)
        plot_results(Xth, yth, Xz, yz, base_model, final_model, age, args.out_root, round_step=args.round_step, ids_th=ids_th, ids_z=ids_z)
        for model, tag in [(base_model, args.iso.upper()), (final_model, "POLY")]:
            path = os.path.join(args.out_root, f"{age}_{tag}_params.txt")
            with open(path, "w") as f:
                f.write(f"=== {tag} Learned Parameters (Age {age}) ===\n\n")
                for pname, p in model.named_parameters():
                    f.write(f"{pname}:\n{p.data.detach().cpu().numpy()}\n\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
