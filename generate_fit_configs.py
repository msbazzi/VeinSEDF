"""
generate_fit_configs.py
-----------------------
For each animal directory in Single_fitting/, reads:
  1. Base_Poly_new_updated.py  — argparse default= values (hyperparameters)
  2. runsfinal/{age}_POLY_params.txt — learned base model parameters

Writes:
  data/{age}/fit_config.json  — ready to be read by Juvenile_SEDF_final.py

Usage (run from inside each animal's directory, or pass --animals):
  cd Single_fitting
  python generate_fit_configs.py                # all animals found
  python generate_fit_configs.py --animals Audi Lexus
"""

import argparse
import ast
import json
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameter keys to extract from argparse defaults in Base_Poly_new_updated.py
# ---------------------------------------------------------------------------
HPARAM_KEYS = {
    "epochs_base":          ("int",   4000),
    "epochs_warmup":        ("int",   7000),
    "lr_base":              ("float", 5e-2),
    "lr_resid":             ("float", 1e-3),
    "wtheta_boost":         ("float", 3.5),
    "robust_delta":         ("float", 0.5),
    "recruit_start_init":   ("float", 0.95),
    "recruit_start_min":    ("float", 0.90),
    "recruit_start_max":    ("float", 1.55),
    "recruit_end_min":      ("float", 1.15),
    "recruit_end_max":      ("float", 1.80),
    "gate_thr":             ("float", 0.001),
    "seed":                 ("int",   125),
    "recruit_start_mode":   ("str",   "learn"),
}


def _parse_hyperparams(base_poly_path: str) -> dict:
    """Extract argparse default= values by regex."""
    with open(base_poly_path, "r") as f:
        src = f.read()

    results = {}
    for key, (typ, fallback) in HPARAM_KEYS.items():
        # Match:  "--key", ...., default=VALUE
        pattern = rf'["\']--{re.escape(key)}["\'].*?default\s*=\s*([^,\)]+)'
        m = re.search(pattern, src, re.DOTALL)
        if m:
            raw = m.group(1).strip().strip('"').strip("'")
            try:
                if typ == "int":
                    results[key] = int(float(raw))
                elif typ == "float":
                    results[key] = float(raw)
                else:
                    results[key] = raw
            except ValueError:
                results[key] = fallback
        else:
            results[key] = fallback

    return results


# ---------------------------------------------------------------------------
# Parse the POLY_params.txt learned-parameter files
# ---------------------------------------------------------------------------
def _parse_numpy_array(s: str):
    """Parse a scalar or numpy-array-repr string into a Python list or float."""
    s = s.strip()
    if s.startswith("["):
        # numpy array repr: parse with ast after removing formatting
        clean = re.sub(r"\s+", " ", s.replace("\n", " "))
        try:
            return ast.literal_eval(clean)
        except Exception:
            nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", clean)
            return [float(x) for x in nums]
    else:
        return float(s)


def _parse_poly_params(params_path: str) -> dict:
    """
    Parse {age}_POLY_params.txt.  Returns a dict of
    { "base.iso._bth": float, "base.aniso._k1": [float,...], ... }
    excluding tweak.* entries (not needed for warm-starting the base).
    """
    with open(params_path, "r") as f:
        lines = f.readlines()

    params = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Section header line: "base.xxx.yyy:" or "tweak.xxx:"
        if line.endswith(":") and (line.startswith("base.") or line.startswith("tweak.")):
            key = line[:-1]
            # Gather value lines (until next key or EOF)
            val_lines = []
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if nxt == "" or (nxt.endswith(":") and ("." in nxt)):
                    break
                val_lines.append(nxt)
                i += 1
            if val_lines and not key.startswith("tweak."):
                raw = " ".join(val_lines)
                params[key] = _parse_numpy_array(raw)
        else:
            i += 1

    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_config(animal_dir: str, animal_name: str) -> str | None:
    """
    Generate data/{age}/fit_config.json for one animal.
    Returns the output path on success, None if required files are missing.
    """
    runsfinal = os.path.join(animal_dir, "runsfinal")

    # Base_Poly_new_updated.py may sit in the animal root OR in runsfinal/
    base_poly = os.path.join(animal_dir, "Base_Poly_new_updated.py")
    if not os.path.isfile(base_poly):
        base_poly = os.path.join(runsfinal, "Base_Poly_new_updated.py")

    # Params file: try exact name, then any *_POLY_params.txt in runsfinal/
    params_txt = os.path.join(runsfinal, f"{animal_name}_POLY_params.txt")
    if not os.path.isfile(params_txt) and os.path.isdir(runsfinal):
        candidates = [
            os.path.join(runsfinal, f)
            for f in os.listdir(runsfinal)
            if f.endswith("_POLY_params.txt")
        ]
        if candidates:
            params_txt = candidates[0]

    data_dir = os.path.join(animal_dir, "data", animal_name)

    missing = []
    if not os.path.isfile(base_poly):
        missing.append(f"Base_Poly_new_updated.py (looked in {animal_dir} and {runsfinal})")
    if not os.path.isfile(params_txt):
        missing.append(f"*_POLY_params.txt in {runsfinal}")
    if missing:
        print(f"  [SKIP] {animal_name}: missing files: {missing}")
        return None

    hparams = _parse_hyperparams(base_poly)
    init_params = _parse_poly_params(params_txt)

    config = {
        "animal": animal_name,
        "hyperparameters": hparams,
        "initial_params": init_params,
    }

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "fit_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  [OK]   {animal_name}: wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate per-animal fit_config.json files.")
    ap.add_argument("--root", default=".",
                    help="Single_fitting root directory (default: current dir)")
    ap.add_argument("--animals", nargs="*", default=None,
                    help="Animal names to process (default: all found)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)

    # Discover animal dirs (contain Base_Poly_new_updated.py OR runsfinal/)
    if args.animals:
        candidates = args.animals
    else:
        candidates = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and not d.startswith(".")
            and d not in ("__pycache__",)
        )

    print(f"Processing {len(candidates)} animal(s) in {root}")
    ok, skip = 0, 0
    for name in candidates:
        animal_dir = os.path.join(root, name)
        result = generate_config(animal_dir, name)
        if result:
            ok += 1
        else:
            skip += 1

    print(f"\nDone: {ok} config(s) written, {skip} skipped.")


if __name__ == "__main__":
    main()
