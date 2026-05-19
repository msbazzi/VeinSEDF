"""
Postprocess small-on-large outputs for unified variant f.

Reads files like:
    {root}/{N}weeks/{Animal}/runs_unified/*_variant_f_small_on_large.txt
with Mojito read from runs_unified_reduced by default.

and writes:
    small_on_large_variant_f_summary/small_on_large_variant_f_terms.csv
    small_on_large_variant_f_summary/K_FSG/*.png
    small_on_large_variant_f_summary/C_out/*.png

Each plot is a box-and-whisker plot with age in weeks on x and the selected
small-on-large term value in kPa on y.

Usage:
    python postprocess_small_on_large_variant_f.py
    python postprocess_small_on_large_variant_f.py --root . --out_root runs_unified
    python postprocess_small_on_large_variant_f.py --animal_out_root Mojito=runs_unified
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "veinsdf_matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


AGE_DIR_RE = re.compile(r"^(\d+)\s*weeks?$", re.IGNORECASE)
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
K_LABELS = ["theta", "z", "thetaz", "rtheta", "rz"]
K_LABELS_LATEX = {
    "theta": r"\theta",
    "z": r"z",
    "thetaz": r"\theta z",
    "rtheta": r"r\theta",
    "rz": r"rz",
}
C_OUT_LABELS = [
    "C_thetathetathetatheta",
    "C_zzzz",
    "C_thetathetazz",
    "C_thetazthetaz_pre",
]
C_OUT_LABELS_LATEX = {
    "C_thetathetathetatheta": r"$C_{\theta\theta\theta\theta}$",
    "C_zzzz": r"$C_{zzzz}$",
    "C_thetathetazz": r"$C_{\theta\theta zz}$",
    "C_thetazthetaz_pre": r"$C_{\theta z\theta z}^{\mathrm{pre}}$",
}


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def discover_small_on_large_files(
    root: Path,
    out_root: str,
    variant: str,
    animal_out_roots: dict[str, str],
) -> list[tuple[int, str, str, Path]]:
    """Return (age_weeks, animal, out_root, small_on_large_txt) tuples."""
    found: list[tuple[int, str, str, Path]] = []
    for age_entry in sorted(root.iterdir()):
        if not age_entry.is_dir():
            continue
        age_match = AGE_DIR_RE.match(age_entry.name)
        if not age_match:
            continue
        age_weeks = int(age_match.group(1))
        for animal_dir in sorted(age_entry.iterdir()):
            if not animal_dir.is_dir():
                continue
            animal = animal_dir.name
            selected_out_root = animal_out_roots.get(animal, out_root)
            run_dir = animal_dir / selected_out_root
            if not run_dir.is_dir():
                continue
            pattern = str(run_dir / f"*_variant_{variant}_small_on_large.txt")
            matches = sorted(Path(p) for p in glob.glob(pattern))
            if len(matches) == 1:
                found.append((age_weeks, animal, selected_out_root, matches[0]))
            elif len(matches) > 1:
                print(f"  [skip] {age_entry.name}/{animal}: multiple matches")
            else:
                print(f"  [skip] {age_entry.name}/{animal}: no small-on-large file")
    return found


def parse_animal_out_roots(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {"Mojito": "runs_unified_reduced"}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected ANIMAL=OUT_ROOT override, got {value!r}")
        animal, selected_out_root = value.split("=", maxsplit=1)
        animal = animal.strip()
        selected_out_root = selected_out_root.strip()
        if not animal or not selected_out_root:
            raise ValueError(f"Expected ANIMAL=OUT_ROOT override, got {value!r}")
        overrides[animal] = selected_out_root
    return overrides


def _numbers_from_line(line: str) -> list[float]:
    return [float(x) for x in FLOAT_RE.findall(line)]


def parse_small_on_large(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse K_FSG (5x5) and C_out (4,) MPa arrays from a text report."""
    lines = path.read_text(encoding="utf-8").splitlines()

    k_start = None
    c_start = None
    for i, line in enumerate(lines):
        if line.startswith("Computational K_FSG"):
            k_start = i + 1
        elif line.startswith("C_out summary"):
            c_start = i + 1

    if k_start is None:
        raise ValueError(f"Could not find K_FSG block in {path}")
    if c_start is None:
        raise ValueError(f"Could not find C_out summary in {path}")

    k_rows: list[list[float]] = []
    for line in lines[k_start:]:
        vals = _numbers_from_line(line)
        if vals:
            k_rows.append(vals)
        if len(k_rows) == 5:
            break
    k_fsg = np.array(k_rows, dtype=float)
    if k_fsg.shape != (5, 5):
        raise ValueError(f"Expected 5x5 K_FSG in {path}, got shape {k_fsg.shape}")

    c_out = None
    for line in lines[c_start:]:
        vals = _numbers_from_line(line)
        if vals:
            c_out = np.array(vals, dtype=float)
            break
    if c_out is None or c_out.shape != (4,):
        got = None if c_out is None else c_out.shape
        raise ValueError(f"Expected 4-value C_out in {path}, got {got}")

    return k_fsg, c_out


def build_terms_table(items: Iterable[tuple[int, str, str, Path]]) -> pd.DataFrame:
    """Build a tidy table with one row per animal/term."""
    rows: list[dict[str, object]] = []
    for age_weeks, animal, out_root, path in items:
        k_fsg, c_out = parse_small_on_large(path)
        source_prefix = path.name.replace("_small_on_large.txt", "")

        for i, row_label in enumerate(K_LABELS):
            for j, col_label in enumerate(K_LABELS):
                rows.append(
                    {
                        "age_weeks": age_weeks,
                        "animal": animal,
                        "out_root": out_root,
                        "source_prefix": source_prefix,
                        "source_file": str(path),
                        "term_group": "K_FSG",
                        "term": f"K_FSG_{row_label}_{col_label}",
                        "row_label": row_label,
                        "col_label": col_label,
                        "value_mpa": float(k_fsg[i, j]),
                        "value_kpa": float(k_fsg[i, j] * 1000.0),
                    }
                )

        for label, value in zip(C_OUT_LABELS, c_out):
            rows.append(
                {
                    "age_weeks": age_weeks,
                    "animal": animal,
                    "out_root": out_root,
                    "source_prefix": source_prefix,
                    "source_file": str(path),
                    "term_group": "C_out",
                    "term": label,
                    "row_label": "",
                    "col_label": "",
                    "value_mpa": float(value),
                    "value_kpa": float(value * 1000.0),
                }
            )

    return pd.DataFrame(rows)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def term_to_latex(term_group: str, term: str) -> str:
    if term_group == "C_out":
        return C_OUT_LABELS_LATEX.get(term, term)
    if term_group == "K_FSG" and term.startswith("K_FSG_"):
        _, _, row_label, col_label = term.split("_", maxsplit=3)
        row = K_LABELS_LATEX.get(row_label, row_label)
        col = K_LABELS_LATEX.get(col_label, col_label)
        return rf"$K^{{\mathrm{{FSG}}}}_{{{row},{col}}}$"
    return term.replace("_", r"\_")


def _format_tick(value: float, _pos: int) -> str:
    if abs(value) >= 100:
        text = f"{value:.0f}"
    elif abs(value) >= 10:
        text = f"{value:.1f}"
    elif abs(value) >= 1:
        text = f"{value:.2f}"
    else:
        text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def set_three_y_ticks(ax: plt.Axes, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        lower, upper = -1.0, 1.0
    else:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        if np.isclose(lower, upper):
            pad = max(abs(lower) * 0.15, 1.0)
        else:
            pad = 0.12 * (upper - lower)
        lower -= pad
        upper += pad
        if np.min(finite) >= 0:
            lower = 0.0

    ticks = np.linspace(lower, upper, 3)
    ax.set_ylim(lower, upper)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_tick))


def plot_term_boxplot(df: pd.DataFrame, term_group: str, term: str, out_path: Path) -> None:
    sub = df[(df["term_group"] == term_group) & (df["term"] == term)].copy()
    if sub.empty:
        return

    ages = sorted(sub["age_weeks"].unique())
    data = [
        sub.loc[sub["age_weeks"] == age, "value_kpa"].dropna().to_numpy(dtype=float)
        for age in ages
    ]
    all_values = sub["value_kpa"].dropna().to_numpy(dtype=float)
    label = term_to_latex(term_group, term)

    fig, ax = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
    ax.boxplot(
        data,
        tick_labels=[str(age) for age in ages],
        showmeans=True,
        patch_artist=True,
        widths=0.55,
        meanprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#111111",
            "markeredgewidth": 0.8,
            "markersize": 3.5,
        },
        medianprops={"color": "#111111", "linewidth": 1.1},
        boxprops={"facecolor": "#dbeafe", "edgecolor": "#1d4ed8", "linewidth": 0.9},
        whiskerprops={"color": "#1d4ed8", "linewidth": 0.8},
        capprops={"color": "#1d4ed8", "linewidth": 0.8},
        flierprops={
            "marker": ".",
            "markerfacecolor": "#991b1b",
            "markeredgecolor": "#991b1b",
            "alpha": 0.65,
        },
    )

    rng = np.random.default_rng(0)
    for x, values in enumerate(data, start=1):
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.055, 0.055, size=values.size)
        ax.scatter(
            np.full(values.size, x) + jitter,
            values,
            s=9,
            color="#111827",
            alpha=0.45,
            zorder=3,
        )

    set_three_y_ticks(ax, all_values)
    ax.set_xlabel("Age [weeks]")
    ax.set_ylabel(f"{label} (kPa)")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.45, alpha=0.75)
    ax.tick_params(axis="both", direction="out", length=3.0, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_all_terms(df: pd.DataFrame, summary_dir: Path) -> None:
    for term_group, group_df in df.groupby("term_group"):
        out_dir = summary_dir / term_group
        for term in sorted(group_df["term"].unique()):
            plot_term_boxplot(df, term_group, term, out_dir / f"{safe_filename(term)}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create whisker plots for runs_unified variant-f small-on-large terms."
    )
    parser.add_argument("--root", default=".", help="Repository/data root.")
    parser.add_argument("--out_root", default="runs_unified", help="Run output folder to scan.")
    parser.add_argument("--variant", default="f", help="Variant letter to postprocess.")
    parser.add_argument(
        "--animal_out_root",
        action="append",
        default=[],
        metavar="ANIMAL=OUT_ROOT",
        help=(
            "Override the run folder for one animal. Default: "
            "Mojito=runs_unified_reduced. Repeat for multiple animals."
        ),
    )
    parser.add_argument(
        "--summary_dir",
        default="small_on_large_variant_f_summary",
        help="Directory for CSV and plots.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary_dir = (root / args.summary_dir).resolve()
    animal_out_roots = parse_animal_out_roots(args.animal_out_root)

    items = discover_small_on_large_files(root, args.out_root, args.variant, animal_out_roots)
    if not items:
        raise SystemExit(
            f"No *_variant_{args.variant}_small_on_large.txt files found under {root}"
        )

    df = build_terms_table(items)
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"small_on_large_variant_{args.variant}_terms.csv"
    df.sort_values(["term_group", "term", "age_weeks", "animal"]).to_csv(csv_path, index=False)
    plot_all_terms(df, summary_dir)

    print(f"Parsed {len(items)} small-on-large reports.")
    print(f"Wrote {csv_path}")
    print(f"Wrote plots under {summary_dir}")


if __name__ == "__main__":
    main()
