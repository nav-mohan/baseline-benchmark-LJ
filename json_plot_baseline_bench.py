import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

#json_dir = Path("./baseline_bench_data_liquid")   # liquid fcc
#output_dir = Path("plots_liquid")

json_dir = Path("./baseline_bench_data_whileLoopFlush")   # slightly perturbed fcc
output_dir = Path("plots_fcc")

file_pattern = "*.json"

TIME_SCALE = 1e-6  # ns -> ms
TIME_UNIT = "ms"

SORT_BY_DENSITY = True
SHOW_ONE_AT_A_TIME = True

FIT_DENSITY_MIN = 0.5
FIT_DENSITY_MAX = 2.0

ENERGY_COLOR = "tab:blue"
INIT_COLOR = "tab:orange"
RECALC_COLOR = "tab:green"

INIT_FIT_COLOR = "tab:red"
RECALC_FIT_COLOR = "tab:purple"


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_benchmarks(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    model_entry = None
    lj_entry = None

    for entry in data:
        if "model_bench" in entry:
            model_entry = entry
        elif "lj_bench" in entry:
            lj_entry = entry

    if model_entry is None:
        raise ValueError(f"{json_path}: could not find 'model_bench'.")

    if lj_entry is None:
        raise ValueError(f"{json_path}: could not find 'lj_bench'.")

    return model_entry, lj_entry


def extract_arrays(bench):
    density = np.asarray(bench["density"], dtype=float)
    energy = np.asarray(bench["energy"], dtype=float)
    init_time = np.asarray(bench["initTime_AVG"], dtype=float) * TIME_SCALE
    recalc_time = np.asarray(bench["recalcTime_AVG"], dtype=float) * TIME_SCALE

    if SORT_BY_DENSITY:
        idx = np.argsort(density)
        density = density[idx]
        energy = energy[idx]
        init_time = init_time[idx]
        recalc_time = recalc_time[idx]

    return density, energy, init_time, recalc_time


def short_model_name(model_name):
    if "__" in model_name:
        return model_name.split("__")[0]
    return model_name


# ---------------------------------------------------------------------
# Fit helper
# ---------------------------------------------------------------------

def linear_fit_in_density_window(density, y, rho_min=0.5, rho_max=2.0):
    """
    Fit y = m * density + b using only points with density in [rho_min, rho_max].
    """
    density = np.asarray(density, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(density)
        & np.isfinite(y)
        & (density >= rho_min)
        & (density <= rho_max)
    )

    if np.count_nonzero(mask) < 2:
        return None

    x_fit_data = density[mask]
    y_fit_data = y[mask]

    m, b = np.polyfit(x_fit_data, y_fit_data, deg=1)

    x_line = np.linspace(x_fit_data.min(), x_fit_data.max(), 200)
    y_line = m * x_line + b

    return {
        "slope": m,
        "intercept": b,
        "x_line": x_line,
        "y_line": y_line,
        "n_points": np.count_nonzero(mask),
    }


# ---------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------

def plot_single_benchmark(ax_energy, title, density, energy, init_time, recalc_time):
    """
    One subplot with:
      - energy on the left y-axis
      - init time and recalc time on the same right y-axis
      - linear fits for init time and recalc time over density [0.5, 2.0]
    """

    ax_time = ax_energy.twinx()

    # Energy curve
    energy_line, = ax_energy.plot(
        density,
        energy,
        "o-",
        color=ENERGY_COLOR,
        label="Energy",
        markersize=4,
    )

    # Timing curves
    init_line, = ax_time.plot(
        density,
        init_time,
        "s-",
        color=INIT_COLOR,
        label="Init time",
        markersize=4,
    )

    recalc_line, = ax_time.plot(
        density,
        recalc_time,
        "^-",
        color=RECALC_COLOR,
        label="Recalc time",
        markersize=4,
    )

    # Linear fits in selected density window
    init_fit = linear_fit_in_density_window(
        density,
        init_time,
        rho_min=FIT_DENSITY_MIN,
        rho_max=FIT_DENSITY_MAX,
    )

    recalc_fit = linear_fit_in_density_window(
        density,
        recalc_time,
        rho_min=FIT_DENSITY_MIN,
        rho_max=FIT_DENSITY_MAX,
    )

    fit_lines = []

    if init_fit is not None:
        init_fit_line, = ax_time.plot(
            init_fit["x_line"],
            init_fit["y_line"],
            "--",
            color=INIT_FIT_COLOR,
            linewidth=2,
            label=f"Init fit: slope={init_fit['slope']:.3g} {TIME_UNIT}/density",
        )
        fit_lines.append(init_fit_line)

    if recalc_fit is not None:
        recalc_fit_line, = ax_time.plot(
            recalc_fit["x_line"],
            recalc_fit["y_line"],
            "--",
            color=RECALC_FIT_COLOR,
            linewidth=2,
            label=f"Recalc fit: slope={recalc_fit['slope']:.3g} {TIME_UNIT}/density",
        )
        fit_lines.append(recalc_fit_line)

    # Mark fit window
    ax_energy.axvspan(
        FIT_DENSITY_MIN,
        FIT_DENSITY_MAX,
        alpha=0.08,
        label="Fit density window",
    )

    ax_energy.set_title(title)
    ax_energy.set_xlabel("Relative density")
    ax_energy.set_ylabel("Energy", color=ENERGY_COLOR)
    ax_energy.tick_params(axis="y", labelcolor=ENERGY_COLOR)

    ax_time.set_ylabel(f"Time [{TIME_UNIT}]")

    ax_energy.grid(True, alpha=0.3)

    lines = [energy_line, init_line, recalc_line] + fit_lines
    labels = [line.get_label() for line in lines]

    ax_energy.legend(lines, labels, loc="best", fontsize=8)

    return ax_energy, ax_time, init_fit, recalc_fit


def plot_benchmark_file(json_path):
    model_entry, lj_entry = load_benchmarks(json_path)

    model_name = model_entry["model"]
    lj_name = lj_entry["model"]

    model_density, model_energy, model_init, model_recalc = extract_arrays(
        model_entry["model_bench"]
    )

    lj_density, lj_energy, lj_init, lj_recalc = extract_arrays(
        lj_entry["lj_bench"]
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, 6),
        sharex=False,
    )

    _, _, model_init_fit, model_recalc_fit = plot_single_benchmark(
        axes[0],
        short_model_name(model_name),
        model_density,
        model_energy,
        model_init,
        model_recalc,
    )

    _, _, lj_init_fit, lj_recalc_fit = plot_single_benchmark(
        axes[1],
        short_model_name(lj_name),
        lj_density,
        lj_energy,
        lj_init,
        lj_recalc,
    )

    fig.suptitle(json_path.name, fontsize=12)
    fig.tight_layout()

    print()
    print(json_path.name)

    if model_init_fit is not None:
        print(
            f"  Model init slope:   "
            f"{model_init_fit['slope']:.6g} {TIME_UNIT}/density "
            f"using {model_init_fit['n_points']} points"
        )

    if model_recalc_fit is not None:
        print(
            f"  Model recalc slope: "
            f"{model_recalc_fit['slope']:.6g} {TIME_UNIT}/density "
            f"using {model_recalc_fit['n_points']} points"
        )

    if lj_init_fit is not None:
        print(
            f"  LJ init slope:      "
            f"{lj_init_fit['slope']:.6g} {TIME_UNIT}/density "
            f"using {lj_init_fit['n_points']} points"
        )

    if lj_recalc_fit is not None:
        print(
            f"  LJ recalc slope:    "
            f"{lj_recalc_fit['slope']:.6g} {TIME_UNIT}/density "
            f"using {lj_recalc_fit['n_points']} points"
        )

    return fig


# ---------------------------------------------------------------------
# Iterate through all JSON files
# ---------------------------------------------------------------------

json_files = sorted(json_dir.glob(file_pattern))

if not json_files:
    raise FileNotFoundError(f"No JSON files found in {json_dir.resolve()}")

print(f"Found {len(json_files)} JSON files.")

for i, json_path in enumerate(json_files, start=1):
    print(f"[{i}/{len(json_files)}] Plotting {json_path.name}")

    fig = plot_benchmark_file(json_path)

    if SHOW_ONE_AT_A_TIME:
#        plt.show()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{json_path.stem}.png"
        fig.savefig(output_path, dpi=300)
        print(f"Saved {output_path}")
        plt.close(fig)

    else:
        plt.show(block=False)

if not SHOW_ONE_AT_A_TIME:
    plt.show()
