import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------

#json_dir = Path("./liquid")
#output_dir = Path("./plots_liquid")

json_dir = Path("./fcc")
output_dir = Path("./plots_fcc")

file_pattern = "*.json"

TIME_SCALE = 1e-6  # ns -> ms
TIME_UNIT = "ms"

SORT_BY_DENSITY = True

FIT_DENSITY_MIN = 0.5
FIT_DENSITY_MAX = 2.0

SAVE_MAIN_PLOTS = True
SAVE_RATIO_PLOTS = True

SHOW_PLOTS = False


# ---------------------------------------------------------------------
# Plot colors
# ---------------------------------------------------------------------

ENERGY_COLOR = "tab:blue"

INIT_COLOR = "tab:orange"
CACHED_COLOR = "tab:gray"
TINY_COLOR = "tab:green"
LARGE_COLOR = "tab:red"

INIT_FIT_COLOR = "tab:brown"
TINY_FIT_COLOR = "tab:olive"
LARGE_FIT_COLOR = "tab:purple"


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


def get_optional_array(bench, key, fallback_key=None, default=None):
    """
    Get array from benchmark dict.

    If key is missing, use fallback_key.
    If both are missing, use default.
    """
    if key in bench:
        return np.asarray(bench[key], dtype=float)

    if fallback_key is not None and fallback_key in bench:
        return np.asarray(bench[fallback_key], dtype=float)

    if default is not None:
        return np.asarray(default, dtype=float)

    raise KeyError(f"Could not find key '{key}' in benchmark JSON.")


def extract_arrays(bench):
    density = np.asarray(bench["density"], dtype=float)
    energy = np.asarray(bench["energy"], dtype=float)

    init_time = np.asarray(bench["initTime_AVG"], dtype=float) * TIME_SCALE
    init_time_STD = np.asarray(bench["initTime_STD"], dtype=float) * TIME_SCALE

    cached_time = get_optional_array(
        bench,
        "cachedTime_AVG",
        default=np.zeros_like(density),
    ) * TIME_SCALE

    cached_time_STD = get_optional_array(
        bench,
        "cachedTime_STD",
        default=np.zeros_like(density),
    ) * TIME_SCALE

    tiny_time = get_optional_array(
        bench,
        "tinyMoveTime_AVG",
        fallback_key="recalcTime_AVG",
    ) * TIME_SCALE

    tiny_time_STD = get_optional_array(
        bench,
        "tinyMoveTime_STD",
        fallback_key="recalcTime_STD",
    ) * TIME_SCALE

    large_time = get_optional_array(
        bench,
        "largeMoveTime_AVG",
        fallback_key="recalcTime_AVG",
    ) * TIME_SCALE

    large_time_STD = get_optional_array(
        bench,
        "largeMoveTime_STD",
        fallback_key="recalcTime_STD",
    ) * TIME_SCALE

    num_runs_KEPT = np.asarray(bench["numRuns_KEPT"], dtype=float)

    # Avoid divide-by-zero in standard-error computation
    num_runs_KEPT = np.maximum(num_runs_KEPT, 1.0)

    if SORT_BY_DENSITY:
        idx = np.argsort(density)

        density = density[idx]
        energy = energy[idx]

        init_time = init_time[idx]
        init_time_STD = init_time_STD[idx]

        cached_time = cached_time[idx]
        cached_time_STD = cached_time_STD[idx]

        tiny_time = tiny_time[idx]
        tiny_time_STD = tiny_time_STD[idx]

        large_time = large_time[idx]
        large_time_STD = large_time_STD[idx]

        num_runs_KEPT = num_runs_KEPT[idx]

    return {
        "density": density,
        "energy": energy,

        "init_time": init_time,
        "init_time_STD": init_time_STD,

        "cached_time": cached_time,
        "cached_time_STD": cached_time_STD,

        "tiny_time": tiny_time,
        "tiny_time_STD": tiny_time_STD,

        "large_time": large_time,
        "large_time_STD": large_time_STD,

        "num_runs_KEPT": num_runs_KEPT,
    }


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
        "n_points": int(np.count_nonzero(mask)),
    }


# ---------------------------------------------------------------------
# Plotting helper: main benchmark subplot
# ---------------------------------------------------------------------

def plot_single_benchmark(ax_energy, title, arrays):
    """
    One subplot with:
      - energy on the left y-axis
      - init/cached/tiny/large timings on the right y-axis
      - linear fits for init, tinyMoveTime, and largeMoveTime
    """

    density = arrays["density"]
    energy = arrays["energy"]

    init_time = arrays["init_time"]
    init_time_STD = arrays["init_time_STD"]

    cached_time = arrays["cached_time"]
    cached_time_STD = arrays["cached_time_STD"]

    tiny_time = arrays["tiny_time"]
    tiny_time_STD = arrays["tiny_time_STD"]

    large_time = arrays["large_time"]
    large_time_STD = arrays["large_time_STD"]

    num_runs_KEPT = arrays["num_runs_KEPT"]

    stderr_init = init_time_STD / np.sqrt(num_runs_KEPT)
    stderr_cached = cached_time_STD / np.sqrt(num_runs_KEPT)
    stderr_tiny = tiny_time_STD / np.sqrt(num_runs_KEPT)
    stderr_large = large_time_STD / np.sqrt(num_runs_KEPT)

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
    init_line = ax_time.errorbar(
        density,
        init_time,
        yerr=stderr_init,
        fmt="s-",
        color=INIT_COLOR,
        label="Init time",
        markersize=4,
        capsize=3,
    )

    cached_line = ax_time.errorbar(
        density,
        cached_time,
        yerr=stderr_cached,
        fmt=".-",
        color=CACHED_COLOR,
        label="Cached time",
        markersize=3,
        capsize=2,
        alpha=0.65,
    )

    tiny_line = ax_time.errorbar(
        density,
        tiny_time,
        yerr=stderr_tiny,
        fmt="^-",
        color=TINY_COLOR,
        label="Tiny move time",
        markersize=4,
        capsize=3,
    )

    large_line = ax_time.errorbar(
        density,
        large_time,
        yerr=stderr_large,
        fmt="v-",
        color=LARGE_COLOR,
        label="Large move time",
        markersize=4,
        capsize=3,
    )

    # Linear fits in selected density window
    init_fit = linear_fit_in_density_window(
        density,
        init_time,
        rho_min=FIT_DENSITY_MIN,
        rho_max=FIT_DENSITY_MAX,
    )

    tiny_fit = linear_fit_in_density_window(
        density,
        tiny_time,
        rho_min=FIT_DENSITY_MIN,
        rho_max=FIT_DENSITY_MAX,
    )

    large_fit = linear_fit_in_density_window(
        density,
        large_time,
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
            label=f"Init fit: slope={init_fit['slope']:.3g}",
        )
        fit_lines.append(init_fit_line)

    if tiny_fit is not None:
        tiny_fit_line, = ax_time.plot(
            tiny_fit["x_line"],
            tiny_fit["y_line"],
            "--",
            color=TINY_FIT_COLOR,
            linewidth=2,
            label=f"Tiny fit: slope={tiny_fit['slope']:.3g}",
        )
        fit_lines.append(tiny_fit_line)

    if large_fit is not None:
        large_fit_line, = ax_time.plot(
            large_fit["x_line"],
            large_fit["y_line"],
            "--",
            color=LARGE_FIT_COLOR,
            linewidth=2,
            label=f"Large fit: slope={large_fit['slope']:.3g}",
        )
        fit_lines.append(large_fit_line)

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

    lines = [
        energy_line,
        init_line,
        cached_line,
        tiny_line,
        large_line,
    ] + fit_lines

    labels = [
        "Energy",
        "Init time",
        "Cached time",
        "Tiny move time",
        "Large move time",
    ] + [line.get_label() for line in fit_lines]

    ax_energy.legend(lines, labels, loc="best", fontsize=7)

    return ax_energy, ax_time, init_fit, tiny_fit, large_fit


# ---------------------------------------------------------------------
# Main benchmark figure
# ---------------------------------------------------------------------

def plot_benchmark_file(json_path):
    model_entry, lj_entry = load_benchmarks(json_path)

    model_name = model_entry["model"]
    lj_name = lj_entry["model"]

    model_arrays = extract_arrays(model_entry["model_bench"])
    lj_arrays = extract_arrays(lj_entry["lj_bench"])

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(22, 6),
        sharex=False,
    )

    _, _, model_init_fit, model_tiny_fit, model_large_fit = plot_single_benchmark(
        axes[0],
        short_model_name(model_name),
        model_arrays,
    )

    _, _, lj_init_fit, lj_tiny_fit, lj_large_fit = plot_single_benchmark(
        axes[1],
        short_model_name(lj_name),
        lj_arrays,
    )

    fig.suptitle(json_path.name, fontsize=12)
    fig.tight_layout()

    print()
    print(json_path.name)

    def print_fit(label, fit):
        if fit is not None:
            print(
                f"  {label}: "
                f"slope={fit['slope']:.6g} {TIME_UNIT}/density "
                f"using {fit['n_points']} points"
            )

    print_fit("Model init", model_init_fit)
    print_fit("Model tiny move", model_tiny_fit)
    print_fit("Model large move", model_large_fit)

    print_fit("LJ init", lj_init_fit)
    print_fit("LJ tiny move", lj_tiny_fit)
    print_fit("LJ large move", lj_large_fit)

    def safe_ratio(num, den):
        if den is None:
            return None
        if abs(den["slope"]) < 1e-14:
            return None
        return num["slope"] / den["slope"]

    if model_tiny_fit is not None and model_large_fit is not None:
        ratio = safe_ratio(model_large_fit, model_tiny_fit)
        if ratio is not None:
            print(f"  Model large/tiny slope ratio: {ratio:.6g}")

    if lj_tiny_fit is not None and lj_large_fit is not None:
        ratio = safe_ratio(lj_large_fit, lj_tiny_fit)
        if ratio is not None:
            print(f"  LJ large/tiny slope ratio: {ratio:.6g}")

    return fig


# ---------------------------------------------------------------------
# Ratio plot: largeMoveTime / tinyMoveTime
# ---------------------------------------------------------------------

def plot_large_tiny_ratio(json_path):
    model_entry, lj_entry = load_benchmarks(json_path)

    model_name = model_entry["model"]
    lj_name = lj_entry["model"]

    model_arrays = extract_arrays(model_entry["model_bench"])
    lj_arrays = extract_arrays(lj_entry["lj_bench"])

    fig, ax = plt.subplots(figsize=(10, 5))

    model_ratio = model_arrays["large_time"] / model_arrays["tiny_time"]
    lj_ratio = lj_arrays["large_time"] / lj_arrays["tiny_time"]

    ax.plot(
        model_arrays["density"],
        model_ratio,
        "o-",
        label=short_model_name(model_name),
        markersize=4,
    )

    ax.plot(
        lj_arrays["density"],
        lj_ratio,
        "s-",
        label=short_model_name(lj_name),
        markersize=4,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        color="black",
        label="large = tiny",
    )

    ax.set_xlabel("Relative density")
    ax.set_ylabel("largeMoveTime / tinyMoveTime")
    ax.set_title(f"Large-vs-tiny perturbation cost ratio\n{json_path.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------
# Difference plot: largeMoveTime - tinyMoveTime
# ---------------------------------------------------------------------

def plot_large_tiny_difference(json_path):
    model_entry, lj_entry = load_benchmarks(json_path)

    model_name = model_entry["model"]
    lj_name = lj_entry["model"]

    model_arrays = extract_arrays(model_entry["model_bench"])
    lj_arrays = extract_arrays(lj_entry["lj_bench"])

    fig, ax = plt.subplots(figsize=(10, 5))

    model_diff = model_arrays["large_time"] - model_arrays["tiny_time"]
    lj_diff = lj_arrays["large_time"] - lj_arrays["tiny_time"]

    ax.plot(
        model_arrays["density"],
        model_diff,
        "o-",
        label=short_model_name(model_name),
        markersize=4,
    )

    ax.plot(
        lj_arrays["density"],
        lj_diff,
        "s-",
        label=short_model_name(lj_name),
        markersize=4,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
        color="black",
        label="large - tiny = 0",
    )

    ax.set_xlabel("Relative density")
    ax.set_ylabel(f"largeMoveTime - tinyMoveTime [{TIME_UNIT}]")
    ax.set_title(f"Large-vs-tiny perturbation time difference\n{json_path.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------
# Iterate through all JSON files
# ---------------------------------------------------------------------

json_files = sorted(json_dir.glob(file_pattern))

if not json_files:
    raise FileNotFoundError(f"No JSON files found in {json_dir.resolve()}")

output_dir.mkdir(parents=True, exist_ok=True)

print(f"Found {len(json_files)} JSON files.")

for i, json_path in enumerate(json_files, start=1):
    print(f"[{i}/{len(json_files)}] Plotting {json_path.name}")

    if SAVE_MAIN_PLOTS:
        fig = plot_benchmark_file(json_path)
        output_path = output_dir / f"{json_path.stem}.png"
        fig.savefig(output_path, dpi=300)
        print(f"Saved {output_path}")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)

    if SAVE_RATIO_PLOTS:
        ratio_fig = plot_large_tiny_ratio(json_path)
        ratio_output_path = output_dir / f"{json_path.stem}_large_tiny_ratio.png"
        ratio_fig.savefig(ratio_output_path, dpi=300)
        print(f"Saved {ratio_output_path}")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(ratio_fig)

        diff_fig = plot_large_tiny_difference(json_path)
        diff_output_path = output_dir / f"{json_path.stem}_large_tiny_difference.png"
        diff_fig.savefig(diff_output_path, dpi=300)
        print(f"Saved {diff_output_path}")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(diff_fig)
