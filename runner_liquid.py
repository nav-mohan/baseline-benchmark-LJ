import argparse
import importlib
import json
import multiprocessing as mp
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.optimize

import kim_tools.ase as kim_ase_utils
from ase import Atoms
from ase.calculators.kim import KIM
from ase.lattice.cubic import FaceCenteredCubic


# =============================================================================
# Timing helpers
# =============================================================================

def time_energy(atoms):
    start = time.perf_counter_ns()
    pe = atoms.get_potential_energy()
    end = time.perf_counter_ns()
    return pe, end - start


def neighborlist_rebuild_diagnostic(
    atoms,
    base_positions,
    alat,
    pert_small=1e-5,
    pert_large=0.05,
    seed_small=123456,
    seed_large=123457,
):
    """
    Compare three timing regimes after the first energy calculation.

    Assumes:
      - atoms already has a calculator
      - atoms has already had one get_potential_energy() call
      - base_positions is the unperturbed liquid-like reference configuration
    """

    # 1. Cached lookup: no changes
    pe_cached, t_cached = time_energy(atoms)

    # 2. Tiny move
    atoms.set_positions(base_positions.copy())
    kim_ase_utils.randomize_positions(
        atoms,
        pert_small * alat,
        seed=seed_small,
    )

    tiny_changes = atoms.calc.check_state(atoms)

    if "positions" not in tiny_changes:
        raise RuntimeError(
            f"ASE does not see position changes for tiny move. "
            f"changes={tiny_changes}"
        )

    pe_tiny, t_tiny = time_energy(atoms)

    # 3. Large move
    atoms.set_positions(base_positions.copy())
    kim_ase_utils.randomize_positions(
        atoms,
        pert_large * alat,
        seed=seed_large,
    )

    large_changes = atoms.calc.check_state(atoms)

    if "positions" not in large_changes:
        raise RuntimeError(
            f"ASE does not see position changes for large move. "
            f"changes={large_changes}"
        )

    pe_large, t_large = time_energy(atoms)

    return {
        "cached_time": t_cached,
        "tiny_move_time": t_tiny,
        "large_move_time": t_large,
        "cached_energy": pe_cached,
        "tiny_move_energy": pe_tiny,
        "large_move_energy": pe_large,
        "tiny_changes": tiny_changes,
        "large_changes": large_changes,
    }


# =============================================================================
# Liquid-like configuration
# =============================================================================

def make_random_liquid_like_config(
    symbols,
    box_length,
    min_dist,
    seed=13,
    max_attempts_per_atom=10000,
):
    """
    Generate a periodic liquid-like random configuration with a minimum-distance
    constraint using the minimum-image convention.
    """
    rng = np.random.default_rng(seed)

    positions = []
    cell = np.eye(3) * box_length

    def minimum_image_distances(candidate, existing_positions):
        if len(existing_positions) == 0:
            return np.array([])

        dr = candidate - np.asarray(existing_positions)
        dr -= box_length * np.round(dr / box_length)

        return np.linalg.norm(dr, axis=1)

    for i, _sym in enumerate(symbols):
        placed = False

        for _ in range(max_attempts_per_atom):
            candidate = rng.uniform(0.0, box_length, size=3)
            dists = minimum_image_distances(candidate, positions)

            if len(dists) == 0 or np.all(dists >= min_dist):
                positions.append(candidate)
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place atom {i}/{len(symbols)} with min_dist={min_dist}. "
                "Try reducing min_dist or using a larger box."
            )

    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions),
        cell=cell,
        pbc=True,
    )

    return atoms


# =============================================================================
# Outlier filtering
# =============================================================================

def robust_keep_mask(values, zmax=6.0):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.asarray([], dtype=bool)

    median = np.median(values)
    abs_dev = np.abs(values - median)
    mad = np.median(abs_dev)

    if mad == 0:
        std = np.std(values)

        if std == 0:
            return np.ones_like(values, dtype=bool)

        return np.abs(values - median) <= zmax * std

    robust_z = 0.6745 * (values - median) / mad
    return np.abs(robust_z) <= zmax


def reject_timing_ratio_outliers(
    init_times,
    timing_dict,
    ratio_keys=("tinyMoveTime", "largeMoveTime"),
    zmax=6.0,
):
    """
    Reject whole runs based on timing ratios relative to initTime.

    cachedTime is usually too tiny/noisy to use as an outlier criterion, so the
    default uses tinyMoveTime/initTime and largeMoveTime/initTime.
    """
    init_times = np.asarray(init_times, dtype=float)

    if np.any(init_times <= 0):
        raise ValueError("All init times must be positive.")

    n = len(init_times)
    keep_mask = np.ones(n, dtype=bool)
    ratios = {}

    for key in ratio_keys:
        values = np.asarray(timing_dict[key], dtype=float)

        if len(values) != n:
            raise ValueError(f"{key} length does not match init_times length.")

        ratio = values / init_times
        ratios[key] = ratio
        keep_mask &= robust_keep_mask(ratio, zmax=zmax)

    return keep_mask, ratios


def mean_std_filtered(values, keep_mask):
    values = np.asarray(values, dtype=float)
    clean = values[keep_mask]

    if len(clean) == 0:
        raise RuntimeError("No values left after filtering.")

    avg = float(np.mean(clean))
    std = float(np.std(clean, ddof=1) if len(clean) > 1 else 0.0)

    return avg, std


# =============================================================================
# FCC equilibrium helpers
# =============================================================================

def cubic_cell_energy(alat, atoms, ncells_per_side):
    acell = alat[0] * ncells_per_side
    atoms.set_cell([acell, acell, acell], scale_atoms=True)
    return atoms.get_potential_energy()


def find_equilibrium_fcc(
    model: str,
    species: list,
    ncells_per_side: int = 2,
    grid_stepsize: float = 0.01,
    min_alat: float = 2.5,
    max_alat: float = 10.0,
):
    alat_ave = []

    for spec in species:
        atoms_interacting_energy, atoms_interacting_force = (
            kim_ase_utils.check_if_atoms_interacting(
                model,
                symbols=[spec, spec],
            )
        )

        if not atoms_interacting_energy:
            print(
                f"WARNING: {model} has no non-trivial energy interaction "
                f"for species {spec}. Skipping..."
            )
            continue

        if not atoms_interacting_force:
            print(
                f"WARNING: {model} has no non-trivial force interaction "
                f"for species {spec}. Skipping..."
            )
            continue

        calc = KIM(model)
        alat = min_alat
        found = False

        while alat <= max_alat:
            atoms = FaceCenteredCubic(
                size=(ncells_per_side, ncells_per_side, ncells_per_side),
                latticeconstant=alat,
                symbol=spec,
                pbc=True,
            )
            atoms.set_calculator(calc)

            try:
                res = scipy.optimize.minimize(
                    cubic_cell_energy,
                    alat,
                    args=(atoms, ncells_per_side),
                    method="Nelder-Mead",
                    tol=1e-6,
                )

                if res.success and np.isfinite(res.fun):
                    alat_ave.append(float(res.x[0]))
                    found = True
                    break

                alat += grid_stepsize

            except BaseException:
                alat += grid_stepsize

        if not found:
            print(f"WARNING: could not find FCC equilibrium alat for {model} {spec}")

    if len(alat_ave) == 0:
        return np.float64(-1.0)

    return float(np.mean(alat_ave))


def generate_alat_range(alat_eq, min_frac=0.75, max_frac=2.0, num_alats=50):
    t = np.linspace(0, 1, num_alats)
    t_squared = 1 - (1 - t) ** 2
    result = min_frac + (max_frac - min_frac) * t_squared
    return result * alat_eq


# =============================================================================
# Worker subprocess for one liquid-like lattice constant
# =============================================================================

def benchmark_one_alat_worker(
    queue,
    model,
    species,
    alat,
    alat_eq,
    i_alat,
    ncells_per_side,
    pert_amp,
    average_iterations,
    zmax,
    pert_small,
):
    """
    Worker process for one lattice constant using liquid-like configurations.

    The child process returns averaged timings for:
      - initTime
      - cachedTime
      - tinyMoveTime
      - largeMoveTime

    largeMoveTime is also returned as recalcTime for backward compatibility.
    """
    try:
        natoms = 4 * ncells_per_side**3
        box_length = ncells_per_side * alat
        min_dist = 0.5 * alat / np.sqrt(2)

        initTime_runs = []
        cachedTime_runs = []
        tinyMoveTime_runs = []
        largeMoveTime_runs = []

        initEnergy_runs = []
        cachedEnergy_runs = []
        tinyMoveEnergy_runs = []
        largeMoveEnergy_runs = []

        for t in range(average_iterations):
            seed_symbols = 1000000 * i_alat + 10 * t + 0
            seed_config = 1000000 * i_alat + 10 * t + 1
            seed_init = 1000000 * i_alat + 10 * t + 2
            seed_small = 1000000 * i_alat + 10 * t + 3
            seed_large = 1000000 * i_alat + 10 * t + 4

            # New liquid-like configuration per timing run.
            # This averages over both positional disorder and species disorder.
            rng = np.random.default_rng(seed=seed_symbols)
            symbols = rng.choice(species, size=natoms, replace=True).tolist()

            atoms = make_random_liquid_like_config(
                symbols=symbols,
                box_length=box_length,
                min_dist=min_dist,
                seed=seed_config,
            )

            base_positions = atoms.get_positions().copy()

            calc = KIM(model)
            atoms.set_calculator(calc)

            # First/cold-ish compute configuration
            kim_ase_utils.randomize_positions(
                atoms,
                pert_amp * alat,
                seed=seed_init,
            )

            pe_init, t_init = time_energy(atoms)

            initTime_runs.append(t_init)
            initEnergy_runs.append(pe_init)

            diag = neighborlist_rebuild_diagnostic(
                atoms=atoms,
                base_positions=base_positions,
                alat=alat,
                pert_small=pert_small,
                pert_large=pert_amp,
                seed_small=seed_small,
                seed_large=seed_large,
            )

            cachedTime_runs.append(diag["cached_time"])
            tinyMoveTime_runs.append(diag["tiny_move_time"])
            largeMoveTime_runs.append(diag["large_move_time"])

            cachedEnergy_runs.append(diag["cached_energy"])
            tinyMoveEnergy_runs.append(diag["tiny_move_energy"])
            largeMoveEnergy_runs.append(diag["large_move_energy"])

        timing_dict = {
            "cachedTime": cachedTime_runs,
            "tinyMoveTime": tinyMoveTime_runs,
            "largeMoveTime": largeMoveTime_runs,
        }

        keep_mask, ratios = reject_timing_ratio_outliers(
            init_times=initTime_runs,
            timing_dict=timing_dict,
            ratio_keys=("tinyMoveTime", "largeMoveTime"),
            zmax=zmax,
        )

        if np.sum(keep_mask) == 0:
            raise RuntimeError(
                f"All timing runs rejected as outliers for alat={alat}. "
                f"tiny ratios={ratios['tinyMoveTime']}, "
                f"large ratios={ratios['largeMoveTime']}"
            )

        init_avg, init_std = mean_std_filtered(initTime_runs, keep_mask)
        cached_avg, cached_std = mean_std_filtered(cachedTime_runs, keep_mask)
        tiny_avg, tiny_std = mean_std_filtered(tinyMoveTime_runs, keep_mask)
        large_avg, large_std = mean_std_filtered(largeMoveTime_runs, keep_mask)

        init_arr = np.asarray(initTime_runs, dtype=float)
        cached_arr = np.asarray(cachedTime_runs, dtype=float)
        tiny_arr = np.asarray(tinyMoveTime_runs, dtype=float)
        large_arr = np.asarray(largeMoveTime_runs, dtype=float)

        init_clean = init_arr[keep_mask]
        cached_clean = cached_arr[keep_mask]
        tiny_clean = tiny_arr[keep_mask]
        large_clean = large_arr[keep_mask]

        energy_all = np.concatenate([
            np.asarray(initEnergy_runs, dtype=float)[keep_mask],
            np.asarray(tinyMoveEnergy_runs, dtype=float)[keep_mask],
            np.asarray(largeMoveEnergy_runs, dtype=float)[keep_mask],
        ])

        density = (alat_eq / alat) ** 3

        result = {
            "ok": True,
            "kind": "ok",
            "i_alat": int(i_alat),
            "alat": float(alat),
            "density": float(density),

            "energy_AVG": float(np.mean(energy_all)),

            "initTime_AVG": init_avg,
            "initTime_STD": init_std,

            "cachedTime_AVG": cached_avg,
            "cachedTime_STD": cached_std,

            "tinyMoveTime_AVG": tiny_avg,
            "tinyMoveTime_STD": tiny_std,

            "largeMoveTime_AVG": large_avg,
            "largeMoveTime_STD": large_std,

            # Backward-compatible alias: old recalcTime means large-move recompute.
            "recalcTime_AVG": large_avg,
            "recalcTime_STD": large_std,

            "cachedOverInit_AVG": float(np.mean(cached_clean / init_clean)),
            "cachedOverInit_STD": float(
                np.std(cached_clean / init_clean, ddof=1)
                if len(init_clean) > 1 else 0.0
            ),

            "tinyMoveOverInit_AVG": float(np.mean(tiny_clean / init_clean)),
            "tinyMoveOverInit_STD": float(
                np.std(tiny_clean / init_clean, ddof=1)
                if len(init_clean) > 1 else 0.0
            ),

            "largeMoveOverInit_AVG": float(np.mean(large_clean / init_clean)),
            "largeMoveOverInit_STD": float(
                np.std(large_clean / init_clean, ddof=1)
                if len(init_clean) > 1 else 0.0
            ),

            # Backward-compatible ratio.
            "ratio_AVG": float(np.mean(large_clean / init_clean)),
            "ratio_STD": float(
                np.std(large_clean / init_clean, ddof=1)
                if len(init_clean) > 1 else 0.0
            ),

            "numRuns_KEPT": int(np.sum(keep_mask)),
            "numRuns_REMOVED": int(np.sum(~keep_mask)),

            "removed_tinyMoveOverInit": [
                float(x) for x in ratios["tinyMoveTime"][~keep_mask]
            ],
            "removed_largeMoveOverInit": [
                float(x) for x in ratios["largeMoveTime"][~keep_mask]
            ],

            "error": None,
            "traceback": None,
        }

        queue.put(result)

    except BaseException as exc:
        queue.put({
            "ok": False,
            "kind": "python_exception",
            "i_alat": int(i_alat),
            "alat": float(alat),
            "density": float((alat_eq / alat) ** 3),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })


# =============================================================================
# Parent-side subprocess runner
# =============================================================================

def run_one_alat_subprocess(
    model,
    species,
    alat,
    alat_eq,
    i_alat,
    ncells_per_side=8,
    pert_amp=0.05,
    average_iterations=10,
    zmax=6.0,
    timeout_s=120.0,
    start_method="spawn",
    pert_small=1e-5,
):
    ctx = mp.get_context(start_method)
    queue = ctx.Queue()

    proc = ctx.Process(
        target=benchmark_one_alat_worker,
        args=(
            queue,
            model,
            species,
            alat,
            alat_eq,
            i_alat,
            ncells_per_side,
            pert_amp,
            average_iterations,
            zmax,
            pert_small,
        ),
    )

    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()

        return {
            "ok": False,
            "kind": "timeout",
            "i_alat": int(i_alat),
            "alat": float(alat),
            "density": float((alat_eq / alat) ** 3),
            "exitcode": proc.exitcode,
            "error": f"Timed out after {timeout_s} seconds",
            "traceback": None,
        }

    exitcode = proc.exitcode

    if exitcode != 0:
        if exitcode == -signal.SIGSEGV:
            kind = "segfault"
        elif exitcode is not None and exitcode < 0:
            kind = f"signal_{-exitcode}"
        else:
            kind = "nonzero_exit"

        return {
            "ok": False,
            "kind": kind,
            "i_alat": int(i_alat),
            "alat": float(alat),
            "density": float((alat_eq / alat) ** 3),
            "exitcode": exitcode,
            "error": f"Child exited with code {exitcode}",
            "traceback": None,
        }

    if queue.empty():
        return {
            "ok": False,
            "kind": "no_result",
            "i_alat": int(i_alat),
            "alat": float(alat),
            "density": float((alat_eq / alat) ** 3),
            "exitcode": exitcode,
            "error": "Child exited cleanly but returned no result",
            "traceback": None,
        }

    result = queue.get()
    result["exitcode"] = exitcode
    return result


# =============================================================================
# Benchmark driver
# =============================================================================

def do_bench(model: str, species: list, alat_eq: float):
    alat_range = generate_alat_range(alat_eq)

    print(f"\tscanning range {alat_range}")

    ncells_per_side = 8
    pert_amp = 0.05
    pert_small = 1e-5
    average_iterations = 10
    zmax = 6.0
    timeout_s = 120.0

    results = []

    for i_alat, alat in enumerate(alat_range):
        print(f"\t\tSTART alat={alat:.6f} [{i_alat + 1}/{len(alat_range)}]")

        result = run_one_alat_subprocess(
            model=model,
            species=species,
            alat=float(alat),
            alat_eq=float(alat_eq),
            i_alat=i_alat,
            ncells_per_side=ncells_per_side,
            pert_amp=pert_amp,
            pert_small=pert_small,
            average_iterations=average_iterations,
            zmax=zmax,
            timeout_s=timeout_s,
            start_method="spawn",
        )

        if not result["ok"]:
            print(f"\t\tFAILED alat={alat:.6f}")
            print(f"\t\tkind: {result.get('kind')}")
            print(f"\t\texitcode: {result.get('exitcode')}")
            print(f"\t\terror: {result.get('error')}")

            if result.get("traceback"):
                print(result["traceback"])

            results.append(result)
            continue

        print(
            f"\t\tDONE alat={alat:.6f} "
            f"cached/init={result['cachedOverInit_AVG']:.4f} "
            f"tiny/init={result['tinyMoveOverInit_AVG']:.4f} "
            f"large/init={result['largeMoveOverInit_AVG']:.4f} "
            f"kept={result['numRuns_KEPT']}/{average_iterations}"
        )

        results.append(result)

    ok_results = [r for r in results if r["ok"]]
    ok_results = sorted(ok_results, key=lambda r: r["i_alat"])

    model_bench_results = {
        "density": [r["density"] for r in ok_results],
        "alat": [r["alat"] for r in ok_results],
        "energy": [r["energy_AVG"] for r in ok_results],

        "initTime_AVG": [r["initTime_AVG"] for r in ok_results],
        "initTime_STD": [r["initTime_STD"] for r in ok_results],

        "cachedTime_AVG": [r["cachedTime_AVG"] for r in ok_results],
        "cachedTime_STD": [r["cachedTime_STD"] for r in ok_results],

        "tinyMoveTime_AVG": [r["tinyMoveTime_AVG"] for r in ok_results],
        "tinyMoveTime_STD": [r["tinyMoveTime_STD"] for r in ok_results],

        "largeMoveTime_AVG": [r["largeMoveTime_AVG"] for r in ok_results],
        "largeMoveTime_STD": [r["largeMoveTime_STD"] for r in ok_results],

        # Backward-compatible alias for old plotting code.
        "recalcTime_AVG": [r["recalcTime_AVG"] for r in ok_results],
        "recalcTime_STD": [r["recalcTime_STD"] for r in ok_results],

        "cachedOverInit_AVG": [r["cachedOverInit_AVG"] for r in ok_results],
        "cachedOverInit_STD": [r["cachedOverInit_STD"] for r in ok_results],

        "tinyMoveOverInit_AVG": [r["tinyMoveOverInit_AVG"] for r in ok_results],
        "tinyMoveOverInit_STD": [r["tinyMoveOverInit_STD"] for r in ok_results],

        "largeMoveOverInit_AVG": [r["largeMoveOverInit_AVG"] for r in ok_results],
        "largeMoveOverInit_STD": [r["largeMoveOverInit_STD"] for r in ok_results],

        # Backward-compatible ratio.
        "ratio_AVG": [r["ratio_AVG"] for r in ok_results],
        "ratio_STD": [r["ratio_STD"] for r in ok_results],

        "numRuns_KEPT": [r["numRuns_KEPT"] for r in ok_results],
        "numRuns_REMOVED": [r["numRuns_REMOVED"] for r in ok_results],

        "failed": [
            {
                "i_alat": r["i_alat"],
                "alat": r["alat"],
                "density": r["density"],
                "kind": r.get("kind"),
                "exitcode": r.get("exitcode"),
                "error": r.get("error"),
            }
            for r in results
            if not r["ok"]
        ],
    }

    return model_bench_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-module", required=True)
    args = parser.parse_args()

    input_module = importlib.import_module(args.input_module)
    input_data = input_module.input_data

    output_dir = Path("liquid")
    output_dir.mkdir(parents=True, exist_ok=True)

    for data in input_data:
        model = data["model"]
        species = data["species"]
        model_shortname = data["model_shortname"]

        print(f"MODEL : {model_shortname}")

        mixed_alat_eq = find_equilibrium_fcc(model, species)
        print(f"\tEQUIL_ALAT : {mixed_alat_eq}")

        model_bench = do_bench(model, species, mixed_alat_eq)
        print(f"\tDONE BENCH {model}")

        lj_model = "LennardJones612_UniversalShifted__MO_959249795837_003"
        lj_bench = do_bench(lj_model, species, mixed_alat_eq)
        print("\tDONE BENCH LJ")

        plotdata = [
            {
                "model": model,
                "model_bench": model_bench,
            },
            {
                "model": lj_model,
                "lj_bench": lj_bench,
            },
        ]

        out_path = output_dir / f"{model}_LJ_baseline_{'-'.join(species)}_liquid.json"

        with open(out_path, "w") as file:
            json.dump(plotdata, file, indent=4)

        print(f"\tWROTE {out_path}")


if __name__ == "__main__":
    main()