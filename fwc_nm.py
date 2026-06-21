from kim_tools.ase import *
from typing import Union

import logging
import math
import multiprocessing as mp
import signal
import traceback

import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.kim.kim import KIM
from ase.data import atomic_numbers, covalent_radii
from ase.lattice.cubic import FaceCenteredCubic

import kimpy


FWC_NCELLS_PER_SIDE = 1


def _energy_worker(model_name: str, species: str, alat: float, queue):
    """
    Worker process. This process is allowed to crash; the parent process survives.
    """

    try:
        energy_config = generate_fcc_compute_energy(
            model=model_name,
            species=species,
            alat=alat,
        )

        if energy_config is None:
            queue.put(
                {
                    "ok": False,
                    "energy": None,
                    "ncells": None,
                    "error": None,
                }
            )
        else:
            pe, ncells = energy_config
            queue.put(
                {
                    "ok": True,
                    "energy": pe,
                    "ncells": ncells,
                    "error": None,
                }
            )

    except RuntimeError as e:
        queue.put(
            {
                "ok": False,
                "energy": None,
                "ncells": None,
                "error": e,
            }
        )
    except Exception:
        queue.put(
            {
                "ok": False,
                "energy": None,
                "ncells": None,
                "error": traceback.format_exc(),
            }
        )


def generate_fcc_compute_energy_safe(
    model: str,
    species: str,
    alat: float,
    timeout: float = 600.0,
):
    """
    Run generate_fcc_compute_energy in a child process so segfaults do not kill
    the parent process.

    Returns:
        (energy, ncells) on success
        None on Python exception, timeout, or segfault
    """

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    proc = ctx.Process(
        target=_energy_worker,
        args=(model, species, alat, queue),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"\t\tTIMEOUT at alat = {alat}")
        return None

    exitcode = proc.exitcode

    if exitcode != 0:
        if exitcode < 0:
            sig = -exitcode
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = f"signal {sig}"

            print(f"\t\tCRASH at alat = {alat}: child died with {sig_name}")
        else:
            print(f"\t\tFAIL at alat = {alat}: child exit code {exitcode}")

        return None

    if queue.empty():
        print(f"\t\tFAIL at alat = {alat}: child exited but returned no result")
        return None

    result = queue.get()

    if result["ok"] is False:
        print(f"\t\tPYTHON ERROR at alat = {alat}")
        print(result["error"])
        return None

    return result["energy"], result["ncells"]


def generate_fcc_compute_energy(
    model: Union[str, Calculator],
    species: str,
    alat: float,
) -> Union[tuple[float, int], None]:
    """
    Construct an FCC lattice, evaluate its energy, and return total energy and
    supercell size. Returns None if the model fails to compute the energy.
    """

    ncells_per_side = FWC_NCELLS_PER_SIDE
    atoms = FaceCenteredCubic(
        size=(ncells_per_side, ncells_per_side, ncells_per_side),
        latticeconstant=alat,
        symbol=species,
        pbc=True,
    )

    if isinstance(model, str):
        calc = KIM(model)
    else:
        calc = model

    atoms.set_calculator(calc)

    try:
        pe = atoms.get_potential_energy()

        if hasattr(calc, "clean"):
            calc.clean()
        if hasattr(calc, "__del__"):
            calc.__del__()

        return pe, ncells_per_side

    except Exception as e:
        print(e)
        return None


def fcc_atoms_in_supercell(n_cells_per_side: int) -> int:
    """Return the number of atoms in an FCC supercell."""

    atoms_per_unit_cell = 4
    total_unit_cells = n_cells_per_side**3
    return int(atoms_per_unit_cell * total_unit_cells)


def _validate_led_order(order: int) -> None:
    supported_orders = {3, 5, 7, 9}
    if order not in supported_orders:
        raise ValueError(
            f"Unsupported LED order {order}. "
            f"Supported orders are {sorted(supported_orders)}."
        )


def _led_start_index(order: int) -> int:
    _validate_led_order(order)
    return (order - 1) // 2


def _led_stop_index_exclusive(n_points: int, order: int) -> int:
    _validate_led_order(order)
    return n_points - ((order + 1) // 2)


def _led_coefficients(order: int) -> list[float]:
    _validate_led_order(order)
    normalization = float(order + 1)
    return [
        ((-1) ** (k + 1)) * math.comb(order, k) / normalization
        for k in range(order + 1)
    ]


def local_edge_detection(x: list, y: list, order: int = 5) -> list:
    """
    Compute Local Edge Detection values for the x-y curve.

    For a given order, LED[k] corresponds to the original point
    x[k + (order - 1)//2].
    """

    _validate_led_order(order)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    led_values = []
    n_points = len(y)
    start_idx = _led_start_index(order)
    stop_idx = _led_stop_index_exclusive(n_points, order)

    if stop_idx <= start_idx:
        return led_values

    coeffs = _led_coefficients(order)
    left_radius = (order - 1) // 2

    for j in range(start_idx, stop_idx):
        stencil_start = j - left_radius
        led = 0.0
        for k, coeff in enumerate(coeffs):
            led += coeff * y[stencil_start + k]
        led_values.append(led)

    return led_values


def filter_good_alat(
    alats: list,
    energies_per_atom: list,
    leds: list,
    etol: list = [5e-2, 5e2],
    led_tol: float = 1.0,
    led_order: int = 5,
) -> dict:
    """
    Filter a good lattice constant based on energy and LED criteria.

    If an edge is detected at original index i, discard i - led_order through
    i + led_order, inclusive.
    """

    _validate_led_order(led_order)

    N = len(alats)
    if len(energies_per_atom) != N:
        raise ValueError("alats and energies_per_atom must have the same length")

    start_idx = _led_start_index(led_order)
    end_idx = _led_stop_index_exclusive(N, led_order)

    expected_led_count = max(0, end_idx - start_idx)
    if len(leds) != expected_led_count:
        raise ValueError(
            f"Expected {expected_led_count} LED values for N={N} and "
            f"led_order={led_order}, but got {len(leds)}."
        )

    edge_indices = []
    discarded_indices = set()

    for led_idx, led in enumerate(leds):
        original_idx = start_idx + led_idx
        if abs(led) > led_tol:
            edge_indices.append(original_idx)
            discard_start = max(0, original_idx - led_order)
            discard_stop = min(N - 1, original_idx + led_order)
            discarded_indices.update(range(discard_start, discard_stop + 1))

    valid_energy_per_atom = []
    valid_leds = []
    valid_alats = []
    valid_original_indices = []

    for i in range(start_idx, end_idx):
        if i in discarded_indices:
            continue

        alat = alats[i]
        energy_per_atom = energies_per_atom[i]
        led = leds[i - start_idx]

        if abs(energy_per_atom) > etol[1] or abs(energy_per_atom) < etol[0]:
            continue

        valid_leds.append(led)
        valid_energy_per_atom.append(energy_per_atom)
        valid_alats.append(alat)
        valid_original_indices.append(i)

    if len(valid_alats) == 0:
        return _failure_result(
            all_alats=alats,
            all_energies_per_atom=energies_per_atom,
            led_order=led_order,
            edge_indices=edge_indices,
            discarded_indices=sorted(discarded_indices),
            search_strategy="edge_detection",
        )

    indices = _local_minima_indices(valid_alats, valid_energy_per_atom)
    min_index = min(indices, key=lambda i: valid_energy_per_atom[i])

    return {
        "good_alat": valid_alats[min_index],
        "min_led": valid_leds[min_index],
        "good_ncells": FWC_NCELLS_PER_SIDE,
        "valid_alats": valid_alats,
        "valid_energy_per_atom": valid_energy_per_atom,
        "indices": indices,
        "min_index": min_index,
        "min_original_index": valid_original_indices[min_index],
        "all_alats": alats,
        "all_energies_per_atom": energies_per_atom,
        "led_order": led_order,
        "edge_indices": edge_indices,
        "discarded_indices": sorted(discarded_indices),
        "valid_original_indices": valid_original_indices,
        "search_strategy": "edge_detection",
    }


def query_kim_influence_distance(model_name: str) -> float:
    """Return the KIM model influence distance."""

    units_accepted, kim_model = kimpy.model.create(
        kimpy.numbering.zeroBased,
        kimpy.length_unit.A,
        kimpy.energy_unit.eV,
        kimpy.charge_unit.e,
        kimpy.temperature_unit.K,
        kimpy.time_unit.ps,
        model_name,
    )

    try:
        return float(kim_model.get_influence_distance())
    finally:
        if hasattr(kim_model, "destroy"):
            kim_model.destroy()


def energy_plateau_detected(
    alats,
    energies_per_atom,
    window=8,
    slope_tol=1e-3,
    range_tol=1e-3,
):
    """
    Detect whether the last window of energy-per-atom values has plateaued.
    """

    if len(energies_per_atom) < window:
        return False

    x = np.array(alats[-window:], dtype=float)
    y = np.array(energies_per_atom[-window:], dtype=float)

    if not np.all(np.isfinite(y)):
        return False

    recent_range = np.max(y) - np.min(y)
    slope, intercept = np.polyfit(x, y, deg=1)

    return abs(slope) < slope_tol and recent_range < range_tol


def _failure_result(
    all_alats=None,
    all_energies_per_atom=None,
    led_order=5,
    edge_indices=None,
    discarded_indices=None,
    search_strategy="failed",
    nelder_mead_attempts=None,
    fallback_reason=None,
):
    return {
        "good_alat": -1.0,
        "min_led": -1.0,
        "good_ncells": -1,
        "valid_alats": [-1.0],
        "valid_energy_per_atom": [0],
        "indices": [0],
        "min_index": 0,
        "all_alats": [] if all_alats is None else all_alats,
        "all_energies_per_atom": (
            [] if all_energies_per_atom is None else all_energies_per_atom
        ),
        "led_order": led_order,
        "edge_indices": [] if edge_indices is None else edge_indices,
        "discarded_indices": (
            [] if discarded_indices is None else discarded_indices
        ),
        "valid_original_indices": [],
        "search_strategy": search_strategy,
        "nelder_mead_attempts": (
            [] if nelder_mead_attempts is None else nelder_mead_attempts
        ),
        "fallback_reason": fallback_reason,
    }


def _round_alat(alat: float) -> float:
    return round(float(alat), 8)


def _compute_energy_per_atom_cached(
    model,
    species: str,
    alat: float,
    energy_cache: dict,
    use_safe: bool,
    timeout: float = 300.0,
):
    cache_key = _round_alat(alat)
    if cache_key in energy_cache:
        return energy_cache[cache_key]

    if use_safe:
        if not isinstance(model, str):
            raise ValueError(
                "Safe subprocess execution requires model to be a KIM model name "
                "string. Pass use_safe=False when using a Calculator object."
            )
        val = generate_fcc_compute_energy_safe(
            model=model,
            species=species,
            alat=float(alat),
            timeout=timeout,
        )
    else:
        val = generate_fcc_compute_energy(
            model=model,
            species=species,
            alat=float(alat),
        )

    if val is None:
        energy_cache[cache_key] = None
        return None

    energy_total, ncell = val
    natoms = fcc_atoms_in_supercell(ncell)
    energy_pa = energy_total / natoms
    result = {
        "alat": float(alat),
        "energy_total": float(energy_total),
        "energy_per_atom": float(energy_pa),
        "ncells": int(ncell),
    }
    energy_cache[cache_key] = result
    return result


def _scan_alat_range(
    model,
    species: str,
    a_min: float,
    a_max: float,
    del_a: float,
    energy_cache: dict,
    use_safe: bool,
    timeout: float = 300.0,
    switch_to_unsafe_after_angstrom: Union[float, None] = None,
    early_stop_plateau: bool = False,
    min_scan_alat: float = 6.5,
    plateau_window: int = 20,
    plateau_slope_tol: float = 1e-3,
    plateau_range_tol: float = 5e-4,
):
    """
    Scan an alat range and return successful alats/energies/ncells.

    If switch_to_unsafe_after_angstrom is set, safe execution is used until the
    requested physical width has produced consecutive successful evaluations.
    For del_a=0.1 and switch_to_unsafe_after_angstrom=1.0, this means ten
    consecutive successful safe points.
    """

    a_min = float(a_min)
    a_max = float(a_max)
    del_a = float(del_a)

    if a_max < a_min:
        return {
            "alats": [],
            "energies_per_atom": [],
            "ncells": [],
            "switched_to_unsafe": False,
            "switch_alat": None,
        }

    na = int(math.floor((a_max - a_min) / del_a + 1e-9))

    alats = []
    energies_per_atom = []
    ncells = []

    current_use_safe = bool(use_safe)
    switched_to_unsafe = False
    switch_alat = None
    consecutive_safe_successes = 0

    if switch_to_unsafe_after_angstrom is None:
        successes_needed = None
    else:
        successes_needed = max(
            1,
            int(math.ceil(float(switch_to_unsafe_after_angstrom) / del_a)),
        )

    for j in range(na + 1):
        a = a_min + j * del_a
        a = _round_alat(a)

        try:
            result = _compute_energy_per_atom_cached(
                model=model,
                species=species,
                alat=a,
                energy_cache=energy_cache,
                use_safe=current_use_safe,
                timeout=timeout,
            )
        except Exception as e:
            print("gen-fcc exception", e)
            result = None

        if result is None:
            if current_use_safe:
                consecutive_safe_successes = 0
            continue

        alats.append(result["alat"])
        energies_per_atom.append(result["energy_per_atom"])
        ncells.append(result["ncells"])

        print(
            f"\t\talat = {result['alat']} | "
            f"energy/atom = {result['energy_per_atom']}"
        )

        if current_use_safe:
            consecutive_safe_successes += 1
            if (
                successes_needed is not None
                and consecutive_safe_successes >= successes_needed
            ):
                current_use_safe = False
                switched_to_unsafe = True
                switch_alat = result["alat"]
                print(
                    "\t\tSwitching to direct execution after "
                    f"{consecutive_safe_successes} consecutive safe successes"
                )

        if (
            early_stop_plateau
            and result["alat"] > min_scan_alat
            and energy_plateau_detected(
                alats,
                energies_per_atom,
                window=plateau_window,
                slope_tol=plateau_slope_tol,
                range_tol=plateau_range_tol,
            )
        ):
            print(
                "\t\tEarly stopping: energy plateau detected near "
                f"alat = {result['alat']}"
            )
            break

    return {
        "alats": alats,
        "energies_per_atom": energies_per_atom,
        "ncells": ncells,
        "switched_to_unsafe": switched_to_unsafe,
        "switch_alat": switch_alat,
    }


def _local_minima_indices(alats, energies_per_atom):
    from scipy.signal import find_peaks

    y = np.asarray(energies_per_atom, dtype=float)
    finite = np.isfinite(y)

    if len(y) == 0:
        return []

    if np.sum(finite) == 0:
        return []

    indices, _ = find_peaks(-y)
    indices = [int(i) for i in indices if finite[i]]

    if len(indices) == 0:
        indices = [int(np.nanargmin(y))]

    return indices


def _coarse_starting_points(
    alats,
    energies_per_atom,
    max_starting_points: int = 6,
):
    minima_indices = _local_minima_indices(alats, energies_per_atom)
    minima_indices = sorted(
        minima_indices,
        key=lambda i: energies_per_atom[i],
    )
    minima_indices = minima_indices[:max_starting_points]

    return [
        {
            "index": int(i),
            "alat": float(alats[i]),
            "energy_per_atom": float(energies_per_atom[i]),
        }
        for i in minima_indices
    ]


def _nelder_mead_worker(
    model_name: str,
    species: str,
    start_alat: float,
    amin: float,
    amax: float,
    energy_bound: list,
    queue,
):
    """
    Run Nelder-Mead in a child process. Crashes and optimizer hangs are contained
    by the parent-side timeout.
    """

    try:
        from scipy.optimize import minimize

        evaluations = []

        def objective(x):
            alat = float(np.ravel(x)[0])

            if not np.isfinite(alat) or alat < amin or alat > amax:
                return 1.0e100

            val = generate_fcc_compute_energy(
                model=model_name,
                species=species,
                alat=alat,
            )

            if val is None:
                return 1.0e100

            energy_total, ncell = val
            energy_pa = float(energy_total) / fcc_atoms_in_supercell(ncell)

            if (
                not np.isfinite(energy_pa)
                or abs(energy_pa) > energy_bound[1]
                or abs(energy_pa) < energy_bound[0]
            ):
                return 1.0e100

            evaluations.append(
                {
                    "alat": alat,
                    "energy_per_atom": energy_pa,
                    "ncells": int(ncell),
                }
            )
            return energy_pa

        result = minimize(
            objective,
            x0=np.array([float(start_alat)]),
            method="Nelder-Mead",
            options={
                "xatol": 1.0e-4,
                "fatol": 1.0e-8,
                "maxiter": 80,
                "maxfev": 160,
                "disp": False,
            },
        )

        if len(evaluations) == 0:
            queue.put(
                {
                    "ok": False,
                    "status": "no_valid_evaluations",
                    "message": str(result.message),
                    "start_alat": float(start_alat),
                    "evaluations": evaluations,
                }
            )
            return

        best_eval = min(evaluations, key=lambda row: row["energy_per_atom"])
        best_alat = float(best_eval["alat"])
        best_energy_pa = float(best_eval["energy_per_atom"])

        if not result.success:
            queue.put(
                {
                    "ok": False,
                    "status": "optimizer_unsuccessful",
                    "message": str(result.message),
                    "start_alat": float(start_alat),
                    "best_alat": best_alat,
                    "best_energy_per_atom": best_energy_pa,
                    "evaluations": evaluations,
                }
            )
            return

        queue.put(
            {
                "ok": True,
                "status": "success",
                "message": str(result.message),
                "start_alat": float(start_alat),
                "good_alat": best_alat,
                "good_energy_per_atom": best_energy_pa,
                "good_ncells": int(best_eval["ncells"]),
                "optimizer_x": float(np.ravel(result.x)[0]),
                "optimizer_fun": float(result.fun),
                "nfev": int(result.nfev),
                "nit": int(result.nit),
                "evaluations": evaluations,
            }
        )

    except Exception:
        queue.put(
            {
                "ok": False,
                "status": "python_exception",
                "message": traceback.format_exc(),
                "start_alat": float(start_alat),
            }
        )


def scipy_nelder_mead_safe(
    model: str,
    species: str,
    start_alat: float,
    amin: float,
    amax: float,
    energy_bound: list,
    timeout: float = 600.0,
):
    """
    Run Nelder-Mead in a subprocess. Returns a diagnostic dictionary.
    """

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    proc = ctx.Process(
        target=_nelder_mead_worker,
        args=(model, species, start_alat, amin, amax, energy_bound, queue),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "ok": False,
            "status": "timeout",
            "message": f"Nelder-Mead timed out after {timeout} seconds",
            "start_alat": float(start_alat),
        }

    exitcode = proc.exitcode
    if exitcode != 0:
        if exitcode < 0:
            sig = -exitcode
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = f"signal {sig}"
            message = f"child died with {sig_name}"
        else:
            message = f"child exit code {exitcode}"

        return {
            "ok": False,
            "status": "crash",
            "message": message,
            "start_alat": float(start_alat),
        }

    if queue.empty():
        return {
            "ok": False,
            "status": "no_result",
            "message": "child exited but returned no result",
            "start_alat": float(start_alat),
        }

    return queue.get()


def _nelder_mead_failure_allows_led_fallback(attempt: dict) -> bool:
    """
    Return True only for failure modes where the Nelder-Mead subprocess itself
    failed to complete cleanly.

    This keeps the LED path as a crash/timeout fallback, not a replacement for a
    normal but unsuccessful optimizer result.
    """

    return attempt.get("status") in {
        "timeout",
        "crash",
        "no_result",
        "python_exception",
    }


def _central_second_derivative(alats, energies_per_atom, index):
    if index <= 0 or index >= len(alats) - 1:
        return None

    x0 = float(alats[index - 1])
    x1 = float(alats[index])
    x2 = float(alats[index + 1])
    y0 = float(energies_per_atom[index - 1])
    y1 = float(energies_per_atom[index])
    y2 = float(energies_per_atom[index + 1])

    h_left = x1 - x0
    h_right = x2 - x1

    if h_left <= 0 or h_right <= 0:
        return None

    return 2.0 * (
        y0 / (h_left * (h_left + h_right))
        - y1 / (h_left * h_right)
        + y2 / (h_right * (h_left + h_right))
    )


def _window_is_continuous_by_led(
    alats,
    energies_per_atom,
    led_tol: float,
    led_order: int,
):
    leds = local_edge_detection(alats, energies_per_atom, order=led_order)
    start_idx = _led_start_index(led_order)

    edge_indices = [
        start_idx + led_idx
        for led_idx, led in enumerate(leds)
        if abs(led) > led_tol
    ]

    return len(edge_indices) == 0, leds, edge_indices


def _edge_detection_fallback_from_starting_points(
    model,
    species: str,
    starting_points: list,
    amin: float,
    amax: float,
    energy_cache: dict,
    energy_bound: list,
    led_tol: float,
    led_order: int,
):
    """
    Reuse coarse starting points for the multi-resolution edge-detection fallback:
        level 1: del_a=0.01 over +/-0.5 A
        level 2: del_a=0.001 over +/-0.05 A around level-1 minima
        LED on level-2 windows
        choose accepted local minimum with largest positive second derivative
    """

    accepted_candidates = []
    rejected_candidates = []
    level1_min_alats = []

    all_level2_alats = []
    all_level2_energies = []

    for start in starting_points:
        a0 = float(start["alat"])

        level1 = _scan_alat_range(
            model=model,
            species=species,
            a_min=max(amin, a0 - 0.5),
            a_max=min(amax, a0 + 0.5),
            del_a=0.01,
            energy_cache=energy_cache,
            use_safe=False,
        )

        l1_min_indices = _local_minima_indices(
            level1["alats"],
            level1["energies_per_atom"],
        )

        for l1_idx in l1_min_indices:
            l1_alat = float(level1["alats"][l1_idx])
            level1_min_alats.append(l1_alat)

            level2 = _scan_alat_range(
                model=model,
                species=species,
                a_min=max(amin, l1_alat - 0.05),
                a_max=min(amax, l1_alat + 0.05),
                del_a=0.001,
                energy_cache=energy_cache,
                use_safe=False,
            )

            all_level2_alats.extend(level2["alats"])
            all_level2_energies.extend(level2["energies_per_atom"])

            if len(level2["alats"]) < led_order + 1:
                rejected_candidates.append(
                    {
                        "coarse_start_alat": a0,
                        "level1_alat": l1_alat,
                        "reason": "too_few_level2_points",
                    }
                )
                continue

            continuous, leds, edge_indices = _window_is_continuous_by_led(
                level2["alats"],
                level2["energies_per_atom"],
                led_tol=led_tol,
                led_order=led_order,
            )

            l2_min_indices = _local_minima_indices(
                level2["alats"],
                level2["energies_per_atom"],
            )
            l2_min_idx = min(
                l2_min_indices,
                key=lambda i: level2["energies_per_atom"][i],
            )

            curvature = _central_second_derivative(
                level2["alats"],
                level2["energies_per_atom"],
                l2_min_idx,
            )

            candidate = {
                "coarse_start_alat": a0,
                "level1_alat": l1_alat,
                "alat": float(level2["alats"][l2_min_idx]),
                "energy_per_atom": float(level2["energies_per_atom"][l2_min_idx]),
                "curvature": None if curvature is None else float(curvature),
                "leds": leds,
                "edge_indices": edge_indices,
                "level2_alats": level2["alats"],
                "level2_energies_per_atom": level2["energies_per_atom"],
            }

            if not continuous:
                candidate["reason"] = "led_discontinuity"
                rejected_candidates.append(candidate)
                continue

            if curvature is None or curvature <= 0.0:
                candidate["reason"] = "non_positive_curvature"
                rejected_candidates.append(candidate)
                continue

            energy_pa = candidate["energy_per_atom"]
            if abs(energy_pa) > energy_bound[1] or abs(energy_pa) < energy_bound[0]:
                candidate["reason"] = "energy_out_of_bounds"
                rejected_candidates.append(candidate)
                continue

            accepted_candidates.append(candidate)

    if len(accepted_candidates) == 0:
        return _failure_result(
            all_alats=all_level2_alats,
            all_energies_per_atom=all_level2_energies,
            led_order=led_order,
            search_strategy="edge_detection_fallback_failed",
            fallback_reason="no_continuous_positive_curvature_candidate",
        ) | {
            "accepted_candidates": accepted_candidates,
            "rejected_candidates": rejected_candidates,
            "level1_min_alats": level1_min_alats,
        }

    best = max(accepted_candidates, key=lambda row: row["curvature"])

    return {
        "good_alat": best["alat"],
        "good_energy_per_atom": best["energy_per_atom"],
        "good_curvature": best["curvature"],
        "min_led": 0.0,
        "good_ncells": FWC_NCELLS_PER_SIDE,
        "valid_alats": [row["alat"] for row in accepted_candidates],
        "valid_energy_per_atom": [
            row["energy_per_atom"] for row in accepted_candidates
        ],
        "indices": list(range(len(accepted_candidates))),
        "min_index": accepted_candidates.index(best),
        "all_alats": all_level2_alats,
        "all_energies_per_atom": all_level2_energies,
        "led_order": led_order,
        "edge_indices": best["edge_indices"],
        "discarded_indices": [],
        "valid_original_indices": [],
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
        "level1_min_alats": level1_min_alats,
        "search_strategy": "edge_detection_fallback",
    }


def _bounds_for_species(model, species: str):
    cov = covalent_radii[atomic_numbers[species]]
    amin = max(np.sqrt(2) * cov, 1.5)
    amax = 12.0
    min_scan_alat = 6.5

    if isinstance(model, str) and model[:3] != "Sim":
        min_cutoff = query_kim_influence_distance(model)
        if min_cutoff > amax:
            amax = 2.0 * min_cutoff
            min_scan_alat = min_cutoff

    return amin, amax, min_scan_alat


def find_working_configuration_FCC(
    model: Union[str, Calculator],
    species: str,
    energy_bound: list = [5e-2, 5e2],
    led_tol: float = 1.0,
    led_order: int = 5,
    nelder_mead_timeout: float = 600.0,
    max_starting_points: int = 6,
) -> dict:
    """
    Find an FCC configuration using Nelder-Mead as the primary minimizer.

    Procedure:
        1. Coarse scan with del_a=0.1 over [amin, amax].
        2. Use scipy.signal.find_peaks on -E(a) to identify coarse local minima.
        3. Run Nelder-Mead in a subprocess from each coarse minimum.
        4. Return the first successful Nelder-Mead minimum.
        5. If all Nelder-Mead attempts crash, time out, or fail, reuse the same
           coarse starting points in the multi-resolution LED fallback.
    """

    _validate_led_order(led_order)

    amin, amax, min_scan_alat = _bounds_for_species(model, species)

    plateau_detection_window = 20
    plateau_detection_slope = 1e-3
    plateau_detection_range = 5e-4

    energy_cache = {}

    coarse_scan = _scan_alat_range(
        model=model,
        species=species,
        a_min=amin,
        a_max=amax,
        del_a=0.1,
        energy_cache=energy_cache,
        use_safe=isinstance(model, str),
        timeout=600.0,
        switch_to_unsafe_after_angstrom=1.0 if isinstance(model, str) else None,
        early_stop_plateau=True,
        min_scan_alat=min_scan_alat,
        plateau_window=plateau_detection_window,
        plateau_slope_tol=plateau_detection_slope,
        plateau_range_tol=plateau_detection_range,
    )

    if len(coarse_scan["alats"]) == 0:
        return _failure_result(
            led_order=led_order,
            search_strategy="failed",
            fallback_reason="coarse_scan_has_no_successful_points",
        )

    starting_points = _coarse_starting_points(
        coarse_scan["alats"],
        coarse_scan["energies_per_atom"],
        max_starting_points=max_starting_points,
    )

    nelder_mead_attempts = []

    if isinstance(model, str):
        for start in starting_points:
            attempt = scipy_nelder_mead_safe(
                model=model,
                species=species,
                start_alat=start["alat"],
                amin=amin,
                amax=amax,
                energy_bound=energy_bound,
                timeout=nelder_mead_timeout,
            )
            nelder_mead_attempts.append(attempt)

            if attempt.get("ok", False):
                evaluations = attempt.get("evaluations", [])
                return {
                    "good_alat": attempt["good_alat"],
                    "good_energy_per_atom": attempt["good_energy_per_atom"],
                    "min_led": None,
                    "good_ncells": attempt["good_ncells"],
                    "valid_alats": [attempt["good_alat"]],
                    "valid_energy_per_atom": [
                        attempt["good_energy_per_atom"]
                    ],
                    "indices": [0],
                    "min_index": 0,
                    "all_alats": [row["alat"] for row in evaluations],
                    "all_energies_per_atom": [
                        row["energy_per_atom"] for row in evaluations
                    ],
                    "led_order": led_order,
                    "edge_indices": [],
                    "discarded_indices": [],
                    "valid_original_indices": [],
                    "search_strategy": "nelder_mead",
                    "coarse_min_alats": [
                        row["alat"] for row in starting_points
                    ],
                    "coarse_scan_alats": coarse_scan["alats"],
                    "coarse_scan_energies_per_atom": coarse_scan[
                        "energies_per_atom"
                    ],
                    "coarse_safe_to_direct_switch": {
                        "enabled": True,
                        "after_consecutive_safe_angstrom": 1.0,
                        "switched_to_unsafe": coarse_scan[
                            "switched_to_unsafe"
                        ],
                        "switch_alat": coarse_scan["switch_alat"],
                    },
                    "nelder_mead_attempts": nelder_mead_attempts,
                }

    else:
        nelder_mead_attempts.append(
            {
                "ok": False,
                "status": "skipped",
                "message": (
                    "Subprocess Nelder-Mead is skipped for Calculator objects; "
                    "safe subprocess execution requires a model name string."
                ),
            }
        )

    fallback_allowed = (
        len(nelder_mead_attempts) > 0
        and all(
            _nelder_mead_failure_allows_led_fallback(attempt)
            for attempt in nelder_mead_attempts
        )
    )

    if not fallback_allowed:
        return _failure_result(
            all_alats=coarse_scan["alats"],
            all_energies_per_atom=coarse_scan["energies_per_atom"],
            led_order=led_order,
            search_strategy="nelder_mead_failed_no_led_fallback",
            nelder_mead_attempts=nelder_mead_attempts,
            fallback_reason=(
                "LED fallback is only used when all Nelder-Mead subprocess "
                "attempts crash, time out, raise, or return no result."
            ),
        ) | {
            "coarse_min_alats": [row["alat"] for row in starting_points],
            "coarse_scan_alats": coarse_scan["alats"],
            "coarse_scan_energies_per_atom": coarse_scan[
                "energies_per_atom"
            ],
            "coarse_safe_to_direct_switch": {
                "enabled": isinstance(model, str),
                "after_consecutive_safe_angstrom": 1.0,
                "switched_to_unsafe": coarse_scan["switched_to_unsafe"],
                "switch_alat": coarse_scan["switch_alat"],
            },
        }

    fallback = _edge_detection_fallback_from_starting_points(
        model=model,
        species=species,
        starting_points=starting_points,
        amin=amin,
        amax=amax,
        energy_cache=energy_cache,
        energy_bound=energy_bound,
        led_tol=led_tol,
        led_order=led_order,
    )

    fallback["fallback_reason"] = "all_nelder_mead_attempts_failed"
    fallback["nelder_mead_attempts"] = nelder_mead_attempts
    fallback["coarse_min_alats"] = [row["alat"] for row in starting_points]
    fallback["coarse_scan_alats"] = coarse_scan["alats"]
    fallback["coarse_scan_energies_per_atom"] = coarse_scan["energies_per_atom"]
    fallback["coarse_safe_to_direct_switch"] = {
        "enabled": isinstance(model, str),
        "after_consecutive_safe_angstrom": 1.0,
        "switched_to_unsafe": coarse_scan["switched_to_unsafe"],
        "switch_alat": coarse_scan["switch_alat"],
    }

    return fallback


def find_working_configurations_FCC(
    model: Union[str, Calculator],
    species_list: list[str],
    energy_bound: list = [5e-2, 5e2],
    led_tol: float = 1.0,
    led_order: int = 5,
    nelder_mead_timeout: float = 600.0,
    max_starting_points: int = 6,
) -> dict:
    """
    Convenience wrapper for scanning multiple species. Each species is optimized
    independently with find_working_configuration_FCC.
    """

    results = {}
    for species in species_list:
        results[species] = find_working_configuration_FCC(
            model=model,
            species=species,
            energy_bound=energy_bound,
            led_tol=led_tol,
            led_order=led_order,
            nelder_mead_timeout=nelder_mead_timeout,
            max_starting_points=max_starting_points,
        )
    return results


def find_working_configurations_FCC_nelder_mead_then_led(
    model: Union[str, Calculator],
    species_list: list[str],
    energy_bound: list = [5e-2, 5e2],
    led_tol: float = 1.0,
    led_order: int = 5,
    nelder_mead_timeout: float = 600.0,
    max_starting_points: int = 6,
) -> dict:
    """
    Multi-species driver matching the stricter policy:

        1. Run the Nelder-Mead-first search independently for every species.
        2. If at least one species succeeds with Nelder-Mead, return those
           results and do not use LED as a global fallback.
        3. If every species has only crash/timeout-style Nelder-Mead failures,
           return the per-species LED fallback results produced by
           find_working_configuration_FCC.

    The per-species function already carries the coarse scan and reuses its
    coarse starting points when LED fallback is allowed.
    """

    results = {}
    for species in species_list:
        results[species] = find_working_configuration_FCC(
            model=model,
            species=species,
            energy_bound=energy_bound,
            led_tol=led_tol,
            led_order=led_order,
            nelder_mead_timeout=nelder_mead_timeout,
            max_starting_points=max_starting_points,
        )

    any_nelder_mead_success = any(
        result.get("search_strategy") == "nelder_mead"
        for result in results.values()
    )

    if any_nelder_mead_success:
        for result in results.values():
            if result.get("search_strategy") == "edge_detection_fallback":
                result["ignored_by_multispecies_policy"] = True
                result["ignore_reason"] = (
                    "At least one species succeeded with Nelder-Mead, so the "
                    "global multi-species policy does not need LED fallback."
                )

    return {
        "results": results,
        "any_nelder_mead_success": any_nelder_mead_success,
        "used_global_led_fallback": not any_nelder_mead_success,
    }
