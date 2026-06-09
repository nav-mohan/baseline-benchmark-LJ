import json 

import numdifftools as nd
import numpy as np
import math

import kim_tools.ase as kim_ase_utils
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.kim import KIM, get_model_supported_species
import scipy.optimize
import copy 
import time
import random
import os

import json

# choose which input_data to import, whether PORTABLE-MODEL, SIM_MODEL or TORCH_MODEL
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument("--input-module", required=True)
args = parser.parse_args()

input_module = importlib.import_module(args.input_module)
input_data = input_module.input_data

# to choose which type of models to run, execute script.py like so
# python script.py --input-module input_data_TORCH
# python script.py --input-module input_data_PORT
# python script.py --input-module input_data_SIM
######################################################################################################

## why step discontinuities? maybe due to neighoborlist. 
## so count the number of neighbor atoms at each density 
## see whether the average number of neighbors also is step-like in sync with runtime
## havent used this yet because i dont know how to get cutoff for each model. 
# from ase.neighborlist import neighbor_list
# def average_neighbor_count(atoms, cutoff):
#     i, j = neighbor_list("ij", atoms, cutoff)
#     counts = np.bincount(i, minlength=len(atoms))
#     return np.mean(counts), np.min(counts), np.max(counts)


import numpy as np
from ase import Atoms


def make_random_liquid_like_config(symbols,box_length,min_dist,seed=13,max_attempts_per_atom=10000,):
    """
        Generate a periodic liquid-like random configuration.

        Parameters
        ----------
        symbols : list[str]
            One chemical symbol per atom, e.g. ["Al"] * 2048 or mixed species list.
        box_length : float
            Cubic periodic box length.
        min_dist : float
            Minimum allowed pair distance using minimum-image convention.
        seed : int
            RNG seed.
        max_attempts_per_atom : int
            Maximum random placement attempts per atom.

        Returns
        -------
        atoms : ase.Atoms
            Periodic random configuration.
    """
    rng = np.random.default_rng(seed)

    positions = []
    cell = np.eye(3) * box_length

    def minimum_image_distances(candidate, existing_positions):
        if len(existing_positions) == 0:
            return np.array([])

        dr = candidate - np.asarray(existing_positions)

        # Minimum-image convention for cubic PBC
        dr -= box_length * np.round(dr / box_length)

        return np.linalg.norm(dr, axis=1)

    for i, sym in enumerate(symbols):
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

def cubic_cell_energy(alat, atoms, ncells_per_side):
    """
    Calculate the energy of the passed 'atoms' structure containing a
    cubic structure with 'ncells_per_side'. Scale to lattice constant
    'alat' (passed as a nd array of length 1) and return the energy.
    """
    acell = alat[0] * ncells_per_side
    atoms.set_cell([acell, acell, acell], scale_atoms=True)
    e = atoms.get_potential_energy()
    return e

def find_equilibrium_fcc(model:str,species:list, ncells_per_side:int = 2, grid_stepsize:float=0.01, min_alat:float = 2.5, max_alat:float = 10.0):
    alat_ave = []
    for spec in species:
        # Check if this species has non-trivial force and energy interactions
        atoms_interacting_energy, atoms_interacting_force = kim_ase_utils.check_if_atoms_interacting(
            model, symbols=[spec, spec]
        )
        if not atoms_interacting_energy:
            print("")
            print(
                "WARNING: The model provided, {}, does not possess a non-trivial energy "
                "interaction for species {} as required by this Verification "
                "Check. Skipping...".format(model, spec)
            )
            print("")
            continue

        if not atoms_interacting_force:
            print("")
            print(
                "WARNING: The model provided, {}, does not possess a non-trivial force "
                "interaction for species {} as required by this Verification Check.  "
                "Skipping...".format(model, spec)
            )
            print("")
            continue

        # find equilibrium lattice constant, so that the numerical derivatives
        # of all potentials are evaluated in a similar portion of their
        # potential energy surface, making comparisons between potentials
        # more meaningful.
        calc = KIM(model)
        alat = min_alat
        done = False
        while not done:
            atoms = FaceCenteredCubic(
                size=(ncells_per_side,ncells_per_side,ncells_per_side), 
                latticeconstant=alat, 
                symbol=spec, 
                # pbc=False
                pbc=True
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
                alat = res.x[0]
                done = True
            except:  # noqa: E722
                # failed for some reason (assume it's because of KIM error)
                alat += grid_stepsize
                if alat > max_alat:
                    done = True
        alat_ave.append(alat)

    if len(alat_ave) == 0:
        alat_ave = [np.float64(-1.0)]
    return np.mean(alat_ave)

def generate_alat_range(alat_eq, min_frac = 0.75, max_frac = 2, num_alats=50):
    # 1. Create a linear space from 0 to 1
    t = np.linspace(0, 1, num_alats)
    # 2. Square it to bunch points near 1
    t_squared = 1 - (1-t)**2
    # 3. Scale and shift to your target range (0.75 to 3)
    result = min_frac + (max_frac- min_frac) * t_squared 
    alat_range = result * alat_eq
    return alat_range

import numpy as np


def reject_ratio_outliers(init_times, recalc_times, zmax=4):
    """
    Remove benchmark runs where recalcTime/initTime is an outlier.

    Parameters
    ----------
    init_times : array-like
        Init/cold compute times.
    recalc_times : array-like
        Recalc/warm compute times.
    zmax : float
        Robust z-score cutoff.

    Returns
    -------
    init_clean : np.ndarray
        Init times with ratio-outlier runs removed.
    recalc_clean : np.ndarray
        Recalc times with ratio-outlier runs removed.
    keep_mask : np.ndarray[bool]
        True for runs kept.
    ratios : np.ndarray
        recalc/init ratio for every original run.
    """
    init_times = np.asarray(init_times, dtype=float)
    recalc_times = np.asarray(recalc_times, dtype=float)

    if len(init_times) != len(recalc_times):
        raise ValueError("init_times and recalc_times must have the same length.")

    if np.any(init_times <= 0):
        raise ValueError("All init times must be positive to compute recalc/init ratio.")

    ratios = recalc_times / init_times

    median = np.median(ratios)
    abs_dev = np.abs(ratios - median)
    mad = np.median(abs_dev)

    if mad == 0:
        # Fallback if most ratios are identical.
        std = np.std(ratios)
        if std == 0:
            keep_mask = np.ones_like(ratios, dtype=bool)
        else:
            keep_mask = np.abs(ratios - median) <= zmax * std
    else:
        robust_z = 0.6745 * (ratios - median) / mad
        keep_mask = np.abs(robust_z) <= zmax

    return init_times[keep_mask], recalc_times[keep_mask], keep_mask, ratios


def do_bench(model:str,species:list, alat_eq:float):
    alat_range:np.array = generate_alat_range(alat_eq)
    print(f"\tscanning range {alat_range}")
    ncells_per_side = 8
    pert_amp = 0.05
    average_iterations = 10

    # data for plotting graphs
    density:list = []
    energy:list = []
    initTime_AVG:list = []
    initTime_STD:list = []
    recalcTime_AVG:list = []
    recalcTime_STD:list = []
    
    # _atoms = None # _atoms is an ATOMS object with alat=1 and sufficient unit-cells per side to accomodate the entire species array
    # while True:
        # _atoms = FaceCenteredCubic(
        #     size=(ncells_per_side, ncells_per_side, ncells_per_side),
        #     latticeconstant=1.,
        #     symbol="H",
        #     pbc=True,
        # )
        # #increment ncells_per_side to accomodate all unique atoms in species
        # if len(_atoms) < len(species): 
        #     ncells_per_side += 1
        # else:
        #     break

    for alat in alat_range:
        # atoms_alat = copy.deepcopy(_atoms)
        # atoms_alat.set_cell([ncells_per_side*alat]*3,scale_atoms = True) # change the density of the unit-cell

        # isntead of FCC, do a liquid
        # Choose a minimum distance.
        # Start conservatively around 0.6 to 0.8 times the nearest-neighbor FCC distance.
        natoms = 4 * ncells_per_side**3
        box_length = ncells_per_side * alat
        # symbols = [species[0]] * natoms # For a single-species model:
        rng = np.random.default_rng(seed=12345)
        symbols = rng.choice(species, size=natoms, replace=True).tolist() # For a mixed-species benchmark, random mixture:
        min_dist = 0.5 * alat / np.sqrt(2)
        atoms_alat = make_random_liquid_like_config(
            symbols=symbols,
            box_length=box_length,
            min_dist=min_dist,
            seed=int(1000000 * alat),
        )

        initTime_runs:list      = []
        recalcTime_runs:list    = []
        energy_runs:list        = []

        
        # print(f"\t\tSTART {alat}")
        for t in range(average_iterations):
            # print(f"\t\t\tITERATION {t}/{average_iterations}")
            atoms = copy.deepcopy(atoms_alat)
            kim_ase_utils.randomize_species(atoms,species)
            calc = KIM(model)
            atoms.set_calculator(calc)
            kim_ase_utils.randomize_positions(atoms,pert_amp * alat)
            
            # initial calculation
            init_start = time.perf_counter_ns()
            pe1 = atoms.get_potential_energy()
            init_end = time.perf_counter_ns()
            initTime_runs.append(init_end - init_start)
            energy_runs.append(pe1)
            # print(f"\t\t\tinitTime {init_end - init_start}")


            # attempt(1) at flushing the cache - 
            # didnt work as well as i'd like but it's what i'm going with for now
            # i'm seeing step-discontinutiotes and that doesnt work well for linear interpolations
            # the step-discont are due to FCC neighborlists (atoms moving in/out of neighborlists)
            # we dont actually care about the specific configuration. no reason to prefer FCC
            atoms.set_positions(atoms_alat.get_positions())
            pos_before_pert = atoms.get_positions().copy()
            while True:
                kim_ase_utils.randomize_positions(atoms,0.0001 * alat)
                pos_after_pert = atoms.get_positions().copy()
                max_pos_change =  np.max(np.abs(pos_after_pert - pos_before_pert))
                # print("\t\t\tpositions changed:", max_pos_change)
                if max_pos_change > 1e-6: break


            ## attempt(2) at flushing the cache - just make a dummy call to get_potntial_energy() at a different configuration
            ## this produced results pretty similar to attempt(1)
            # atoms.set_positions(atoms_alat.get_positions())
            # pe2 = atoms.get_potential_energy()
            # kim_ase_utils.randomize_positions(atoms,0.0001 * alat)


            
            ## attempt(3) to flush the cache 
            ## this reset all the way back to initTime
            # atoms.calc.results.clear() # Flush ASE-level cached results
            # atoms.calc.reset() # More complete ASE calculator reset
            # atoms.set_positions(atoms_alat.get_positions())
            # kim_ase_utils.randomize_positions(atoms,0.0001 * alat)
            
            # print(f"\t\t\tflushCache")


            # recalcualtion
            recalc_start = time.perf_counter_ns()
            pe2 = atoms.get_potential_energy()
            recalc_end = time.perf_counter_ns()
            recalcTime_runs.append(recalc_end- recalc_start)
            energy_runs.append(pe2)
            # print(f"\t\t\trecalctime {recalc_end- recalc_start}")
            
            cached_start = time.perf_counter_ns()
            pe_cached = atoms.get_potential_energy()
            cached_end = time.perf_counter_ns()
            print(f"\t\t\t {t}/{average_iterations} : {(recalc_end - recalc_start)/(init_end - init_start):.4f} {(cached_end - cached_start)/(init_end - init_start):.4f}")


        # print(f"\t\tDONE {alat}")

        # mean/variance of the 10 iteration 
        density.append((alat_eq/alat)**3)# normalize density by equilibrium-density
        energy.append(np.mean(energy_runs))
        # initTime_AVG.append(np.mean(initTime_runs))
        # initTime_STD.append(np.std(initTime_runs))
        # recalcTime_AVG.append(np.mean(recalcTime_runs))
        # recalcTime_STD.append(np.std(recalcTime_runs))

        init_clean, recalc_clean, keep_mask, ratios = reject_ratio_outliers(initTime_runs,recalcTime_runs,zmax=6,)
        init_arr = np.asarray(initTime_runs, dtype=float)
        recalc_arr = np.asarray(recalcTime_runs, dtype=float)
        removed_init = init_arr[~keep_mask]
        removed_recalc = recalc_arr[~keep_mask]
        removed_ratios = ratios[~keep_mask]
        # print("\t\tratios:", ratios)
        # print("\t\tkept ratios:", ratios[keep_mask])
        print("\t\tremoved:", removed_ratios)
        initTime_AVG.append(np.mean(init_clean))
        initTime_STD.append(np.std(init_clean, ddof=1) if len(init_clean) > 1 else 0.0)
        recalcTime_AVG.append(np.mean(recalc_clean))
        recalcTime_STD.append(np.std(recalc_clean, ddof=1) if len(recalc_clean) > 1 else 0.0)
        # print(f"\t\t{alat:.4f} {np.mean(initTime_runs):.4f} | {np.mean(recalcTime_runs):.4f}")
        # print(f"\t\t{alat:.4f} {np.mean(recalcTime_runs)/np.mean(initTime_runs):.4f}")


    model_bench_results = {
        "density" : density,
        "energy" : energy,
        "initTime_AVG" : initTime_AVG,
        "initTime_STD" : initTime_STD,
        "recalcTime_AVG" : recalcTime_AVG,
        "recalcTime_STD" : recalcTime_STD,
    }
    return model_bench_results



def main():
    for data in input_data:
        
        model,species,model_shortname = data["model"], data["species"], data["model_shortname"]
        print(f"MODEL : {model_shortname}")
        
        # find range of densities.
        # this is based on model's equilibrium
        # the same range of densities will be used for LJ as well (is that correct?)
        mixed_alat_eq = find_equilibrium_fcc(model,species)
        print(f"\tEQUIL_ALAT : {mixed_alat_eq}")

        # do model benchmark
        model_bench = do_bench(model,species,mixed_alat_eq)
        print(f"\tDONE BENCH {model}")
        
        # Do LJ benchmark
        lj_bench = do_bench("LennardJones612_UniversalShifted__MO_959249795837_003",species,mixed_alat_eq)
        print(f"\tDONE BENCH LJ")
        
        plotdata = [
            {
                "model":model,
                "model_bench" : model_bench
            },
            {
                "model":"LennardJones612_UniversalShifted__MO_959249795837_003",
                "lj_bench" : lj_bench
            }
        ]
        with open(f"{model}_LJ_baseline_{'-'.join(species)}.json", "w") as file:
            json.dump(plotdata, file, indent=4)

if __name__ == '__main__':
    main()


# QUESTION : Shoudl I be comparing LJ at the same densities? 
