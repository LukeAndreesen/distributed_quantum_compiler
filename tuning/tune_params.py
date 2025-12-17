#!/usr/bin/env python3
"""
Randomized hyperparameter tuning for partition_graph across bundled QASM tests.

Evaluates parameter samples over all test*.qasm files (and multiple seeds),
aggregates normalized improvement, and reports the best configurations.

Usage examples:
    python3 tune_params.py --trials 40 --seeds 42 1337
    python3 tune_params.py --trials 100 --qpu-size 3 --time-budget 300
    # Override input location if needed:
    python3 tune_params.py --root ./some/dir --trials 50
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from qiskit.converters import circuit_to_dag

from src import build_graph_from_dag, load_qasm3_file, partition_graph


@dataclass
class Params:
    qpu_size: int = 3
    population_size: int = 200
    num_gens: int = 100
    seed: int = 42
    temperature: float = 0.5
    mutation_probability: float = 0.8
    mutation_iterations: int = 100
    verbose: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrialResult:
    params: Dict
    mean_initial: float
    mean_final: float
    mean_improvement: float  # (initial - final) / initial
    total_seconds: float
    per_file: Dict[str, Dict[str, float]]


def discover_test_files(root: Path) -> List[Path]:
    return sorted(root.glob("test*.qasm"))


def evaluate_one(qasm_path: Path, base_params: Dict, seed: int) -> Tuple[float, float]:
    qc = load_qasm3_file(qasm_path)
    dag = circuit_to_dag(qc)
    graph, depth = build_graph_from_dag(dag)
    params = dict(base_params)
    params["seed"] = seed
    _, final_cost, _, initial_best_cost = partition_graph(
        graph, depth, dag.num_qubits(), params
    )
    return float(initial_best_cost), float(final_cost)


def evaluate_params(
    qasm_files: Sequence[Path], base_params: Dict, seeds: Sequence[int]
) -> Tuple[float, float, float, Dict[str, Dict[str, float]]]:
    t0 = time.perf_counter()
    per_file: Dict[str, Dict[str, float]] = {}
    initials: List[float] = []
    finals: List[float] = []

    for q in qasm_files:
        # Average over multiple seeds for robustness
        file_initials: List[float] = []
        file_finals: List[float] = []
        for s in seeds:
            init_cost, final_cost = evaluate_one(q, base_params, seed=s)
            file_initials.append(init_cost)
            file_finals.append(final_cost)
        init_mean = float(np.mean(file_initials))
        final_mean = float(np.mean(file_finals))
        per_file[q.name] = {
            "initial_mean": init_mean,
            "final_mean": final_mean,
            "improvement": (init_mean - final_mean) / init_mean if init_mean > 0 else 0.0,
        }
        initials.append(init_mean)
        finals.append(final_mean)

    mean_initial = float(np.mean(initials)) if initials else math.inf
    mean_final = float(np.mean(finals)) if finals else math.inf
    mean_improvement = (mean_initial - mean_final) / mean_initial if mean_initial > 0 else 0.0
    elapsed = time.perf_counter() - t0
    return mean_initial, mean_final, mean_improvement, per_file, elapsed


def sample_params(rng: random.Random, base_qpu_size: int) -> Params:
    # Heuristic search space; adjust as needed
    population_size = int(rng.choice([50, 100, 150, 200, 300, 400]))
    num_gens = int(rng.choice([30, 50, 80, 100, 120, 150]))
    temperature = 10 ** rng.uniform(-1.0, 0.2)  # ~[0.1, 1.6]
    mutation_probability = rng.uniform(0.2, 0.9)
    mutation_iterations = int(rng.choice([20, 40, 60, 80, 100, 150, 200]))
    return Params(
        qpu_size=base_qpu_size,
        population_size=population_size,
        num_gens=num_gens,
        temperature=float(temperature),
        mutation_probability=float(mutation_probability),
        mutation_iterations=mutation_iterations,
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1] / "tests" / "circuits"
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="*", default=[101, 202])
    parser.add_argument("--qpu-size", type=int, default=3)
    parser.add_argument("--time-budget", type=int, default=0, help="Optional seconds budget")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON results file")
    parser.add_argument("--random-seed", type=int, default=1337)
    args = parser.parse_args()

    qasm_files = discover_test_files(args.root)
    if not qasm_files:
        print("No test*.qasm files found under", args.root)
        return

    rng = random.Random(args.random_seed)

    best: TrialResult | None = None
    all_results: List[TrialResult] = []
    start = time.perf_counter()

    for t in range(1, args.trials + 1):
        if args.time_budget and (time.perf_counter() - start) > args.time_budget:
            print("Time budget reached; stopping search.")
            break
        p = sample_params(rng, args.qpu_size)
        mean_initial, mean_final, mean_improvement, per_file, elapsed = evaluate_params(
            qasm_files, p.to_dict(), args.seeds
        )
        tr = TrialResult(
            params=p.to_dict(),
            mean_initial=mean_initial,
            mean_final=mean_final,
            mean_improvement=mean_improvement,
            total_seconds=elapsed,
            per_file=per_file,
        )
        all_results.append(tr)
        if (best is None) or (tr.mean_improvement > best.mean_improvement):
            best = tr
        print(
            f"Trial {t:03d}: improv={tr.mean_improvement:.3f} final={tr.mean_final:.2f} "
            f"pop={p.population_size} gens={p.num_gens} temp={p.temperature:.3f} "
            f"mut_p={p.mutation_probability:.2f} mut_iters={p.mutation_iterations}"
        )

    if best is not None:
        print("\nBest configuration (by mean improvement):")
        print(json.dumps(best.params, indent=2))
        print(
            f"Mean initial: {best.mean_initial:.2f}\n"
            f"Mean final:   {best.mean_final:.2f}\n"
            f"Mean improv:  {best.mean_improvement:.3f}\n"
            f"Time (s):     {best.total_seconds:.2f}"
        )

    if args.out:
        # Save all trials to JSON for later analysis
        payload = {
            "results": [
                {
                    "params": r.params,
                    "mean_initial": r.mean_initial,
                    "mean_final": r.mean_final,
                    "mean_improvement": r.mean_improvement,
                    "total_seconds": r.total_seconds,
                    "per_file": r.per_file,
                }
                for r in all_results
            ]
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print("Saved results to", args.out)


if __name__ == "__main__":
    main()
