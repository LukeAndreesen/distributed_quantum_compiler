#!/usr/bin/env python3
"""Batch runner for partition_graph across bundled QASM test circuits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from playground import build_graph_from_dag, load_qasm3_file, partition_graph

DEFAULT_PARAMS = {
    "qpu_size": 3,
    "population_size": 200,
    "num_gens": 100,
    "seed": 42,
    "temperature": 0.5,
    "mutation_probability": 0.8,
    "mutation_iterations": 100,
    "verbose": False,
}


def discover_test_files(root: Path) -> Sequence[Path]:
    """Return all test*.qasm files in sorted order."""
    return sorted(root.glob("test*.qasm"))


def evaluate_circuit(qasm_path: Path, params: dict) -> tuple[float, float]:
    """Compute initial and final best costs for the given QASM circuit."""
    dag = load_qasm3_file(qasm_path)
    graph, depth = build_graph_from_dag(dag)
    _, final_cost, _, initial_best_cost = partition_graph(
        graph,
        depth,
        dag.num_qubits(),
        params,
    )
    return initial_best_cost, final_cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing test*.qasm files (defaults to script directory).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override applied to each evaluation (with index offset).",
    )
    args = parser.parse_args()

    root = args.root
    test_files = discover_test_files(root)
    if not test_files:
        print("No QASM test files found.")
        return

    for idx, qasm_path in enumerate(test_files):
        params = DEFAULT_PARAMS.copy()
        if args.seed is not None:
            params["seed"] = args.seed + idx
        initial_cost, final_cost = evaluate_circuit(qasm_path, params)
        print(
            f"{qasm_path.name}: initial best cost = {initial_cost}, final best cost = {final_cost}"
        )


if __name__ == "__main__":
    main()
