"""
src.compiler

High-level compilation entry point.

This module provides the main compile() function that orchestrates the full
compilation pipeline from QASM input to distributed circuit output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from .analysis import confirm_candidate_validity
from .build import assemble_circuit
from .frontend import load_qasm3_file
from .ir import build_graph_from_dag
from .optimization import partition_graph


def compile(qasm_path: str | Path, qpu_size: int = 5, **kwargs: Any) -> QuantumCircuit:
    """Compile a QASM circuit for distributed execution across multiple QPUs.

    This function runs the complete compilation pipeline:
    1. Load the OpenQASM 3 file
    2. Build an interaction graph from the circuit DAG
    3. Optimize qubit-to-QPU assignment using genetic algorithm + KL
    4. Validate the partition
    5. Assemble the final circuit with teleportation and remote gate placeholders

    Args:
        qasm_path: Path to the QASM3 file.
        qpu_size: Number of qubits per QPU (default: 5).
        **kwargs: Additional parameters for partition_graph:
            - population_size: GA population size (default: 50)
            - num_gens: Number of generations (default: 150)
            - seed: Random seed (default: 42)
            - temperature: Softmax temperature (default: 1.06)
            - mutation_probability: Probability of mutation (default: 0.8)
            - mutation_iterations: KL iterations per mutation (default: 100)
            - verbose: Print progress (default: False)

    Returns:
        Compiled Qiskit QuantumCircuit with teleportation and remote gate
        placeholders inserted.

    Raises:
        ValueError: If the final partition is invalid.
    """
    # Set default parameters
    params = {
        "qpu_size": qpu_size,
        "population_size": kwargs.get("population_size", 50),
        "num_gens": kwargs.get("num_gens", 150),
        "seed": kwargs.get("seed", 42),
        "temperature": kwargs.get("temperature", 1.06),
        "mutation_probability": kwargs.get("mutation_probability", 0.8),
        "mutation_iterations": kwargs.get("mutation_iterations", 100),
        "verbose": kwargs.get("verbose", False),
    }
    
    # Load circuit
    qc = load_qasm3_file(qasm_path)
    dag = circuit_to_dag(qc)
    
    # Build interaction graph
    graph, num_layers = build_graph_from_dag(dag)
    
    # Optimize partition
    partition, final_cost, _, initial_cost = partition_graph(
        graph, num_layers, dag.num_qubits(), params
    )
    
    # Validate
    if not confirm_candidate_validity(partition, params["qpu_size"]):
        raise ValueError("Final partition candidate is invalid.")
    
    # Assemble and return circuit
    return assemble_circuit(partition, dag, params["qpu_size"])
