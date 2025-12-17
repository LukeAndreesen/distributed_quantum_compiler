"""
src

Distributed Quantum Compiler

A modular quantum circuit compiler that optimizes qubit placement across multiple
QPUs. The compiler pipeline includes:
  - OpenQASM 3 loading
  - Layered DAG and interaction graph construction
  - Genetic algorithm + Kernighan-Lin optimization for qubit placement
  - Circuit lowering with teleportation and remote-gate placeholders

Public API:
- compile: High-level function to compile a QASM file to a distributed circuit (returns circuit and network map).
- assemble_circuit: Assemble final circuit from partition and DAG (returns circuit and network map).
- build_graph_from_dag: Build interaction graph from circuit DAG.
- confirm_candidate_validity: Validate partition candidate.
- load_qasm3_file: Load OpenQASM 3 file to Qiskit circuit.
- partition_graph: Optimize qubit-to-QPU assignment.
- RemoteGatePlaceholder: Placeholder for remote gate operations.
- TeleportPlaceholder: Placeholder for teleportation operations.
"""

from __future__ import annotations

from .analysis import confirm_candidate_validity
from .builder import assemble_circuit
from .compiler import compile
from .frontend import load_qasm3_file
from .gates import RemoteGatePlaceholder, TeleportPlaceholder
from .ir import build_graph_from_dag
from .optimization import partition_graph

__all__ = [
    "assemble_circuit",
    "build_graph_from_dag",
    "compile",
    "confirm_candidate_validity",
    "load_qasm3_file",
    "partition_graph",
    "RemoteGatePlaceholder",
    "TeleportPlaceholder",
]
