"""
src.ir

Intermediate representations.

Utilities for converting Qiskit DAGs into layered interaction graphs used
by the optimizer.

Public API:
- build_graph_from_dag: Convert a Qiskit DAG to an interaction graph.
"""

from .interaction_graph import build_graph_from_dag

__all__ = ["build_graph_from_dag"]
