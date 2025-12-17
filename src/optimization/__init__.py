"""
src.optimization

Optimization algorithms for the compiler.

Implements cost functions and optimizers for assigning logical qubits to QPUs
across layers (partitioning).

Public API:
- partition_graph: Run the optimizer and return the best partition schedule.
"""

from .genetic import partition_graph

__all__ = ["partition_graph"]
