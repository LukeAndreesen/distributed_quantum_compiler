"""
src.optimization.cost

Cost function for partition evaluation.

Provides the objective function used by the genetic algorithm to evaluate
partition quality based on interaction graph cuts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

def compute_cost(graph, candidate):
    """Compute the total cut cost of a partition candidate.

    The cost is the sum of edge weights for all edges crossing between different
    QPUs in the partition.

    Args:
        graph: Interaction graph (NetworkX Graph) with weighted edges.
        candidate: Partition array of shape ``(depth, num_qubits)`` where
            ``candidate[layer, qubit]`` is the assigned QPU id.

    Returns:
        Total weight of edges crossing QPU boundaries.
    """
    cost = 0
    for u, v, data in graph.edges(data=True):
        # qpu assignemnts are elements (layer, qubit) for each candidate
        qpu_assignment_u = candidate[u[1], u[0]]  
        qpu_assignment_v = candidate[v[1], v[0]]  
        if qpu_assignment_u != qpu_assignment_v:
            cost += data['weight']
    return cost
