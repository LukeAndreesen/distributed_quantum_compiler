"""
src.analysis.cut_metrics

Partition validation and cut metrics.

Provides functions to validate partition candidates against QPU capacity
constraints and compute cut-based quality metrics.
"""

from __future__ import annotations

from collections import Counter
import numpy as np

def confirm_candidate_validity(candidate: np.ndarray, qpu_size: int) -> bool:
    """Validate that a partition candidate satisfies QPU capacity constraints.

    Checks that no QPU is assigned more than ``qpu_size`` logical qubits in any layer.

    Args:
        candidate: Partition array of shape ``(depth, num_qubits)`` where
            ``candidate[layer, qubit]`` is the assigned QPU id.
        qpu_size: Maximum number of qubits allowed per QPU per layer.

    Returns:
        True if the candidate is valid (all capacity constraints satisfied),
        False otherwise.
    """
    # ensure all qpus have at most qpu_size qubits assigned per layer
    depth, num_qubits = candidate.shape
    for layer in range(depth):
        qpu_counter = Counter()
        for qubit in range(num_qubits):
            qpu_assignment = candidate[layer, qubit]
            qpu_counter[qpu_assignment] += 1
        for qpu, count in qpu_counter.items():
            if count > qpu_size:
                print(
                    f"Invalid candidate: layer {layer} exceeds capacity on QPU {int(qpu)} "
                    f"(assigned {int(count)} > capacity {int(qpu_size)})"
                )
                return False
    return True
