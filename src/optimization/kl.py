"""
src.optimization.kl

Kernighan-Lin local search mutation.

Implements a KL-style swap mutation that greedily reduces the cut weight by
swapping qubit assignments within a layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def get_cut_weight(
    graph: Any, node: tuple, qpu_assignment: int, crossover_candidate: np.ndarray
) -> int:
    """Compute the cut weight for a node given its QPU assignment.

    Args:
        graph: Interaction graph (NetworkX Graph).
        node: Node tuple (qubit, layer).
        qpu_assignment: QPU id assigned to this node.
        crossover_candidate: Current partition candidate.

    Returns:
        Sum of edge weights crossing from this node to nodes on different QPUs.
    """
    neighbors = graph.neighbors(node)
    cut_weight = 0
    for neighbor in neighbors:
        neighbor_qpu = crossover_candidate[neighbor[1], neighbor[0]]
        if neighbor_qpu != qpu_assignment:
            edge_data = graph.get_edge_data(node, neighbor)
            cut_weight += edge_data['weight']
    return cut_weight
       

def kl_mutation(
    crossover_candidate: np.ndarray,
    graph: Any,
    num_layers: int,
    rng: np.random.Generator,
    mutation_probability: float,
    max_iterations: int,
) -> np.ndarray:
    """Apply Kernighan-Lin style swap mutation to a partition candidate.

    With probability mutation_probability, performs up to max_iterations greedy
    swaps within a random interval of layers. Each swap is accepted only if it
    reduces the total cut weight.

    Args:
        crossover_candidate: Partition to mutate, shape (depth, num_qubits).
        graph: Interaction graph (NetworkX Graph).
        num_layers: Number of circuit layers.
        rng: NumPy random generator.
        mutation_probability: Probability of applying mutation.
        max_iterations: Maximum number of swap attempts.

    Returns:
        Mutated partition candidate (or unchanged if mutation not triggered).
    """
    # make copy of candidate
    candidate_copy = np.copy(crossover_candidate)
    # Perform mutation with probability p
    random_val = rng.random()
    if random_val > mutation_probability:
        return candidate_copy
    # pick a random interval of layers - perform mutations within ~1/3 of rotations 
    interval_len = max(1, num_layers // 3)
    interval_start = rng.integers(0, num_layers - interval_len + 1)
    interval_end = interval_start + interval_len
    for i in range(max_iterations):
        # pick a layer in the interval
        layer_idx = rng.integers(interval_start, interval_end)
        random_layer = candidate_copy[layer_idx]
        # pick two qubit assignments within the layer to swap
        idx_a, idx_b = rng.choice(len(random_layer), size=2, replace=False)
        qpu_a = random_layer[idx_a]
        qpu_b = random_layer[idx_b]
        if qpu_a == qpu_b:
            continue # qubits already assigned to same qpu
        # count current weight of cut edges for each qubit
        cut_weight_a = get_cut_weight(graph, (idx_a, layer_idx), qpu_a, candidate_copy)
        cut_weight_b = get_cut_weight(graph, (idx_b, layer_idx), qpu_b, candidate_copy)
        current_weight = cut_weight_a + cut_weight_b
        # count weight of cut edges if we swapped assignments
        cut_weight_a_swapped = get_cut_weight(graph, (idx_a, layer_idx), qpu_b, candidate_copy)
        cut_weight_b_swapped = get_cut_weight(graph, (idx_b, layer_idx), qpu_a, candidate_copy)
        swapped_weight = cut_weight_a_swapped + cut_weight_b_swapped
        # if swapping reduces cut weight, perform swap
        if current_weight > swapped_weight:
            candidate_copy[layer_idx, idx_a], candidate_copy[layer_idx, idx_b] = qpu_b, qpu_a

    return candidate_copy
