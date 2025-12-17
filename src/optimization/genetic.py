"""
src.optimization.genetic

Genetic optimization routines for the compiler.

This module implements a capacity-safe genetic algorithm for assigning logical
qubits to QPUs across circuit layers. The optimizer minimizes an interaction-graph
cut cost and supports local improvement via KL-style swap mutations.

Public API:
- partition_graph: Run the optimizer and return the best partition schedule.
"""

from __future__ import annotations

import math
import numpy as np

def softmax(x: np.ndarray | list, temperature: float = 0.5) -> np.ndarray:
    """Compute softmax probabilities with temperature scaling.

    Args:
        x: Input values.
        temperature: Temperature parameter for scaling (lower = more peaked).

    Returns:
        Softmax probabilities that sum to 1.
    """
    x = np.array(x, dtype=np.float64)
    scaled = x / temperature
    scaled = scaled - scaled.max()
    exp_x = np.exp(scaled)
    return exp_x / exp_x.sum()


def random_static_candidate(
    depth: int, num_qubits: int, qpu_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a random static partition candidate.

    Assigns qubits to QPUs in random chunks of size ``qpu_size``, then tiles the
    assignment across all layers.

    Args:
        depth: Number of circuit layers.
        num_qubits: Number of logical qubits.
        qpu_size: QPU capacity (qubits per QPU).
        rng: NumPy random generator.

    Returns:
        Partition array of shape ``(depth, num_qubits)``.
    """
    
    # random order of qubits
    perm = rng.permutation(num_qubits)

    phi = np.empty(num_qubits, dtype=int)
    # assign in chunks of size of qpu
    for idx, q in enumerate(perm):
        phi[q] = idx // qpu_size  

    # repeat across layers
    return np.tile(phi, (depth, 1))  # shape (depth, num_qubits)


def partition_graph(graph, depth, num_qubits, params):
    """Optimize a multi-QPU qubit assignment schedule for a layered circuit.

    The optimizer searches for a layer-by-layer assignment of logical qubits to QPUs
    that minimizes the weighted cut of an interaction graph. The returned partition
    is a ``(depth, num_qubits)`` array where entry ``[layer, qubit]`` is the assigned QPU id.

    Args:
        graph: Interaction graph with nodes of the form ``(logical_qubit, layer_index)``
            and weighted edges representing state continuity and multi-qubit gates.
        depth: Number of circuit layers.
        num_qubits: Number of logical qubits in the original circuit.
        params: Optimizer configuration. Required keys:
            - ``qpu_size``: Maximum number of logical qubits per QPU per layer.
            Optional keys:
            - ``population_size``, ``num_gens``, ``seed``, ``temperature``,
              ``mutation_probability``, ``mutation_iterations``, ``verbose``.

    Returns:
        A tuple ``(best, best_cost, initial_best, initial_best_cost)``:
            - ``best``: Best partition found, shape ``(depth, num_qubits)``.
            - ``best_cost``: Cost of the best partition.
            - ``initial_best``: Best partition from initial random population.
            - ``initial_best_cost``: Cost of ``initial_best``.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    # TODO: handle distinct qpu sizes case
    qpu_size = params.get("qpu_size")
    if qpu_size is None:
        raise ValueError("Parameter 'qpu_size' is required in params.")

    population_size = params.get("population_size", 200)
    num_gens = params.get("num_gens", 100)
    seed = params.get("seed", 42)
    temperature = params.get("temperature", 0.5)
    mutation_probability = params.get("mutation_probability", 0.8)
    mutation_iterations = params.get("mutation_iterations", 100)
    verbose = params.get("verbose", False)

    # Import locally to avoid circular dependency
    from .cost import compute_cost
    from .kl import kl_mutation

    # Generate population of candidate assignments
    rng = np.random.default_rng(seed)
    L = []
    initial_costs = []
    initial_best_cost = math.inf
    initial_best_candidate = None
    for _ in range(population_size):
        static = random_static_candidate(depth, num_qubits, qpu_size, rng)
        L.append(static)
        initial_cost = compute_cost(graph, static)
        initial_costs.append(initial_cost)
        if initial_cost < initial_best_cost:
            initial_best_cost = initial_cost
            initial_best_candidate = static.copy()
        
    if verbose:
        print(f"average initial cost: {np.mean(initial_costs)}")
        print("best initial cost:", initial_best_cost)
    # Initialize best tracker from initial population
    best = initial_best_candidate.copy()
    best_cost = initial_best_cost

    for gen in range(1, num_gens + 1):
        # Compute costs for current population
        cost_list = [compute_cost(graph, cand) for cand in L]

        # Track global best
        gen_best_idx = int(np.argmin(cost_list))
        gen_best_cost = cost_list[gen_best_idx]
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best = L[gen_best_idx].copy()

        if verbose and gen % 10 == 0:
            print(f"Generation {gen}: best cost {best_cost}")

        # Convert costs to fitness: lower cost → higher probability
        fitness = [-c for c in cost_list]
        distance = softmax(fitness, temperature=temperature)

        # Elitism - Keep top 10 candidates
        elitism_count = min(10, population_size)
        elite_indices = np.argsort(cost_list)[:elitism_count]
        new_population = [L[i].copy() for i in elite_indices]

        # Generate offspring for remaining slots via mutation-only (capacity-safe)
        offspring_needed = population_size - elitism_count
        for _ in range(offspring_needed):
            parent_idx = int(np.random.choice(len(L), p=distance))
            child = L[parent_idx].copy()
            child = kl_mutation(
                child,
                graph,
                depth,
                rng,
                mutation_probability,
                mutation_iterations,
            )
            new_population.append(child)

        # Truncate in case of rounding errors
        L = new_population[:population_size]

    return best, best_cost, initial_best_candidate, initial_best_cost
