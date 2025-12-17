import sys
import math
from collections import Counter, defaultdict
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.converters import circuit_to_dag
import qiskit.qasm3 as qasm3
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# TODO: implement / learn about pass manager?

def load_qasm3_file(file_path):
    """
    Load qasm3 file and return as Qiskit QuantumCircuit
    
    :param file_path: Path to the qasm3 file
    :return: Qiskit QuantumCircuit object
    """
    with open(file_path, 'r') as file:
        qasm3_code = file.read()
    # Parse the QASM3 code to a QuantumCircuit
    quantum_circuit = qasm3.loads(qasm3_code)

    return quantum_circuit


def build_graph_from_dag(dag):
    """
    """
    graph = nx.Graph()
 #   print(dag.num_qubits())
    num_qubits = dag.num_qubits()
    num_layers = 0

    for layer_num, layer in enumerate(dag.layers()):
        num_layers += 1
        # Add qubit nodes for this layer
        for n in range(num_qubits):
            graph.add_node((n, layer_num))
            # Add state edges from previous layer
            if layer_num > 0:
                graph.add_edge((n, layer_num - 1), (n, layer_num), weight=1, edge_type='state')
        # Add gate edges for this layer
        layer_dag = layer["graph"]
        layer_nodes = layer_dag.two_qubit_ops() # all operations
        # TODO: do we need to generalize for 2+ qubit gates
        for node in layer_nodes:
            qubit_indices = [dag.find_bit(q).index for q in node.qargs]

            graph.add_edge((qubit_indices[0], layer_num), (qubit_indices[1], layer_num), weight=5, edge_type='gate')
    return graph, num_layers


def random_static_candidate(depth, num_qubits, qpu_size, rng):
    
    # random order of qubits
    perm = rng.permutation(num_qubits)

    phi = np.empty(num_qubits, dtype=int)
    # assign in chunks of size of qpu
    for idx, q in enumerate(perm):
        phi[q] = idx // qpu_size  

    # repeat across layers
    return np.tile(phi, (depth, 1))  # shape (depth, num_qubits)


def softmax(x, temperature=0.5):
    x = np.array(x, dtype=np.float64)
    scaled = x / temperature
    scaled = scaled - scaled.max()
    exp_x = np.exp(scaled)
    return exp_x / exp_x.sum()


def get_cut_weight(graph, node, qpu_assignment, crossover_candidate):
    neighbors = graph.neighbors(node)
    cut_weight = 0
    for neighbor in neighbors:
        neighbor_qpu = crossover_candidate[neighbor[1], neighbor[0]]
        if neighbor_qpu != qpu_assignment:
            edge_data = graph.get_edge_data(node, neighbor)
            cut_weight += edge_data['weight']
    return cut_weight
       

def confirm_candidate_validity(candidate, qpu_size):
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
            

def kl_mutation(
    crossover_candidate,
    graph,
    num_layers,
    rng,
    mutation_probability,
    max_iterations,
):
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

def partition_graph(graph, depth, num_qubits, params):
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

def compute_cost(graph, candidate):
    cost = 0
    for u, v, data in graph.edges(data=True):
        # qpu assignemnts are elements (layer, qubit) for each candidate
        qpu_assignment_u = candidate[u[1], u[0]]  
        qpu_assignment_v = candidate[v[1], v[0]]  
        if qpu_assignment_u != qpu_assignment_v:
            cost += data['weight']
    return cost

def update_maps(partition, layer_num, qpu_size):
    layer = partition[layer_num]
    # complete moving logic for easy state change checks
    logical_to_qpu = {}
    qpu_counter = Counter()
    logical_to_physical = {}
    # Assign each qubit to its distributed QPU
    for logical_index, qpu_assignment in enumerate(layer):
        # cast to int for clarity
        logical_to_qpu[int(logical_index)] = qpu_assignment
        logical_index, qpu_assignment = int(logical_index), int(qpu_assignment)
      #  qpu_to_logical[qpu_assignment].append(logical_index)
        physical_map = (qpu_size * qpu_assignment) + qpu_counter[qpu_assignment]
        qpu_counter[qpu_assignment] += 1
        logical_to_physical[logical_index] = physical_map

    return logical_to_qpu, logical_to_physical
    

def TeleportPlaceholder(dest: str, n_qubits: int = 1):
    g = Gate(name="teleport", num_qubits=n_qubits, params=[])
    g.label = f"teleport → {dest}"   # what you see in drawings
    return g

def RemoteGatePlaceholder(op_name: str, qpu_a: int, qpu_b: int, n_qubits: int = 2):
    g = Gate(name=op_name, num_qubits=n_qubits, params=[])
    g.label = f"{op_name} @ QPU {qpu_a} & QPU {qpu_b}"   # what you see in drawings
    return g

def assemble_circuit(partition, dag, qpu_size):

    t = Parameter("destination_qpu")  
    teleport = Gate(name="teleport_state", num_qubits=1, params=[t])
    num_qubits = dag.num_qubits()

    # Create a fresh QuantumCircuit
    qc = qiskit.QuantumCircuit(num_qubits)
    # Iterate through original QC insturctions
    logical_to_qpu, logical_to_physical = update_maps(partition, 0, qpu_size)
    for layer_num, layer in enumerate(dag.layers()):
        layer_dag = layer["graph"]
        for node in layer_dag.op_nodes():
            qubit_indices = [dag.find_bit(q).index for q in node.qargs]
            # TODO: generalize for 2+ qubit gates
            if len(qubit_indices) > 1:
                qpu_a = logical_to_qpu[qubit_indices[0]]
                physical_a = logical_to_physical[qubit_indices[0]]
                qpu_b = logical_to_qpu[qubit_indices[1]]
                physical_b = logical_to_physical[qubit_indices[1]]
                op = node.op.name
                if qpu_a != qpu_b:
                    qc.append(
                        RemoteGatePlaceholder(op, qpu_a, qpu_b, n_qubits=2),
                        [physical_a, physical_b],
                    )
                else:
                    qc.append(node.op, [physical_a, physical_b])
            else:
                qubit = qubit_indices[0]
                qpu = logical_to_qpu[qubit]
                physical = logical_to_physical[qubit]
                qc.append(node.op, [physical])
        if layer_num + 1 < partition.shape[0]:
            next_logical_to_qpu, next_logical_to_physical = update_maps(partition, layer_num + 1, qpu_size)
            for qubit in range(num_qubits):
                current_qpu = logical_to_qpu[qubit]
                next_qpu = next_logical_to_qpu[qubit]
                current_physical = logical_to_physical[qubit]
                next_physical = next_logical_to_physical[qubit]
                if current_qpu != next_qpu:
                    qc.append(TeleportPlaceholder(f"qpu{next_qpu}:q{next_physical}"), [current_physical])
        print(qc)
    

def main():

    file_path = Path(sys.argv[1])

    if not file_path.is_file():
        print(f"Error: {file_path} is not a file.")
        raise SystemExit(1)


    qc = load_qasm3_file(file_path)
    dag = circuit_to_dag(qc)
    # graph = build_graph(qc)
    graph, num_layers = build_graph_from_dag(dag)
    params = {
        "qpu_size": 5,
        "population_size": 50,
        "num_gens": 150,
        "seed": 42,
        "temperature": 1.06,
        "mutation_probability": 0.8,
        "mutation_iterations": 100,
        "verbose": False,
    }
    partition, final_cost, _, initial_best_cost = partition_graph(
        graph,
        num_layers,
        dag.num_qubits(),
        params,
    )
   # print(f"Initial best cost: {initial_best_cost}")
   # print(f"Final best cost: {final_cost}")
  #  print("Final partition assignment (layer x qubit):")
    print(partition)
    if partition.shape[0] != num_layers:
        raise ValueError("Partition depth does not match number of layers.")
    if not confirm_candidate_validity(partition, params["qpu_size"]):
        raise ValueError("Final partition candidate is invalid.")
    assemble_circuit(partition, dag, params["qpu_size"])

if __name__ == "__main__":
    main()
