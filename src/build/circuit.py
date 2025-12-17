"""
src.build.circuit

Circuit assembly and qubit mapping.

Provides utilities to map logical qubits to physical qubits according to a
partition schedule, and assemble the final distributed circuit with placeholders.
"""

from __future__ import annotations

import qiskit
from qiskit.circuit import Parameter
from collections import Counter

from ..gates.placeholders import TeleportPlaceholder, RemoteGatePlaceholder

def update_maps(partition, layer_num, qpu_size):
    """Compute logical-to-QPU and logical-to-physical mappings for a layer.

    Args:
        partition: Partition array of shape ``(depth, num_qubits)``.
        layer_num: Layer index to compute mappings for.
        qpu_size: Number of qubits per QPU.

    Returns:
        A tuple ``(logical_to_qpu, logical_to_physical)`` where:
            - ``logical_to_qpu``: Maps logical qubit index to QPU id.
            - ``logical_to_physical``: Maps logical qubit index to global physical index.
    """
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

def assemble_circuit(partition, dag, qpu_size):
    """Assemble the final distributed circuit from a partition and DAG.

    Iterates through the DAG layers and inserts:
    - Original gates (if qubits are on the same QPU)
    - RemoteGatePlaceholder (for gates across QPUs)
    - TeleportPlaceholder (when a qubit moves between QPUs across layers)

    Args:
        partition: Optimized partition array of shape ``(depth, num_qubits)``.
        dag: Qiskit DAGCircuit.
        qpu_size: Number of qubits per QPU.

    Returns:
        Qiskit QuantumCircuit with placeholders inserted.
    """

    t = Parameter("destination_qpu")  
    teleport = qiskit.circuit.Gate(name="teleport_state", num_qubits=1, params=[t])
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
    
    return qc
