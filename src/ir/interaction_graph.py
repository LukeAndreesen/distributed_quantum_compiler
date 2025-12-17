"""
src.ir.interaction_graph

Layered interaction graph construction.

Converts a Qiskit DAG to a NetworkX graph where nodes represent (qubit, layer)
pairs and edges represent state continuity and multi-qubit gate interactions.
"""

from __future__ import annotations

from typing import Any, Tuple

import networkx as nx

def build_graph_from_dag(dag: Any) -> Tuple[nx.Graph, int]:
    """Build a layered interaction graph from a Qiskit DAG.

    Each node in the graph is a tuple ``(logical_qubit, layer_index)``. Edges connect:
    - ``(q, layer)`` to ``(q, layer+1)`` with ``weight=1`` (state continuity)
    - ``(q1, layer)`` to ``(q2, layer)`` with ``weight=5`` (two-qubit gate interaction)

    Args:
        dag: Qiskit DAGCircuit object.

    Returns:
        A tuple ``(graph, num_layers)`` where:
            - ``graph``: NetworkX Graph with nodes ``(qubit, layer)`` and weighted edges.
            - ``num_layers``: Total number of layers in the circuit.
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
