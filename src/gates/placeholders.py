"""
src.gates.placeholders

Placeholder gates for distributed operations.

Defines custom Qiskit Gate objects that serve as placeholders in the lowered
circuit for operations that will be implemented via distributed protocols.
"""

from __future__ import annotations

from qiskit.circuit import Gate

def TeleportPlaceholder(dest: str, n_qubits: int = 1) -> Gate:
    """Create a placeholder gate for qubit teleportation.

    Args:
        dest: Destination string (e.g., "qpu1:q3").
        n_qubits: Number of qubits (default: 1).

    Returns:
        Qiskit Gate object with teleportation label.
    """
    g = Gate(name="teleport", num_qubits=n_qubits, params=[])
    g.label = f"teleport → {dest}"   # what you see in drawings
    return g

def RemoteGatePlaceholder(op_name: str, qpu_a: int, qpu_b: int, n_qubits: int = 2) -> Gate:
    """Create a placeholder gate for a remote (cross-QPU) gate operation.

    Args:
        op_name: Name of the gate operation (e.g., "cx").
        qpu_a: First QPU id.
        qpu_b: Second QPU id.
        n_qubits: Number of qubits (default: 2).

    Returns:
        Qiskit Gate object with remote operation label.
    """
    g = Gate(name=op_name, num_qubits=n_qubits, params=[])
    g.label = f"{op_name} @ QPU {qpu_a} & QPU {qpu_b}"   # what you see in drawings
    return g
