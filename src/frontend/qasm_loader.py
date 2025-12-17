"""
src.frontend.qasm_loader

OpenQASM 3 file loading.

Provides utilities to load OpenQASM 3 files and convert them to Qiskit
QuantumCircuit objects.
"""

from __future__ import annotations

from pathlib import Path

import qiskit.qasm3 as qasm3
from qiskit import QuantumCircuit

def load_qasm3_file(file_path: str | Path) -> QuantumCircuit:
    """Load an OpenQASM 3 file and return as Qiskit QuantumCircuit.

    Args:
        file_path: Path to the QASM3 file.

    Returns:
        Qiskit QuantumCircuit object parsed from the QASM3 code.
    """
    with open(file_path, 'r') as file:
        qasm3_code = file.read()
    # Parse the QASM3 code to a QuantumCircuit
    quantum_circuit = qasm3.loads(qasm3_code)

    return quantum_circuit
