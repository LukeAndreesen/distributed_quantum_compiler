"""
src.frontend

Frontend parsing and loading.

This package handles input file parsing and conversion to internal
representations. Currently supports OpenQASM 3 via Qiskit's parser.

Public API:
- load_qasm3_file: Load an OpenQASM 3 file and return a Qiskit circuit.
"""

from .qasm_loader import load_qasm3_file

__all__ = ["load_qasm3_file"]
