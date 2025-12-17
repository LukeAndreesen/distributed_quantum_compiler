"""
src.build

Circuit building and assembly.

Constructs the final distributed circuit by mapping logical qubits to physical
indices and inserting teleportation/remote-gate placeholders.

Public API:
- assemble_circuit: Build a distributed circuit from partition and DAG.
- update_maps: Compute logical→QPU and logical→physical mappings.
"""

from .circuit import assemble_circuit, update_maps

__all__ = ["assemble_circuit", "update_maps"]
