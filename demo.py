from src import compile
import matplotlib.pyplot as plt
import qiskit.qasm3 as qasm3

# Simple - just compile a QASM file
compiled_circuit, network_map = compile("tests/circuits/test1.qasm", qpu_size=5)

# Returns a Qiskit QuantumCircuit with teleportation 
print(compiled_circuit)
print("Network Map:", network_map)
