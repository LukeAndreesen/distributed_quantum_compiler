from src import compile
import matplotlib.pyplot as plt
import qiskit.qasm3 as qasm3

# Simple - just compile a QASM file
compiled_circuit = compile("tests/circuits/test2.qasm", qpu_size=5)

# Returns a Qiskit QuantumCircuit with teleportation 
print(compiled_circuit)
