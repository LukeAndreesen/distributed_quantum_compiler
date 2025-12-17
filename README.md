# Distributed Quantum Compiler

A quantum circuit compiler that optimizes qubit placement across multiple QPUs.

## Quick Start

```python
from src import compile

# Compile a QASM circuit
compiled_circuit = compile("tests/circuits/test0.qasm", qpu_size=5)

# The returned circuit includes teleportation and remote gate placeholders
print(compiled_circuit)
```

### With Custom Parameters

```python
from src import compile

compiled_circuit = compile(
    "circuit.qasm",
    qpu_size=5,
    population_size=100,
    num_gens=200,
    temperature=1.0,
    verbose=True
)
```

## Run Tests

Test the compiler on all example circuits:

```bash
python tests/test_partition.py
```

## Tune Parameters

Run hyperparameter optimization:

```bash
python tune_params.py --trials 50
```
