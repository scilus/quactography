import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, Sampler


def print_hamiltonian_circuit(hamiltonian_term, binary_paths_classical_read):
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    circuit = QuantumCircuit(len(binary_paths_classical_read[0]))
    for i in range(len(binary_paths_classical_read)):
        for j in range(len(binary_paths_classical_read[i])):
            if binary_paths_classical_read[i][j] == "1":
                circuit.x(j)

        print(
            circuit,
            "\n Cost for path (classical read -> left=q0)",
            binary_paths_classical_read[i],
            " : ",
            estimator.run(circuit, hamiltonian_term).result().values[0],
        )
