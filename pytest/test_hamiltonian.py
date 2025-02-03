import pytest
from quactography.hamiltonian.hamiltonian_qubit_edge import *
from quactography.graph.undirected_graph import *
from qiskit.quantum_info import PauliList



def test_hamiltonian_qubit_edge():
    graph = Graph(
    np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ]
    ),0,3,)
    h = Hamiltonian_qubit_edge(graph,1)
    assert h.alpha == 1
    assert h.graph == graph
    assert h.starting_node_c.paulis == PauliList(['IIIII','IIIIZ','IIIZI'])
    assert h.ending_node_c.paulis == PauliList(['IIIII','IZIII','ZIIII'])
    assert h.hint_c.paulis == PauliList(['IIIII','IZZIZ','IZZIZ','IIIII','IIIII','ZIZZI','ZIZZI','IIIII'])
    