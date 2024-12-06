Quactography demo: From adjacency matrix to Hamiltonian and optimal path found with QAOA
==========================================================================================

Motivation and general picture of this project : 

This project can be seen in two different research areas. First, the mapping from diffusion data
with white matter masks and fiber orientation distribution functions peaks to a graph (found in data file simplePhantoms or fodf.nii.gz and wm.nii.gz). Future 
works are left to do in this area to find a best suited mapping between data to graph, how to filter the graph 
without loss of information and more. Then, as a second and mainly developped aspect in this project is 
the quantum mapping of the optimisation problem at hand, which is, from any adjacency matrix which 
represents a connected graph, find the path that maximizes the weights from start node to end node
without going through an intermediate node more than once (maximum simple path). In order to do this, 
due to the limitation of graph size or number of qubits with which a local quantum algorithm can work with (which is our case
with a local quantum Algorithm called QAOA using Qiskit library and local simulation of quantum computer), 
we must find a way to extract from diffusion data, regions where quantum algorithm could be useful, where local tracking leads
to potential false tracks; this code has yet to be implemented. For now, we have in the data/test_graphs file ( or scripts/toy_graphs) 
that were manually built which are the simplest representations of the crossing regions; where a global 
tracking approach is of interest. With these toy graphs, we can test multiple scripts which leads to the quantum solution 
of the optimal path for a given graph with a defined start and end node. Furthurmore, to get more 
information about the solutions found by QAOA the Quatum Approximate Optimization Algorithm, 
there are scripts to plot the path, plot the distribution of probabilities of solutions found, 
a histogram of the 10% most probable paths and the cost landscape for the given graph if we require only 
one layer (reps) of QAOA gates in the quantum circuit that is built with the quantum cost function to minimize (called Hamiltonian). 


Terminology
---------------

Operators: Quantum gate which has a matrix representation in our case, that can be applied to a ket (a vector in our case).
Thus, when applied to a vector in its basis (eigenvector), returns an eigenvalue which has a physical interpretation (or mapped to a physical system 
such as spin etc.) 

Hamiltonian : Quantum formulation of a classical cost function to optimize, we go from boolean variables and functions to operators which 
are represented by matrices, also called quantum gates. A sum of operators that, when applied to a vector (path in graph), returns a scalar 
representing the cost of the given path (eigenvalue of the operator).     

Mixer Hamiltonian : The Hamiltonian which is easy to find the state that minimizes the energy, used as starting point in QAOA circuit. 
We start in this Hamiltonian to evolve slowly towards the Hamiltonian which represents our real cost function. 

Driver Hamiltonian : Quantum formulation of the cost function of our problem, we end up in this Hamiltonian's ground state if
QAOA circuit functions as expected, and the evolution is under right conditions. 

Adiabatic Evolution : Physics Theorem which says that if we start in the ground state (lowest energy state of system),
and evolve the system gradually (slowly enough) to another system, we end up in the ground state of the new system. This
is done by a homotopy where at time t=0 we start in a simple well-known state where we know the state that minimizes the energy (cost function)
then at time t=1, we finish in a new system (represented by the quantum cost function of our problem which is not well known)
and we should end up in the ground state of this new system (minimum found through adiabatic evolution). 
Best known homotopy: (1-t) H_mixer + t H_driver. 

Qubit : Two-level system either 0 or 1 which corresponds to wether a path is absent (0) 
or present (1) in path. Represented by a vector (also called ket in quantum formalism). 

Pauli gates : Any cost function can be mapped to a Hamiltonian by a set of universal quantum gates, 

Pauli string : Qiskit library can do operations with quantum gates (operators)

Optimal Parameters : QAOA works as a parametrized circuit, and for each layer of gates (Mixer and Driver Hamiltonian), 
it takes in a set of parameters (gamma and beta). The optimal parameters are the gamma's and beta's whith gamma from 0 to 2pi 
and beta from 0 to pi, which minimizes the cost measured by quantum circuit with the highest probability. 


QUBO Quadratic Unconstrained Binary Optimization formulation : A way of expressing a cost function with constraints as 
a cost function without constraints, but penalties that represent the constraints. This requires a coefficient determined 
by us in order to add a weight and control how much breaking a constraint raises our cost, the more the coefficient is high, 
the more breaking the associated constraint is costly. 

Adjacency matrix : Matrix with nodes indices as rows and columns, and elements of the matrix 
which represent the weight between any pair of nodes in graph. 


Methodology 
------------------

