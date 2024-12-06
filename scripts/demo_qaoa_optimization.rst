A quactography demo: From adjacency matrix to Hamiltonian and optimal path with QAOA
====

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
