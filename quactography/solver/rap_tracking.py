import nibabel as nib
import numpy as np

from quactography.adj_matrix.reconst import (
                    build_adjacency_matrix,
                    build_weighted_graph,
                    add_end_point_edge
)
from quactography.adj_matrix.filter import (
                    remove_orphan_nodes,
                    remove_intermediate_connections,
                    extract_slice_at_index
)
from quactography.graph.undirected_graph import Graph
from quactography.graph.utils import get_output_nodes
from quactography.hamiltonian.hamiltonian_qubit_edge import Hamiltonian_qubit_edge
from quactography.solver.qaoa_solver_qu_edge import multiprocess_qaoa_solver_edge_rap
from quactography.solver.Dijkstra import dijkstra_stepwise

def quack_rap(in_nodes_mask_img, in_sh_img, start_point, reps, alpha,
         keep_mask=None, threshold=0.2, slice_index=None,
         axis_name="axial", sh_order=8, prev_direction=[0,0,0], theta=45):
    """Build adjacency matrix from diffusion data (white matter mask and fodf peaks).
    Parameters
    ----------
    in_nodes_mask : str
        Input nodes mask image (.nii.gz file).
    in_sh : str
        Input SH image (.nii.gz file).
    start_point : int
        Starting node index in the graph.
    keep_mask : str, optional
        Nodes that must not be filtered out. If None, all nodes are filtered.
    threshold : float, optional
        Cut all weights below a given threshold. Default is 0.2.
    slice_index : int, optional
        If None, a 3D graph is built. If an integer, a slice is extracted
        along the specified axis.
    axis_name : str, optional   
        Axis along which a slice is taken. Default is "axial".
    sh_order : int, optional
        Maximum SH order. Default is 8.
    prev_direction : list, optional
        Previous direction of the streamline, used to determine the propagation direction.
        Default is [0, 0, 0].
    theta : float, optional
        Aperture angle in degrees for the propagation direction. Default is 45.

    Returns
    -------
    line : list
        List of coordinates for the streamline.
    """
    
    
    #nodes_mask_im = nib.load(in_nodes_mask)
    #sh_im = nib.load(in_sh)

    nodes_mask = in_nodes_mask_img.get_fdata().astype(bool)


    keep_node_indices = None
    if keep_mask:
        keep_mask = nib.load(keep_mask).get_fdata().astype(bool)
        keep_node_indices = np.flatnonzero(keep_mask)

    sh = in_sh_img.get_fdata()
    
    # adjacency graph
    adj_matrix, node_indices, labes = build_adjacency_matrix(nodes_mask)
    
    # assign edge weights
    weighted_graph, node_indices = build_weighted_graph(
        adj_matrix, node_indices, sh, sh_order
    )

    # filter graph edges by weight
    weighted_graph[weighted_graph < threshold] = 0.0

    # remove intermediate nodes that connect only two nodes
    weighted_graph = remove_intermediate_connections(
        weighted_graph, node_indices, keep_node_indices
    )

    # remove nodes without edges
    weighted_graph, node_indices = remove_orphan_nodes(
        weighted_graph, node_indices, keep_node_indices
    )
    if slice_index is not None:
        weighted_graph, node_indices = extract_slice_at_index(
            weighted_graph, node_indices, nodes_mask.shape, slice_index, axis_name
        )
    
    # Get end points of the streamline

    end_points = get_output_nodes(
        nodes_mask,
        entry_node=start_point,
        propagation_direction=prev_direction,
        angle_rad=theta
    )
    # Add end point edges to the adjacency matrix
    weighted_graph = add_end_point_edge(weighted_graph, end_points, labels=labes)
    end = np.flatnonzero(weighted_graph)

    #function to process the graph before quantum path finding 
    line = rap_funct(
        weighted_graph,
        starting_node = labes[start_point[0], start_point[1], start_point[2]],
        ending_node = end[-1],
        alphas = [alpha],
        reps = reps,
    )
    line.pop()
    sline = np.unravel_index(line, labes.shape)
    llist = []
    for i in range(len(sline[0])):
        llist.append([sline[0][i], sline[1][i], sline[2][i]])
    sline = llist
    return sline, prev_direction, True

def rap_funct(weighted_graph, starting_node, ending_node, alphas,
                reps, number_processors=2, optimizer="Differential"):
    """
    Process he Graph in order to create the Hamiltonian matrix before optimization
    with QAOA algorithm. The Hamiltonian is constructed with qubits as edges.

    Parameters
    ----------  
    graph : str
        Path to the input graph file (npz file).
    starting_node : int
        Starting node of the graph.
    ending_node : int
        Ending node of the graph.
    alphas : list of float, optional
        List of alpha values for the Hamiltonian. Default is [1.2].
    reps : int, optional
        Number of repetitions for the QAOA algorithm, determines the number
        of sets of gamma and beta angles. Default is 1.
    number_processors : int, optional
        Number of CPU to use for multiprocessing. Default is 2.
    optimizer : str, optional
        Optimizer to use for the QAOA algorithm. Default is "Differential".
    Returns
    -------
    line : list
        List of coordinates for the streamline.
    """
    # graph = Graph(weighted_graph, starting_node, ending_node)
    # if graph.number_of_edges > 17: 
    #        raise Exception("RAPGraph: max number of points exceeded")
    

    # # Construct Hamiltonian when qubits are set as edges,
    # # then optimize with QAOA/scipy:

    # hamiltonians = [Hamiltonian_qubit_edge(graph, alpha) for alpha in alphas]

    # # print(hamiltonians[0].total_hamiltonian.simplify())

    # print("\n Calculating qubits as edges......................")
    # # Run the multiprocess QAOA solver for edge qubits
    # # and return the optimal path as a list of coordinates.
    # line = multiprocess_qaoa_solver_edge_rap(
    #     hamiltonians,
    #     reps,
    #     number_processors,
    #     graph.number_of_edges,
    #     optimizer
    #     )
    
    line = dijkstra_stepwise(
        weighted_graph,
        starting_node,
        ending_node
    )
    return line