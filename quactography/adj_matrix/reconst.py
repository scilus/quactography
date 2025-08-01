import numpy as np
from dipy.reconst.shm import sh_to_sf
from dipy.core.sphere import Sphere
from quactography.graph.utils import get_output_nodes



DIRECTIONS_2D = np.array(
    [
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [-np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
        [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
        [np.sqrt(2.0) / 2.0, -np.sqrt(2.0) / 2.0],
        [-np.sqrt(2.0) / 2.0, -np.sqrt(2.0) / 2.0],
    ]
)


def build_adjacency_matrix(nodes_mask):
    """
    Build the adjacency matrix for a given set of nodes in a 2D image.

    Parameters
    ----------
    nodes_mask : np.ndarray
        Binary mask of the nodes.
    Returns
    -------
    adj_matrix : np.ndarray
        Adjacency matrix of the nodes.
    """




    # 1st. Assign labels to non-zero voxels (nodes)
    node_indices = np.flatnonzero(nodes_mask)

    # List of every column in the image:
    labels_volume = np.zeros(nodes_mask.shape, dtype=int)

    # Names every element in labels_volume as a node from bottom to top 
    # (lower y number to higher), column left to right (lower x to higher x)
    labels_volume[nodes_mask] = np.arange(len(node_indices))

    # Creates an empty adjacency matrix with the nodes as rows and columns (dimension)
    adj_matrix = np.zeros((len(node_indices), len(node_indices)))

    # node traversal
    for i in range(len(node_indices)):
        label = node_indices[i]

        # Coordinates of voxel
        x, y, z = np.unravel_index(label, nodes_mask.shape)

        # Adds possibility of an edge to the actual node closest neighbour in 26 directions
        for x_offset in [-1, 0, 1]:
            for y_offset in [-1, 0, 1]:
                for z_offset in [-1, 0, 1]:
                    if x_offset == 0 and y_offset == 0 and z_offset == 0:
                        continue
                    else:
                        adj_matrix = _add_edge_perhaps(
                            x + x_offset, y + y_offset, z + z_offset, i, nodes_mask, labels_volume, adj_matrix
                        )

    return adj_matrix, node_indices, labels_volume


def build_weighted_graph(adj_matrix, node_indices, sh, sh_order=12):
    """
    Build the weighted graph for a given set of nodes in a 2D image.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix of the nodes.
    node_indices : np.ndarray
        Indices of the nodes in the graph.
    sh : np.ndarray
        Spherical harmonics coefficients.
    sh_order: int, optional
        Spherical harmonics maximum order.
    Returns
    -------
    weighted_graph : np.ndarray
        Weighted adjacency matrix of the nodes.
    node_indices : np.ndarray
        Indices of the nodes in the graph.
    """
    # Get directions depending if we are in axial, coronal or sagittal :
    sphere = _get_sphere()
    sf = sh_to_sf(sh, sphere, sh_order=sh_order, basis_type="tournier07")
    sf[sf < 0.0] = 0.0
    sf /= np.max(sf, axis=(0, 1), keepdims=True)
    # sf = sf / np.max(sf, axis=(-1), keepdims=True)
    sf *= 0.5

    # print(sh.shape)
    weighted_graph = np.zeros_like(adj_matrix)
    x, y, z = np.unravel_index(node_indices, sh.shape[:3])

    # node traversal
    for it, node_row in enumerate(adj_matrix):
        nb_connections = np.count_nonzero(node_row)
        if nb_connections > 0:
            start_x, start_y, start_z = x[it], y[it], z[it]
            # which nodes are connected to every starting node:
            connected_xs, connected_ys, connected_zs = x[node_row > 0], y[node_row > 0], z[node_row > 0]

            w_list = []
            for conn_idx in range(nb_connections):
                conn_x, conn_y, conn_z = connected_xs[conn_idx], connected_ys[conn_idx], connected_zs[conn_idx]

                direction = np.array([[conn_x, conn_y, conn_z]], dtype=float) - np.array(
                    [[start_x, start_y, start_z]], dtype=float
                )
                # The directions :

                dir_id = np.argmax(np.dot(direction, sphere.vertices.T))

                w = sf[start_x, start_y, start_z, dir_id] + sf[conn_x, conn_y, conn_z, dir_id]
                w_list.append(w)

            weights = np.zeros((len(node_row),))
            weights[node_row > 0] = np.asarray(w_list)
            weighted_graph[it, :] = weights

    return weighted_graph, node_indices


def _get_sphere():
    directions = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue
                vec = np.array([i, j, k], dtype=float)
                vec /= np.linalg.norm(vec)
                directions.append(vec)
    directions = np.asarray(directions)
    return Sphere(xyz=directions)


def _add_edge_perhaps(
    pos_x, pos_y, pos_z, current_node, nodes_mask, labels_volume, adj_matrix
):
    if _is_valid_pos(pos_x, pos_y, pos_z, nodes_mask):
        neighbor_label = labels_volume[pos_x, pos_y, pos_z]
        adj_matrix[current_node, neighbor_label] = 1
    return adj_matrix


def add_end_point_edge(adj_matrix, end, labels):
    """
    Add a node and edges to the end points of the ROI in the adjacency matrix.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix of the graph.
    end_points : list of tuples
        List of end points as (x, y, z) coordinates.
    labels_volume : np.ndarray
        List of every column in the image:
    
    Returns
    -------
    np.ndarray
        Updated adjacency matrix with end point edges added.
    """
    #labels = np.unravel_index(node_indes, adj_matrix.shape)
    #new_shape = (adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1)
    adj_matrix = np.lib.pad(adj_matrix, (0, 1), 'constant', constant_values=(0))

    for i in end:
        start = labels[i[0], i[1], i[2]]
        adj_matrix[start, -1] = 1
        adj_matrix[-1, start] = 1  # Assuming undirected graph
    return adj_matrix

def _is_valid_pos(pos_x, pos_y, pos_z, nodes_mask):
    if pos_x < 0 or pos_x >= nodes_mask.shape[0]:
        return False
    if pos_y < 0 or pos_y >= nodes_mask.shape[1]:
        return False
    if pos_z < 0 or pos_z >= nodes_mask.shape[2]:
        return False
    return nodes_mask[pos_x, pos_y, pos_z]

if __name__ == "__main__":
    # little demo code here and proto test 
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    mat, nodes, labels = build_adjacency_matrix(mask)
    entry_node = np.array([1, 1, 1])
    propagation_direction = np.array([0, 1, 1])
    angle_rad = np.pi / 8

    indices = get_output_nodes(mask, entry_node, propagation_direction, angle_rad)
    new = add_end_point_edge(mat,indices,labels)
    print(new)
