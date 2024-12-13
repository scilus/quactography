import numpy as np
from dipy.reconst.shm import sh_to_sf
from dipy.core.sphere import Sphere


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
    assert len(nodes_mask.shape) == 2

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
        x, y = np.unravel_index(label, nodes_mask.shape)

        # Adds possibility of an edge to the actual node closest neighbour in 8 directions
        for offset in [-1, 1]:
            adj_matrix = _add_edge_perhaps(
                x + offset, y, i, nodes_mask, labels_volume, adj_matrix
            )
            adj_matrix = _add_edge_perhaps(
                x, y + offset, i, nodes_mask, labels_volume, adj_matrix
            )
            for y_offset in [-1, 1]:
                adj_matrix = _add_edge_perhaps(
                    x + offset, y + y_offset, i, nodes_mask, labels_volume, adj_matrix
                )
    # # Verify number of non-zero potential connexions (max) for closest neighbours :
    # num_non_zero = 0
    # adj_matrix_1 = adj_matrix.flatten()
    # for i in adj_matrix_1:
    #     if i != 0:
    #         num_non_zero += 1
    # print(num_non_zero)

    return adj_matrix, node_indices


def build_weighted_graph(adj_matrix, node_indices, sh, axis_name):
    # Get directions depending if we are in axial, coronal or sagittal :
    sphere = _get_sphere_for_axis(axis_name)
    sf = sh_to_sf(sh, sphere, sh_order=12, basis_type="tournier07")
    sf[sf < 0.0] = 0.0
    sf /= np.max(sf, axis=(0, 1), keepdims=True)
    # sf = sf / np.max(sf, axis=(-1), keepdims=True)
    sf *= 0.5

    # print(sh.shape)
    weighted_graph = np.zeros_like(adj_matrix)
    x, y = np.unravel_index(node_indices, sh.shape[:2])

    # node traversal
    for it, node_row in enumerate(adj_matrix):
        nb_connections = np.count_nonzero(node_row)
        if nb_connections > 0:
            start_x, start_y = x[it], y[it]
            # which nodes are connected to every starting node:
            connected_xs, connected_ys = x[node_row > 0], y[node_row > 0]

            w_list = []
            for conn_idx in range(nb_connections):
                conn_x, conn_y = connected_xs[conn_idx], connected_ys[conn_idx]

                direction = np.array([[conn_x, conn_y]], dtype=float) - np.array(
                    [[start_x, start_y]], dtype=float
                )
                # The directions :

                dir_id = np.argmax(np.dot(direction, DIRECTIONS_2D.T))

                w = sf[start_x, start_y, dir_id] + sf[conn_x, conn_y, dir_id]
                w_list.append(w)

            weights = np.zeros((len(node_row),))
            weights[node_row > 0] = np.asarray(w_list)
            weighted_graph[it, :] = weights

    return weighted_graph, node_indices


def _get_sphere_for_axis(axis_name):
    directions = np.zeros((len(DIRECTIONS_2D), 3))
    if axis_name == "sagittal":
        directions[:, 1] = DIRECTIONS_2D[:, 0]
        directions[:, 2] = DIRECTIONS_2D[:, 1]
    elif axis_name == "coronal":
        directions[:, 0] = DIRECTIONS_2D[:, 0]
        directions[:, 2] = DIRECTIONS_2D[:, 1]
    elif axis_name == "axial":
        directions[:, 0] = DIRECTIONS_2D[:, 0]
        directions[:, 1] = DIRECTIONS_2D[:, 1]
    # print(directions)
    return Sphere(xyz=directions)


def _add_edge_perhaps(
    pos_x, pos_y, current_node, nodes_mask, labels_volume, adj_matrix
):
    if _is_valid_pos(pos_x, pos_y, nodes_mask):
        neighbor_label = labels_volume[pos_x, pos_y]
        adj_matrix[current_node, neighbor_label] = 1
    return adj_matrix


def _is_valid_pos(pos_x, pos_y, nodes_mask):
    if pos_x < 0 or pos_x >= nodes_mask.shape[0]:
        return False
    if pos_y < 0 or pos_y >= nodes_mask.shape[1]:
        return False
    return nodes_mask[pos_x, pos_y]
