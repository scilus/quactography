from scipy.ndimage import binary_erosion
import numpy as np


def get_output_nodes(mask, entry_node, propagation_direction, angle_rad):
    """
    Get output nodes from a hotspot mask. The input mask is expected
    to be hole-free. Output nodes are edge nodes found within angle_rad
    radians from the propagation direction.
    Parameters
    ----------
    mask: ndarray
        Mask defining the hotspot mask.
    entry_node: ndarray (1, 3)
        Node we enter the hotspot from.
    propagation_direction: ndarray (1, 3)
        Direction along which propagation is done.
    angle_rad: float
        Aperture angle in radians.
    Returns
    -------
    indices: ndarray (N, 3)
        Array of output node indices.
    """
    if not mask[entry_node[0], entry_node[1], entry_node[2]]:
        raise ValueError('Entry node should be included in mask.')
    edges = np.logical_and(mask, ~binary_erosion(mask))
    edge_indices = np.argwhere(edges)
    direction_to_edge = edge_indices - np.reshape(entry_node, (1, 3))
    direction_norms = np.linalg.norm(direction_to_edge, axis=-1)

    # remove edge position corresponding to entry_point
    direction_to_edge = direction_to_edge[direction_norms > 0]
    edge_indices = edge_indices[direction_norms > 0]
    direction_norms = direction_norms[direction_norms > 0]

    direction_to_edge = direction_to_edge / direction_norms[:, None]

    propagation_direction_norm = np.linalg.norm(propagation_direction)
    if propagation_direction_norm != 1.0:
        propagation_direction = propagation_direction / propagation_direction_norm

    directions_cos_angle = direction_to_edge.dot(propagation_direction.reshape((3, 1)))
    propagation_cos_angle = np.cos(angle_rad)
    is_valid_edge = directions_cos_angle > propagation_cos_angle
    valid_indices = edge_indices[is_valid_edge.reshape((-1,))]
    return valid_indices


if __name__ == '__main__':
    # little demo code here
    mask = np.zeros((25, 25, 25), dtype=bool)
    mask[5:10, 5:10, 5:10] = True
    entry_node = np.array([5, 5, 5])
    propagation_direction = np.array([0, 1, 1])
    angle_rad = np.pi / 8

    indices = get_output_nodes(mask, entry_node, propagation_direction, angle_rad)
    print(indices)