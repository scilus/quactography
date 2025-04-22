def get_neighbors_diagonal(node, G):
    """
    Yield all 8-directional neighbors of a node that exist in the graph G.

    Parameters
    ----------
    node : tuple
        The current node position, typically a coordinate (x, y).
    G : networkx.Graph
        The graph in which neighbors should be checked.

    Yields
    ------
    neighbor : tuple
        A valid neighboring node in G that is adjacent to `node` in any of
        the 8 possible directions (N, S, E, W, and diagonals).
    """
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]
    for dx, dy in directions:
        neighbor = (node[0] + dx, node[1] + dy)
        if neighbor in G.nodes():
            yield neighbor


def heuristic(current, target):
    """
    Compute the Chebyshev distance between two points on a grid.

    Parameters
    ----------
    current : tuple
        The current node position (x, y).
    target : tuple
        The target node position (x, y).

    Returns
    -------
    int
        The Chebyshev distance between `current` and `target`.
        This is appropriate for 8-directional movement.
    """
    return max(abs(current[0] - target[0]), abs(current[1] - target[1]))
