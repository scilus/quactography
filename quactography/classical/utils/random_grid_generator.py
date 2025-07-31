import numpy as np
import random
import networkx as nx

def generer_grille(size, obstacle_mode="ratio", obstacle_ratio=0.2, obstacle_number=20):
    """
    Generate a random 2D grid and its corresponding NetworkX graph.

    Parameters
    ----------
    size : int
        Size of the grid (the grid will be of shape `size x size`).
        Must be a positive integer.
    obstacle_mode : str
        Strategy used to place obstacles in the grid. Options are:
        - "ratio": place a proportion of obstacles based on `obstacle_ratio`
        - "number": place a fixed number of obstacles based on `obstacle_number`
    obstacle_ratio : float
        Used only if `obstacle_mode` is "ratio". Defines the proportion of cells
        to be turned into obstacles. Must be a float between 0 and 1.
    obstacle_number : int
        Used only if `obstacle_mode` is "number". Defines the exact number of obstacles
        to place in the grid. Must be a positive integer.

    Returns
    -------
    grid : np.ndarray
        A 2D NumPy array of shape `(size, size)` representing the grid, where `1`
        denotes an obstacle and `0` a free cell.
    G : networkx.Graph
        A 2D grid graph where each node is a tuple `(x, y)`. Edges connect 4-neighboring
        nodes (up, down, left, right). Nodes corresponding to obstacles are removed.
    """
    n = size
    grid = np.zeros((n, n))
    G = nx.grid_2d_graph(n, n)

    obstacles = set()

    if obstacle_mode == "ratio":
        num_obstacles = int(n * n * obstacle_ratio)
    else:
        num_obstacles = obstacle_number

    while len(obstacles) < num_obstacles:
        x, y = random.randint(0, n - 1), random.randint(0, n - 1)
        if (x, y) != (0, 0) and (x, y) != (n - 1, n - 1):
            obstacles.add((x, y))
            grid[x, y] = 1  # Add an obstacle
            if (x, y) in G:
                G.remove_node((x, y))

    return grid, G
    
