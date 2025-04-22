import time
import heapq
from travel_related import get_neighbors_diagonal


def dijkstra_stepwise(G, start, target, diagonal_mode="nondiagonal"):
    """
    Perform Dijkstra's algorithm in a stepwise manner on a given graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph where each node is typically a tuple (e.g., (x, y)).
        Edges may have a 'weight' attribute indicating traversal cost
        (default is 1).
    start : hashable
        The starting node for the pathfinding algorithm.
    target : hashable
        The target node to reach.
    diagonal_mode : str, optional
        Neighbor retrieval mode. Must be either:
        - "nondiagonal": only 4-directional neighbors (up, down, left, right)
        - "diagonal": include diagonal neighbors (8 directions total)
        Default is "nondiagonal".

    Returns
    -------
    evaluated_nodes : list
        The list of nodes in the order they were evaluated by the algorithm.
    path_to_current : list of list
        A list containing the reconstructed path from the start node to each
        node evaluated so far, in evaluation order.
    current_distance : float
        The final distance to the target node if found, or to the last
        evaluated node otherwise. Returns None if no path is found.
    """
    start_time = time.time()

    distances = {node: float('inf') for node in G.nodes()}
    distances[start] = 0
    previous_nodes = {node: None for node in G.nodes()}
    evaluated_nodes = []
    path_to_current = []
    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node not in evaluated_nodes:
            evaluated_nodes.append(current_node)

        temp_path = []
        node = current_node
        while node is not None:
            temp_path.append(node)
            if node in previous_nodes:
                node = previous_nodes[node]
            else:
                break
        temp_path.reverse()
        path_to_current.append(temp_path)

        if current_node == target:
            break

        if diagonal_mode == "diagonal":
            neighbors = list(get_neighbors_diagonal(current_node, G))
        else:
            neighbors = list(G.neighbors(current_node))

        for neighbor in neighbors:
            if neighbor not in evaluated_nodes:
                edge_weight = G[current_node][neighbor].get("weight", 1)
                new_distance = current_distance + edge_weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    if distances[target] == float('inf'):
        print("No path found.")
        return None, None, None

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of Dijkstra: {execution_time:.4f} seconds")

    return evaluated_nodes, path_to_current, current_distance
