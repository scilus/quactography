def get_neighbors_diagonal(node, G):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),(1, 1), (-1, -1), (1, -1), (-1, 1) ]
    for dx, dy in directions:
            neighbor = (node[0] + dx, node[1] + dy)
            if neighbor in G.nodes():
                yield neighbor
