def dijkstra_stepwise(G, start, target, diagonal_mode="nondiagonal"):
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

        # Reconstruction du chemin actuel
        temp_path = []
        node = current_node
        while node is not None:
            temp_path.append(node)
            if node in previous_nodes:  # ✅ Vérification pour éviter KeyError
                node = previous_nodes[node]
            else:
                break  # ✅ Stopper si le nœud n'est pas connu
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
                edge_weight = G[current_node][neighbor].get("weight", 1)  # Récupérer le poids réel
                new_distance = current_distance + edge_weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    if distances[target] == float('inf'):
        print("⚠️ Aucun chemin trouvé entre le point de départ et l'arrivée.")
        return None, None  # Retourner None pour indiquer l'absence de chemin

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of Dijkstra: {execution_time:.4f} secondes")
    return evaluated_nodes, path_to_current, current_distance
