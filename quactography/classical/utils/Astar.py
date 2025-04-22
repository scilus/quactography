        
def astar_stepwise(G, start, target, diagonal_mode="nondiagonal"):
    start_time = time.time()
    g_scores = {node: float('inf') for node in G.nodes()}
    g_scores[start] = 0
        
    f_scores = {node: float('inf') for node in G.nodes()}
    f_scores[start] = heuristic(start, target)
        
    previous_nodes = {node: None for node in G.nodes()}
    evaluated_nodes = []
    path_to_current = []
    priority_queue = [(f_scores[start], start)]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_f_score, current_node = heapq.heappop(priority_queue)
            
        if current_node not in evaluated_nodes:
            evaluated_nodes.append(current_node)
            
        temp_path = []
        node = current_node
        while node is not None:
            temp_path.append(node)
            node = previous_nodes[node]
        temp_path.reverse()
        path_to_current.append(temp_path)
            
        #print(f"Step {len(evaluated_nodes)}: Evaluated {current_node}, Path: {temp_path}")

        if current_node == target:
            break

        if diagonal_mode == "diagonal":
            neighbors = list(get_neighbors_diagonal(current_node, G))
        elif diagonal_mode == "nondiagonal":
            neighbors = list(G.neighbors(current_node))

        for neighbor in neighbors:
            edge_weight = G[current_node][neighbor].get("weight", 1)
            tentative_g_score = g_scores[current_node] + edge_weight
            if tentative_g_score < g_scores[neighbor]:
                previous_nodes[neighbor] = current_node
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + heuristic(neighbor, target)
                heapq.heappush(priority_queue, (f_scores[neighbor], neighbor))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of A* : {execution_time:.4f} secondes")
    return evaluated_nodes, path_to_current, current_f_score
