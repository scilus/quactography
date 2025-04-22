import numpy as np

def save_graph(G, output_base, copies=1):
    for i in range(copies):
        output_file = f"{output_base}_{i}.npz"  
        nodes = list(G.nodes())
        edges = list(G.edges())

        nodes_array = np.array(nodes)
        edges_array = np.array(edges)

        np.savez(output_file, nodes=nodes_array, edges=edges_array)

        print(f"✅ Copie {i+1}/{copies} saved as '{output_file}'.")

def load_the_graph(file_path):
    if file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            data = json.load(f)

        G = nx.Graph()  
        for node in data["nodes"]:
            G.add_node(tuple(node))  

        for edge in data["edges"]:
            node1, node2 = tuple(map(tuple, edge[:2])) 
            weight = edge[2] if len(edge) > 2 else 1 
            G.add_edge(node1, node2, weight=weight)

        return G

    elif file_path.endswith(".npz"):
        data = np.load(file_path, allow_pickle=True)
        adjacency_matrix = data['adjacency_matrix']
        node_indices = [
            tuple(node) if isinstance(node, (list, np.ndarray, tuple)) else (node,) for node in data['node_indices']
        ]

        # Ajouter un 0 aux nœuds incomplets
        node_indices = [(n[0], 0) if len(n) == 1 else n for n in node_indices]

        print(f"Loaded adjacency matrix of shape {adjacency_matrix.shape} with {len(node_indices)} nodes.")

        G = nx.Graph()
        for i in range(len(node_indices)):
            G.add_node(node_indices[i])

        for i in range(len(adjacency_matrix)):
            for j in range(i+1, len(adjacency_matrix[i])):
                if adjacency_matrix[i, j] > 0:
                    G.add_edge(node_indices[i], node_indices[j], weight=adjacency_matrix[i, j])

        print(f"Nœuds présents dans G: {list(G.nodes())}")
        print(f"Graphe chargé avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes.")
        return G

    else:
        raise ValueError("Unsupported file format. Use either .json or .npz")
