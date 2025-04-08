import numpy as np

def save_graph(G, output_base, copies=1):
    for i in range(copies):
        output_file = f"{output_base}_{i}.npz"  
        nodes = list(G.nodes())
        edges = list(G.edges())

        nodes_array = np.array(nodes)
        edges_array = np.array(edges)

        np.savez(output_file, nodes=nodes_array, edges=edges_array)

        print(f"Copie {i+1}/{copies} saved as '{output_file}'.")
        
