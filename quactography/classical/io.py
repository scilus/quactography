def save_graph(G, output_base, copies=1):
    for i in range(copies):
        output_file = f"{output_base}_{i}.json"  
        data = {
            "nodes": list(G.nodes()),
            "edges": list(G.edges())
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Copie {i+1}/{copies} saved as '{output_file}'.")
