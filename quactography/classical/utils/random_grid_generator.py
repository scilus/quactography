def generer_grille(size, obstacle_mode="ratio", obstacle_ratio=0.2, obstacle_number=20):
    n = size
    grid = np.zeros((n, n))  
    G = nx.grid_2d_graph(n, n)  

    obstacles = set()

    if obstacle_mode == "ratio": 
        num_obstacles = int(n * n * obstacle_ratio)
    else:
        num_obstacles = obstacle_number

    while len(obstacles) < num_obstacles:
        x, y = random.randint(0, n-1), random.randint(0, n-1)
        if (x, y) != (0, 0) and (x, y) != (n-1, n-1):
            obstacles.add((x, y))
            grid[x, y] = 1  # Ajouter un obstacle
            if (x, y) in G:
                G.remove_node((x, y))
        
    return grid, G
