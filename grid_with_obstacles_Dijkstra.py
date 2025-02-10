import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import heapq
import random

# graph creation
rows, cols = 17, 17
grid = np.zeros((rows, cols))
G = nx.grid_2d_graph(rows, cols)

# initial and target nodes
start = (0, 0)
target = (12, 12)

# obstacles creation
obstacle_ratio = 0.2  
num_obstacles = int(rows * cols * obstacle_ratio)
all_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in [start, target]]
obstacles = random.sample(all_positions, num_obstacles)
for obs in obstacles:
    grid[obs] = 1 
#removing them form the graph
for obs in obstacles:
    if obs in G:
        G.remove_node(obs)

# Function of Dijkstra algorithm. Shown in the animation
def dijkstra_stepwise(G, start, target):
    # Initialisation
    distances = {node: float('inf') for node in G.nodes()}
    distances[start] = 0
    previous_nodes = {node: None for node in G.nodes()}
    evaluated_nodes = []  # List of evaluated nodes
    path_to_current = []  # History of the path to the current node

    # priority queue on the non-evaluated nodes
    priority_queue = [(0, start)]  # (distance, node)
    heapq.heapify(priority_queue)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
    
        if current_node not in evaluated_nodes:
            evaluated_nodes.append(current_node)

        # Reconstruction of the path to the current node
        temp_path = []
        node = current_node
        while node is not None:
            temp_path.append(node)
            node = previous_nodes[node]
        temp_path.reverse()

        path_to_current.append(temp_path)  

        if current_node == target:
            break

        # Evaluation of the current nodes neighbors
        for neighbor in G.neighbors(current_node):
            if neighbor not in evaluated_nodes:
                new_distance = current_distance + 1
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    return evaluated_nodes, path_to_current

# Retrieving the results of Dijkstra's algorithm
evaluated_nodes, path_history = dijkstra_stepwise(G, start, target)

# Creation of the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(grid, cmap="Greys", origin="upper") # The origin is set on the top left corner

# Display of the grid
ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
ax.grid(True, which="minor", color="black", linewidth=0.5)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

# Display of the start and target nodes
start_rect = patches.Rectangle((start[1] - 0.5, start[0] - 0.5), 1, 1, facecolor="green", alpha=0.8)
target_rect = patches.Rectangle((target[1] - 0.5, target[0] - 0.5), 1, 1, facecolor="yellow", alpha=0.8)
ax.add_patch(start_rect)
ax.add_patch(target_rect)

# Initiate the plots
evaluated_patches = []
path_patches = []

# List in use for the confetti animationn
confetti_patches = []
confetti_velocities = [] 

# Update function for the animation
def update(frame):
    # Diplays the evaluated nodes in blue
    if frame < len(evaluated_nodes):
        node = evaluated_nodes[frame]
        if node != start:
            rect = patches.Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, facecolor="blue", alpha=0.6)
        else:
            rect = patches.Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, facecolor="green", alpha=0.6)
        ax.add_patch(rect)
        evaluated_patches.append(rect)
        

    # Delete previous path
    if frame < len(path_history):
        for patch in path_patches:
            patch.remove()
        path_patches.clear()

    # Displays the current shortest path 
    if frame < len(path_history):
        for node in path_history[frame]:
            if node != start:
                rect = patches.Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, facecolor="red", alpha=0.8)
            else:
                rect = patches.Rectangle((node[1] - 0.5, node[0] - 0.5), 1, 1, facecolor="green", alpha=0.8)
            ax.add_patch(rect)
            path_patches.append(rect)

            
    # Getting rid of the previous shortest paths
    if frame == len(path_history) - 1:
        for patch in path_patches:
            patch.set_facecolor("red")  
        path_patches[-1].set_facecolor("cyan") # When the target is reached, becomes cyan

    # confetti effect when we reach the target node
    if frame == len(path_history) - 1:
        # creation of particules around the final node
        confetti_patches.clear()  
        confetti_velocities.clear()  
        for _ in range(50):  # 50 particules
            offset_x = random.uniform(-0.5, 0.5)
            offset_y = random.uniform(-0.5, 0.5)
            color = np.random.rand(3,)  # Random color
            confetti_rect = patches.Circle(
                (target[1] + offset_x, target[0] + offset_y),
                radius=0.1, facecolor=color, alpha=0.7
            )
            ax.add_patch(confetti_rect)
            confetti_patches.append(confetti_rect)
            
            # Define velocity of the particales
            velocity = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]  # Mouvement dans toutes les directions
            confetti_velocities.append(velocity)

    # Animation of the particales
    for i, patch in enumerate(confetti_patches):
        new_center = (
            patch.center[0] + confetti_velocities[i][0],  # X movement
            patch.center[1] + confetti_velocities[i][1]   # Y movement
        )
        patch.set_center(new_center)

    return evaluated_patches + path_patches + confetti_patches

# creation of the animation
ani = animation.FuncAnimation(fig, update, frames=len(evaluated_nodes) + 20, interval=100, blit=False, repeat=False)

plt.show()