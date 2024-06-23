import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def generate_rnd_graph(num_nodes=10, connectivity_percentage=0.3, edge_weight_range=(1, 10)):
    # Create a random geometric graph
    radius = np.sqrt(connectivity_percentage)
    G = nx.random_geometric_graph(num_nodes, radius)

    # Ensure the graph is connected by adding an MST
    if not nx.is_connected(G):
        G = nx.minimum_spanning_tree(G)

    # Add varied weights to edges
    pos = nx.get_node_attributes(G, 'pos')
    for (u, v) in G.edges():
        # Calculate Euclidean distance
        dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        
        # Introduce variability in weights to simulate different road types
        road_type_factor = random.choice([0.8, 1.0, 1.2])  # Simulate local roads, main roads, highways
        weight = dist * road_type_factor * (edge_weight_range[1] - edge_weight_range[0]) + edge_weight_range[0]
        G.edges[u, v]['weight'] = round(weight, 1)

    return G

def visualize_graph(G, paths=None, just_paths=False, node_size=700, offset=0.05):

    colors = {"red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta"}

    # Visualize the graph and the shortest path
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Draw all nodes (and edges)
    if not just_paths:
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', font_size=10, font_color='black')
    else:
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue')
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    # Draw the paths with arrows
    if paths:
        for path in paths:
            arrow_counter = 1
            # Generate a random dark color for the path
            # rnd_color = np.random.rand(3,) * 0.5
            rnd_color = random.choice(list(colors))
            colors.remove(rnd_color)
            path_edges = list(zip(path, path[1:]))
            for u, v in path_edges:
                x_start, y_start = pos[u]
                x_end, y_end = pos[v]
                dx = x_end - x_start
                dy = y_end - y_start

                lenght = np.sqrt(dx ** 2 + dy ** 2)

                dx_norm = dx / lenght
                dy_norm = dy / lenght

                x_end_offset = x_end - offset * dx_norm
                y_end_offset = y_end - offset * dy_norm

                plt.arrow(x_start, y_start, x_end_offset - x_start, y_end_offset - y_start,
                    color=rnd_color, shape='full', lw=2, length_includes_head=True, head_width=0.025, head_length=0.025)
                
                if just_paths:
                    continue

                # Annotate the midpoint with the arrow sequence number
                num_offset = 0.2  # Offset for numbering to be visible
                num_x = x_end - num_offset * dx
                num_y = y_end - num_offset * dy
                plt.text(num_x, num_y, str(arrow_counter), fontsize=12, color=rnd_color, ha='center',
                         bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
                arrow_counter += 1

    # draw egde weights
    if not just_paths:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)

    plt.title("Graph Visualization")
    plt.show()
