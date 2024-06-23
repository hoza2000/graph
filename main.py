import networkx as nx
import matplotlib.pyplot as plt

from graph import generate_rnd_graph, visualize_graph
from algos import find_tsp_shortest_path, kmeans_partition_graph
from utils import calc_path_cost

def main():
    # Gen parameters
    num_nodes = 100
    connectivity_percentage = 0.3
    edge_weight_range = (1, 10)

    # TSP parameters
    cycle = False

    # k-means clustering parameters
    n_clusters = 5
    mandatory_node = None
    min_nodes = 0

    # Visualization parameters
    node_size = 700
    offset = 0.05
    just_paths_final = True

    G = generate_rnd_graph(num_nodes=num_nodes, connectivity_percentage=connectivity_percentage, edge_weight_range=edge_weight_range)
    paths = []
    # paths.append(find_tsp_shortest_path(G))

    graphs = kmeans_partition_graph(G, nx.get_node_attributes(G, 'pos'), n_clusters=n_clusters, mandatory_node=mandatory_node, min_nodes=min_nodes)

    for graph in graphs:
        visualize_graph(graph)

    for graph in graphs:
        if len(graph.nodes) > 1:
            paths.append(find_tsp_shortest_path(graph, cycle=cycle))

    for i in range(len(paths)):
        print(f"Path[{i}]: {paths[i]}", end=", ")
        print(f"Cost: {calc_path_cost(G, paths[i])}")

    visualize_graph(G, paths, just_paths=just_paths_final, node_size=node_size, offset=offset)

if __name__ == "__main__":
    main()