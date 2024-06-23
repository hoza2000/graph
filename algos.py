import networkx as nx
from collections import deque
from networkx.algorithms.approximation import traveling_salesman_problem
from networkx.algorithms.approximation.traveling_salesman import christofides

import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def find_tsp_shortest_path(G, cycle=False):
    path = traveling_salesman_problem(G, cycle=cycle, method=christofides)
    return path

def kmeans_partition_graph(G, pos, n_clusters, mandatory_node=None, min_nodes=0):
    # Extract node positions into a NumPy array
    node_positions = np.array([pos[node] for node in G.nodes])
    nodes = list(G.nodes)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_positions)
    labels = kmeans.labels_

    # Create clusters
    clusters = defaultdict(list)
    for node, label in zip(nodes, labels):
        clusters[label].append(node)

    # Ensure each cluster meets the minimum size requirement
    while any(len(cluster) < min_nodes for cluster in clusters.values()):
        # Identify clusters that are too small and too large
        small_clusters = {label: cluster for label, cluster in clusters.items() if len(cluster) < min_nodes}
        large_clusters = {label: cluster for label, cluster in clusters.items() if len(cluster) > min_nodes}

        # Reassign nodes from large clusters to small clusters
        for small_label, small_cluster in small_clusters.items():
            while len(small_cluster) < min_nodes:
                # Find the largest cluster
                large_label, large_cluster = max(large_clusters.items(), key=lambda item: len(item[1]))
                if len(large_cluster) <= min_nodes:
                    break  # No more nodes can be reassigned without violating the minimum size requirement

                # Reassign a node from the largest cluster to the small cluster
                node_to_reassign = large_cluster.pop()
                small_cluster.append(node_to_reassign)
                clusters[small_label] = small_cluster
                clusters[large_label] = large_cluster

                # Update large_clusters
                if len(large_cluster) <= min_nodes:
                    del large_clusters[large_label]

    # Create subgraphs from clusters
    subgraphs = [G.subgraph(cluster_nodes).copy() for cluster_nodes in clusters.values()]

    # Ensure the mandatory node is in subgraphs
    if mandatory_node is not None:
        for subgraph in subgraphs:
            if mandatory_node not in subgraph.nodes:
                subgraph.add_node(mandatory_node)

    # Make subgraphs connected
    for subgraph in subgraphs:
        if not nx.is_connected(subgraph):
            nodes = list(subgraph.nodes)
            for node1 in nodes:
                for node2 in nodes:
                    if node1 != node2 and not nx.has_path(subgraph, node1, node2):
                        shortest_path = nx.shortest_path(G, node1, node2, weight="weight")
                        for i in range(len(shortest_path) - 1):
                            subgraph.add_edge(shortest_path[i], shortest_path[i + 1], weight=G[shortest_path[i]][shortest_path[i + 1]]['weight'])

    return subgraphs

def bfs_bitmasking_path(G):
    n = len(G.nodes)
    all_visited = (1 << n) - 1  # All nodes visited bitmask
    queue = deque([(node, 1 << i, 0, [node]) for i, node in enumerate(G.nodes)])  # (current_node, bitmask, path_length, path)
    memo = {(node, 1 << i): 0 for i, node in enumerate(G.nodes)}  # Memoization dictionary

    while queue:
        current_node, bitmask, path_length, path = queue.popleft()

        # If all nodes are visited, return the path length and path
        if bitmask == all_visited:
            return path_length, path

        # Explore neighbors
        for neighbor in G.neighbors(current_node):
            next_bitmask = bitmask | (1 << list(G.nodes).index(neighbor))
            if (neighbor, next_bitmask) not in memo or path_length + G[current_node][neighbor]['weight'] < memo[(neighbor, next_bitmask)]:
                memo[(neighbor, next_bitmask)] = path_length + G[current_node][neighbor]['weight']
                queue.append((neighbor, next_bitmask, path_length + G[current_node][neighbor]['weight'], path + [neighbor]))

    return float('inf'), []  # If no path visits all nodes, return infinity and an empty path