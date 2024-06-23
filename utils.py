import networkx as nx

def calc_path_cost(G, path):
    cost = 0
    for i in range(len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            print(f"Edge {path[i]}-{path[i + 1]} does not exist")
            return None
        cost += G.get_edge_data(path[i], path[i + 1])['weight']
    return cost