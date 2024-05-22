from collections import deque

def node_count(edgelist: list) -> int:
    """For the given graph, the task is to
    compute the number of nodes of the graph.

    Parameters:
        edgelist (list): list of edges of the graph
    Returns:
        int: number of nodes of the graph
    """

    nodes = set() # Stores nodes
    for edge in edgelist:
        if edge[0] not in nodes:
            nodes.add(edge[0])

        if edge[1] not in nodes:
            nodes.add(edge[1])

    return len(nodes)


def edge_count(edgelist: list) -> int:
    """For the given graph, the task is to
    compute the number of edges of the graph.

    Parameters:
        edgelist (list): list of edges of the graph
    Returns:
        int: number of edges of the graph
    """

    return len(edgelist)


def node_degree(edgelist: list, node: int) -> int:
    """For the given graph, the task is to
    compute the degree of a given node.

    Parameters:
        edgelist (list): list of edges of the graph
        node (int): input node
    Returns:
        int: degree of input node
    """

    degree = 0
    for edge in edgelist:
        if edge[0] == node or edge[1] == node:
            degree = degree + 1

    return degree


def connected_nodes(edgelist: list, node: int) -> list:
    """For the given graph, the task is to
    find the neighbors of a given node.

    Parameters:
        edgelist (list): list of edges of the graph
        node (int): input node
    Returns:
        list: list of neighbors of node
    """

    neighbors = list()
    for edge in edgelist:
        if edge[0] == node:
            neighbors.append(edge[1])

        if edge[1] == node:
            neighbors.append(edge[0])

    return neighbors


def connected_components_count(edgelist: list) -> int:
    """For the given graph, the task is to
    compute the number of connected components.

    Parameters:
        edgelist (list): list of edges of the graph
    Returns:
        int: number of connected components
    """
    neighbors = dict()
    for edge in edgelist:
        if edge[0] in neighbors:
            neighbors[edge[0]].append(edge[1])
        else:
            neighbors[edge[0]] = [edge[1]]

        if edge[1] in neighbors:
            neighbors[edge[1]].append(edge[0])
        else:
            neighbors[edge[1]] = [edge[0]]


    visited = set()
    number_connected_components = 0
    for node in neighbors:
        if node not in visited:
            visited.add(node)
            queue = [node]

            while len(queue) > 0:
                node = queue.pop()
                for neighbor in neighbors[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            number_connected_components = number_connected_components + 1

    return number_connected_components


def cycle_check(edgelist: list) -> bool:
    """For the given graph, the task is to
    find whether the graph contains a cycle.

    Parameters:
        edgelist (list): list of edges of the graph
    Returns:
        bool: whether the graph contains a cycle
    """

    nodes = list()
    neighbors = dict()
    for edge in edgelist:
        if edge[0] in neighbors:
            neighbors[edge[0]].append(edge[1])
        else:
            neighbors[edge[0]] = [edge[1]]
            nodes.append(edge[0])

        if edge[1] in neighbors:
            neighbors[edge[1]].append(edge[0])
        else:
            neighbors[edge[1]] = [edge[0]]
            nodes.append(edge[1])


    visited = set()
    for node in nodes:
        if node not in visited:
            visited.add(node)
            connected = []
            queue = [node]

            while len(queue) > 0:
                current_node = queue.pop()
                connected.append(current_node)
                for neighbor in neighbors[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            sum_of_degrees = 0
            for current_node in connected:
                sum_of_degrees = sum_of_degrees + len(neighbors[current_node])

            number_of_edges_of_connected_component = sum_of_degrees//2
            if number_of_edges_of_connected_component > len(connected) - 1:
                return True

    return False


def connectivity(edgelist: list, source_node: int, target_node: int) -> bool:
    """For the given graph, the task is to
    find whether there is a path from source_node
    to target_node. If there exists such a path
    return True else return False.

    Parameters:
        edgelist (list): list of edges of the graph
        source_node (int): source node
        target_node (int): target node
    Returns:
        bool: whether there is a path from
        source node to target node
    """

    visited = set() # Keeps track of visited nodes
    to_visit = [source_node] # Stores nodes to be visited
    while len(to_visit) > 0:
        node = to_visit.pop()
        visited.add(node)
        for edge in edgelist:
            if edge[0] == node:
                neighbor = edge[1]
                if neighbor == target_node:
                    return True
                else:
                    if neighbor not in visited:
                        to_visit.append(neighbor)
            elif edge[1] == node:
                neighbor = edge[0]
                if neighbor == target_node:
                    return True
                else:
                    if neighbor not in visited:
                        to_visit.append(neighbor)

    return False


def is_bipartite(edgelist: list) -> bool:
    """For the given graph, the task is to
    find whether the graph is bipartite.
    If it is bipartite return True else
    return False.

    Parameters:
        edgelist (list): list of edges of the graph
    Returns:
        bool: whether the graph is bipartite or not
    """

    nodes = list()
    neighbors = dict()
    for edge in edgelist:
        if edge[0] in neighbors:
            neighbors[edge[0]].append(edge[1])
        else:
            neighbors[edge[0]] = [edge[1]]
            nodes.append(edge[0])

        if edge[1] in neighbors:
            neighbors[edge[1]].append(edge[0])
        else:
            neighbors[edge[1]] = [edge[0]]
            nodes.append(edge[1])

    color = {}
    for node in nodes: # handle disconnected graphs
        if node in color or len(neighbors[node])==0:
            continue
        queue = [node]
        color[node] = 1 # nodes seen with color (1 or 0)
        while len(queue) > 0:
            v = queue.pop()
            c = 1 - color[v] # opposite color of node v
            for w in neighbors[v]:
                if w in color:
                    if color[w] == color[v]:
                        return False
                else:
                    color[w] = c
                    queue.append(w)

    return True


def shortest_path(edgelist: list, source_node: int, target_node: int) -> int:
    """For the given graph, the task is to
    find the length of the shortest path from
    source_node to target_node. The function
    returns the length of that path.

    Parameters:
        edgelist (list): list of edges of the graph
        source_node (int): source node
        target_node (int): target node
    Returns:
        int: length of path from
        source node to target node
    """
    nodes = []
    neighbors = {}
    for edge in edgelist:
        if edge[0] not in neighbors:
            nodes.append(edge[0])
            neighbors[edge[0]] = [edge[1]]
        else:
            neighbors[edge[0]].append(edge[1])

        if edge[1] not in neighbors:
            nodes.append(edge[1])
            neighbors[edge[1]] = [edge[0]]
        else:
            neighbors[edge[1]].append(edge[0])

    queue = set()
    distance_to_source = {}
    for node in nodes:
        distance_to_source[node] = float('inf')
        queue.add(node)

    distance_to_source[source_node] = 0
    while len(queue) > 0:
        min_dist = float('inf')
        closest_node = None
        for node in queue:
            if distance_to_source[node] <= min_dist:
                min_dist = distance_to_source[node]
                closest_node = node

        if closest_node == target_node:
            return distance_to_source[closest_node]

        queue.remove(closest_node)

        for neighbor in neighbors[closest_node]:
            if neighbor in queue:
                alt = distance_to_source[closest_node] + 1
                if alt < distance_to_source[neighbor]:
                    distance_to_source[neighbor] = alt

    return -1


def is_valid_topological_sort(edgelist, proposed_order):
    nodes = set()
    for edge in edgelist:
        nodes.add(edge[1])
        nodes.add(edge[0])

    if len(proposed_order) != len(nodes):
        return False

    successors = dict()
    for node in nodes:
        successors[node] = []

    for edge in edgelist:
        successors[edge[0]].append(edge[1])

    visited = set()
    for node in proposed_order:
        for successor in successors[node]:
            if successor in visited:
                return False

        visited.add(node)

    return True


def is_valid_minimum_spanning_tree(edgelist, proposed_edges):
    nodes = set()
    for edge in edgelist:
        if edge[0] not in nodes:
            nodes.add(edge[0])
        if edge[1] not in nodes:
            nodes.add(edge[1])

    number_of_nodes = len(nodes)

    node_to_set = {}
    for node in nodes:
        node_to_set[node] = node

    edge_set = set(edgelist)
    # print(proposed_edges)
    for edge in proposed_edges:
        if (edge[0], edge[1]) not in edge_set and (edge[1], edge[0]) not in edge_set:
            return False

    edgelist_new = []
    edgelist_new.extend(proposed_edges)
    for edge in edgelist:
        if edge not in edgelist_new:
            edgelist_new.append(edge)

    mst_edges = []
    for edge in edgelist_new:
        if node_to_set[edge[0]] != node_to_set[edge[1]]:
            mst_edges.append(edge)
            prev_set = node_to_set[edge[1]]
            for node in nodes:
                if node_to_set[node] == prev_set:
                    node_to_set[node] = node_to_set[edge[0]]

    mst_edges = set(mst_edges)
    # import pdb; pdb.set_trace()
    if len(mst_edges) != len(proposed_edges):
        return False
    elif len(mst_edges) != len(mst_edges.intersection(proposed_edges)):
        return False

    return True
