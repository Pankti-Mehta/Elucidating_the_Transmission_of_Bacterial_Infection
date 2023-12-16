# Firstname Lastname
# NetID
# COMP 182 Spring 2021 - Homework 5, Problem 3

# You may NOT import anything apart from already imported libraries.
# You can use helper functions from provided.py, but they have
# to be copied over here.

from typing import Tuple
from collections import *
from copy import *


def bfs(graph, startnode):
    """
        Perform a breadth-first search on digraph graph starting at node startnode.

        Arguments:
        graph -- directed graph
        startnode - node in graph to start the search from

        Returns:
        The distances from startnode to each node
    """
    dist = {}

    # Initialize distances
    for node in graph:
        dist[node] = float('inf')
    dist[startnode] = 0

    # Initialize search queue
    queue = deque([startnode])

    # Loop until all connected nodes have been explored
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist

def compute_rdmst(graph, root):
    """
        This function checks if:
        (1) root is a node in digraph graph, and
        (2) every node, other than root, is reachable from root
        If both conditions are satisfied, it calls compute_rdmst_helper
        on (graph, root).

        Since compute_rdmst_helper modifies the edge weights as it computes,
        this function reassigns the original weights to the RDMST.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node id.

        Returns:
        An RDMST of graph rooted at r and its weight, if one exists;
        otherwise, nothing.
    """

    if root not in graph:
        print("The root node does not exist")
        return

    distances = bfs(graph, root)
    for node in graph:
        if distances[node] == float('inf'):
            print("The root does not reach every other node in the graph")
            return

    rdmst = compute_rdmst_helper(graph, root)

    # reassign the original edge weights to the RDMST and computes the total
    # weight of the RDMST
    rdmst_weight = 0
    for node in rdmst:
        for nbr in rdmst[node]:
            rdmst[node][nbr] = graph[node][nbr]
            rdmst_weight += rdmst[node][nbr]

    return (rdmst, rdmst_weight)

def compute_rdmst_helper(graph, root):
    """
        Computes the RDMST of a weighted digraph rooted at node root.
        It is assumed that:
        (1) root is a node in graph, and
        (2) every other node in graph is reachable from root.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node in graph.

        Returns:
        An RDMST of graph rooted at root. The weights of the RDMST
        do not have to be the original weights.
        """

    # reverse the representation of graph
    rgraph = reverse_digraph_representation(graph)

    # Step 1 of the algorithm
    modify_edge_weights(rgraph, root)

    # Step 2 of the algorithm
    rdst_candidate = compute_rdst_candidate(rgraph, root)

    # compute a cycle in rdst_candidate
    cycle = compute_cycle(rdst_candidate)

    # Step 3 of the algorithm
    if not cycle:
        return reverse_digraph_representation(rdst_candidate)
    else:
        # Step 4 of the algorithm

        g_copy = deepcopy(rgraph)
        g_copy = reverse_digraph_representation(g_copy)

        # Step 4(a) of the algorithm
        (contracted_g, cstar) = contract_cycle(g_copy, cycle)
        # cstar = max(contracted_g.keys())

        # Step 4(b) of the algorithm
        new_rdst_candidate = compute_rdmst_helper(contracted_g, root)

        # Step 4(c) of the algorithm
        rdmst = expand_graph(reverse_digraph_representation(rgraph), new_rdst_candidate, cycle, cstar)

        return rdmst

def reverse_digraph_representation(graph: dict) -> dict:
    reversed = {}
    for node in graph.keys():
        reversed[node] = {}
    for node in graph.keys():
        for node_nbr in graph[node]:
            reversed[node_nbr][node] = graph[node][node_nbr]
    return reversed

#test case
#g1 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
#4: {1: 4}, 5: {}}
#g1 = {0:{1:0}, 1:{2:0}, 2:{1:0}}
#print(reverse_digraph_representation(g1))

def modify_edge_weights(rgraph: dict, root: int) -> None:
    smallest_weight = float('inf')
    for node in rgraph.keys():
        for nbr in rgraph[node]:
            if rgraph[node][nbr] < smallest_weight:
                smallest_weight = rgraph[node][nbr]
#subtracting smallest_weight from each edge
    for node in rgraph.keys():
        for nbr in rgraph[node]:
            rgraph[node][nbr] = rgraph[node][nbr] - smallest_weight

#test case
# g2 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
# 4: {1: 4}, 5: {}}
# print(modify_edge_weights(g2,1))

def compute_rdst_candidate(rgraph: dict, root: int) -> dict:
    #reverse graph
    smallest_edges = {}
    for node in rgraph.keys():
        smallest_edges[node] = {}
        for neighbor, weight in rgraph[node].items():
            if neighbor not in smallest_edges[node] or weight < smallest_edges[node][neighbor]:
                smallest_edges[node] = {neighbor:weight}
    smallest_edges[root] = {}
    return smallest_edges
# g1 = {0:{1:0}, 1:{2:0}, 2:{1:0}}
# print(compute_rdst_candidate(g1, 0))

#test case
# g2 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
# 4: {1: 4}, 5: {}}
# print(compute_rdst_candidate(g2,1))

# #Praise
# test_rgraph = {1: {0: 16, 4: 0}, 2: {0: 2, 1: 0}, 3: {0: 12, 2: 0}, 5: {1: 8, 3: 0}, 4: {2: 16, 3: 0}, 0: {}}
# root = 0
# print(compute_rdst_candidate(test_rgraph, 0))

def compute_cycle(rdst_candidate: dict) -> tuple:
    def dfs(node, visited, parent, stack):
        visited.add(node)
        stack.add(node)
        for neighbor in rdst_candidate[node]:
            if neighbor not in visited:
                result = dfs(neighbor, visited, node, stack)
                if result:
                    return result
            elif neighbor != parent and neighbor in stack:
                return tuple(stack)
        stack.remove(node)
        return None

    visited = set()
    for node in rdst_candidate:
        if node not in visited:
            stack = set()
            result = dfs(node, visited, None, stack)
            if result:
                return result
    return None

#test case
# g2 = {0:{}, 1:{2:10}, 2:{3:10}, 3:{1:10}}
# g1 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
# 4: {1: 4}, 5: {}}
# print(compute_cycle(g1))

def contract_cycle(graph: dict, cycle: tuple) -> Tuple[dict, int]:
    cyc_outgoing_nodes = set()
    cyc_incoming_nodes = set()
    #populating cyc_outgoing_nodes and cyc_incoming_nodes
    for cyc in cycle:
        for nbr in graph[cyc]:
            if nbr not in cycle:
                cyc_outgoing_nodes.add(nbr)
    for node in graph:
        for nbr in graph[node]:
            if nbr in cycle and node not in cycle:
                cyc_incoming_nodes.add(node)
    contract_graph = deepcopy(graph)
    cstar = max(contract_graph.keys()) + 1
    for node in cycle:
        if node in contract_graph:
            contract_graph.pop(node)
    for node in cyc_incoming_nodes:
        for nbr in contract_graph[node].copy():
            if nbr in cycle:
                contract_graph[node].pop(nbr)
    contract_graph[cstar] = {}
    for node in cyc_outgoing_nodes:
        min_outgoing_weight = float('inf')
        for cyc in cycle:
            if node in graph[cyc]:
                min_outgoing_weight = min(min_outgoing_weight ,graph[cyc][node])
        contract_graph[cstar][node] = min_outgoing_weight
    for node in cyc_incoming_nodes:
        min_incoming_weight = float('inf')
        for nbr in graph[node]:
            if nbr in cycle:
                min_incoming_weight = min(min_incoming_weight, graph[node][nbr])
        contract_graph[node][cstar] = min_incoming_weight
    return contract_graph, cstar
# Etest1 = {0: {1: 5, 2: 4}, 1: {2: 2}, 2: {1: 2}}
# print(contract_cycle(Etest1, (1,2)))
# # Expect = {0: {3: 4}, 3: {}}, 3
# Etest2 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0}}
# print(contract_cycle(Etest2, compute_cycle(Etest2)))
# # Expect = {1: {2: 2.1, 4: 9.1, 6: 1.0},
# #           2: {1: 2.1, 4: 17, 6: 1.0},
# #           4: {1: 9.1, 2: 17.1, 6: 16},
# #           6: {1: 1.0, 2: 1.0, 4: 16.0}}, 6
# g = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# print(contract_cycle(g,(1,4,3,2)))

def expand_graph(graph: dict, rdst_candidate: dict, cycle: tuple, cstar: int) -> dict:
    expand_graph = deepcopy(rdst_candidate)
    expand_graph.pop(cstar)
    for node in expand_graph:
        if cstar in expand_graph[node]:
            expand_graph[node].pop(cstar)
    for node in cycle:
        expand_graph[node] = {}
    vstar = None
    for node in rdst_candidate:
        if cstar in rdst_candidate[node]:
            # find which node in cycle node is connected to
            incoming_cycle_edge = float('inf')
            for node_graph_nbr in graph[node]:
                if node_graph_nbr in cycle:
                    if graph[node][node_graph_nbr] < incoming_cycle_edge:
                        vstar = node_graph_nbr
                        incoming_cycle_edge = graph[node][node_graph_nbr]
            expand_graph[node][vstar] = rdst_candidate[node][cstar]
    for cstar_nbr in rdst_candidate[cstar]:
        # find which node in cycle cstar_nbr is connected to
        starting_node = None
        outgoing_cycle_edge = float('inf')
        for cyc in cycle:
            if cstar_nbr in graph[cyc]:
                if graph[cyc][cstar_nbr] < outgoing_cycle_edge:
                    starting_node = cyc
                    outgoing_cycle_edge = graph[cyc][cstar_nbr]
        expand_graph[starting_node][cstar_nbr] = rdst_candidate[cstar][cstar_nbr]
    i = len(cycle) - 1
    while i > 0:
        if cycle[i-1] != vstar and cycle[i] in expand_graph and cycle[i-1] in expand_graph[cycle[i]]:
            expand_graph[cycle[i]][cycle[i-1]] = graph[cycle[i]][cycle[i-1]]
        i = i - 1
    if cycle[-1] != vstar and cycle[0] in expand_graph and cycle[-1] in expand_graph[cycle[0]]:
        expand_graph[cycle[0]][cycle[-1]] = graph[cycle[0]][cycle[-1]]
    return expand_graph

    #
    # cycle1 = list(cycle)
    # cycle1.append(cycle[0])
    # print(cycle1)
    # i = 0
    # while i < len(cycle1) :
    #     print(expand_graph)
    #     if cycle1[i-1] != vstar:
    #         expand_graph[cycle1[i]][cycle1[i-1]] = graph[cycle1[i]][cycle1[i-1]]
    #     i += 1
    # return expand_graph

# test_graph = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# rgraph = reverse_digraph_representation(test_graph)
# modify_edge_weights(rgraph, 0)
# rdst = compute_rdst_candidate(rgraph, 0)
# cycle = compute_cycle(rdst)
# rdst_cc, cstar = contract_cycle(reverse_digraph_representation(rgraph), cycle)
# expand_graph(test_graph, rdst_cc, cycle, cstar)
# g = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
# 4: {1: 4}, 5: {}}
#print(expand_graph({0: {1: 0}, 1: {2: 0}, 2: {2: 0, 1: 0}}, {0: {3: 0}, 3: {}}, [1, 2], 3))
# cycle = compute_cycle(g)
# contracted_graph,cstar = contract_cycle(g, cycle)
#print(expand_graph(g, contracted_graph, cycle, cstar))

# g0 = {0: {1: 2, 2: 2, 3: 2}, 1: {2: 2, 5: 2}, 2: {3: 2, 4: 2}, 3: {4: 2, 5: 2}, 4: {1: 2}, 5: {}}
# print(compute_rdmst(g0, 0))
# print("expected: ({0: {1: 2, 2: 2, 3: 2}, 1: {5: 2}, 2: {4: 2}, 3: {}, 4: {}, 5: {}}, 10)")
# g1 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# print(compute_rdmst(g1, 0))
# print("expected: ({0: {2: 4}, 1: {}, 2: {3: 8}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}, 28)")
