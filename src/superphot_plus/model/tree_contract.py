""" File containing support functions for taxonomy.py file.
"""

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def bfs(G, s):
    """Breadth First Search Algorithm 
    Defines a set of frontiers for each level of the hierarchy to define
    contraction along the graph.
    
    Parameters
    ----------
    G : nx.DiGraph
        The graph that we want to search over.
    s : str
        The source/root node of the graph.

    Returns
    -------
    X : set
        The set of visited vertices during BFS.
    i : int
        The count of the number of levels covered.
    frontiers : list set
        The set of vertices included within each frontier found by BFS.
    """
    def explore(X, F, i, frontiers):
        if len(F) == 0: return (X, i-1, frontiers)
        frontiers[i] = F
        X = X.union(F)
        next_F = set()
        for u in F:
            for v in G.neighbors(u):
                if v not in X:
                    next_F.add(v)
        return explore(X, next_F, i+1, frontiers)
    
    return explore(set(), {s}, 1, {}) 


def tree_contract(G, graph, class_weights):
    """Contraction Tree Algorithm 
    Supports contraction of tree and reweighting of leaves (assumes that all
    leaves are the classes one wants to classify over).  
    
    Parameters
    ----------
    G : nx.DiGraph
        The graph that we want to contract.
    graph : dict
        Dictionary built from config.yaml including root, edges, vertices, and
        desired height information of the graph.
    class_weights : dict
        The original class weights dictionary from config.yaml that we edit here
        to represent the contracted tree.

    Returns
    ----------
    mapping : dict
        Maps each original vertex to the direct parent it was contracted to.
    class_new : dict
        The updated dictionary of class weights. Assums that leaves are the classes
        we are classifying over.
    graph : dict
        Updates the vertices and edges list for the graph.
    G_new : nx.DiGraph
        The updated graph using the edited graph variable. 
    """
    # Builds a list of shortest paths from root to node from our trees
    # and groups into a dictionary such that paths of the same length
    # are kept together.
    paths = {}
    for node in graph['vertices']:
        path = nx.shortest_path(G, graph['root'], node)
        if (len(path)-1) not in paths.keys():
            paths[len(path)-1] = [path]
        else: paths[len(path)-1] += [path]

    # Iterate through the keys and for all that are for paths longer than 
    # config.height, we need to condense the weights to the appropriate
    # leveled parent
    mapping = {}
    counts = {}
    contractions = 0
    removals = []
    if graph['height'] > max(paths.keys())+1:
        # make sure the height is valid
        raise ValueError("Height cannot be bigger than the max height of the graph.")
    elif graph['height'] == max(paths.keys())+1:
        print('Using the full tree, no change.')
        return mapping, class_weights, graph, G
    else:
        # Update the weights & graph properties
        n_labels = len(class_weights) # this is the number of leaves/classes overall before we shrink
        n_all = sum([class_weights[i][1] for i in class_weights.keys()]) # get the sum of all numbers of instances
        _,_,fronteirs = bfs(G, graph['root'])

        for level in sorted(fronteirs.items(), reverse=True):
            if level[0] > graph['height']:
                for node in fronteirs[level[0]]:

                    # (1) find the path in paths that ends in node
                    # Path must be in paths at length of path equal to level-1
                    n_path = [p for p in paths[level[0] - 1] if p[-1] == node]
                    
                    # (2) get the direct parent from the one right before
                    # If the only node is the root node (unlikely) then we need
                    # to classify it as 'Null'
                    if len(paths[level[0] - 1]) == 1:
                        dir_par = 'Null'
                    else:
                        dir_par = n_path[0][-2]
                    
                    # (3) Update mapping
                    mapping[node] = dir_par

                    # (4) Update counts class (account for diff groupings of things)
                    if dir_par not in counts.keys() and node in class_weights.keys() and node not in graph['ignored_leaves']:
                        # If this is a leaf node whose parent has not been seen before
                        counts[dir_par] = [0, class_weights[node][1]]
                        contractions += 1
                        class_weights.pop(node, None)
                    elif dir_par not in counts.keys() and node not in class_weights.keys() and node not in graph['ignored_leaves']:
                        # If this is not a leaf node whose parent has not been seen before
                        counts[dir_par] = [0, counts[node][1]]
                        counts.pop(node, None)
                    elif dir_par in counts.keys() and node in class_weights.keys() and node not in graph['ignored_leaves']:
                        # If this is a leaf node whose parent has been seen before
                        counts[dir_par] = [0, counts[dir_par][1]+class_weights[node][1]]
                        contractions += 1
                        class_weights.pop(node, None)
                    elif dir_par in counts.keys() and node in counts.keys() and node not in graph['ignored_leaves']:
                        # If this is a non-leaf node whose parent has been seen before
                        counts[dir_par] = [0, counts[dir_par][1]+counts[node][1]]
                        counts.pop(node, None)
                    else:
                        # Ignore everything else
                        removals += [node]
                        continue

                    # (5) Keep track of removals
                    removals += [node]
        
        # Update the class_weights dictionary
        class_new = dict(list(class_weights.items()) + list(counts.items()))
        n_labels_new = n_labels - contractions + len(counts)
        for label in class_new:
            class_new[label] = [(n_all/(n_labels_new * class_new[label][1])), class_new[label][1]]

        # Update the graph
        for node in removals:
            graph['vertices'].remove(node)
            for edge in graph['edges']:
                if edge[1] == node:
                    graph['edges'].remove(edge)

        G_new = nx.DiGraph()
        for edge_num in range(len(graph['edges'])):
            G_new.add_edge(graph['edges'][edge_num][0], graph['edges'][edge_num][1])


    return mapping, class_new, graph, G_new

"""
Hierarchy Position Encoder.
Taken from stackoverflow: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
(Used in hxe-for-tda)
"""
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
    G: the graph
    root: the root node
    levels: a dictionary
            key: level number (starting from 0)
            value: number of nodes in this level
    width: horizontal space allocated for drawing
    height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})