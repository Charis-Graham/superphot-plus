""" This module implements the tree taxonomy used in hierarchical 
    loss function. 
    It takes inspiration from https://github.com/VTDA-Group/hxe-for-tda/blob/main/hxetda/hxetda.py
"""

import torch
import networkx as nx
import numpy as np

from ..config import SuperphotConfig 
from .tree_contract import *

class Taxonomy:
    """Taxonomic tree class. 
    Supports all data extractions from the tree and data needed
    to train an MLP.  
    
    Parameters
    ----------
    config : SuperphotConfig
        The configuration for the tree for the hierarchy.
    """

    def __init__(self, config: SuperphotConfig):
        super().__init__()

        vertices = config.graph['vertices'] # type list of vertices
        edges = config.graph['edges'] # type [vertex, vertex] list, directed with a->b as [a, b]
        root = config.graph['root'] # type string

        G = nx.DiGraph()
        for edge_num in range(len(edges)):
            G.add_edge(edges[edge_num][0], edges[edge_num][1])
        
        self.mapping, self.weight_dict, self.graph, self.G = tree_contract(G, config.graph, config.class_weights)
        self.edges = self.graph['edges']
        self.vertices = self.graph['vertices']
        self.root = self.graph['root']
        self.ignored_leaves = self.graph['ignored_leaves']

        self.all_paths = None
        self.path_lengths = None
        self.mask_list = None

    def __str__(self):
        """Returns string summary of taxonomy for debugging purposes and 
        sanity checks."""
        string = f"mapping: {self.mapping}\nweight_dict: {self.weight_dict}\ngraph: {self.graph}\n"
        string += f"edges: {self.edges}\nvertices: {self.vertices}\nroot: {self.root}"
        return string

    def calc_paths_and_masks(self):
        """Generates the path information needed for use in the WHXE loss.

        Parameters
        ----------
        
        Returns
        -------
        all_paths : str list list
            List of all the shortest paths between root and and a given node
            in the tree. This is found using Dijkstra's.

        path_lengths : torch.Tensor
            List of the length of each of the shortest paths from root to a node,
            corresponding to the order of elements of all_paths.
        
        mask_list : torch.Tensor list
            A list of masks for each unique partent in G such that, for unique
            parent u, mask_list[i] = 1 if i = index of u's children in vertices.
        """
        
        path_maps = np.zeros((len(self.vertices), len(self.vertices)))
        path_lengths = []
        all_paths = []
        parent_inds = []
        mask_list = []
        torch_v = np.asarray(self.vertices, dtype='str')

        # This loop: 
	    # (1) gets the shortest path between the root and a given node
	    # (2) updates pathlengths and all_paths 
	    # (3) makes path_maps[vert][i]=1 if vertices[i] is on the shortest path from root to vert
	    # (4) generates a list where parent_inds[i] = ind in vertices of G[i]'s direct parent
        for i, node in enumerate(self.vertices):
            short_path = nx.shortest_path(self.G, self.root, node)

            path_lengths.append(len(short_path))
            all_paths.append(short_path)

            for vert in short_path:
                g_ind = np.where(vert == torch_v)[0]
                path_maps[i, g_ind[0]] = 1
            
            if node == self.root:
                parent_inds.append(-1)
            else:
                parent_inds.append(np.where(torch_v==next(self.G.predecessors(node)))[0][0])

        # Makes parent inds into a set of masks to be softmaxed over later. I.e. we are
	    # creating a mask for every unique parent in G such that we put 1 in the index of
	    # its children as listed in the order of vertices.
        for par_ind in np.unique(parent_inds):
            # Skip the root node
            if par_ind == -1:
                continue

            # Finds the indicies of the children that have the current parent 
            g_ind = np.where(parent_inds == par_ind)

            # Make a mask the size of vertices, where mask[i] = 1 if child of vertices[par_ind] 
            
            # Note, I think this might be dependent on the ordering of the vertices list 
            # as this does not carry down information about *which* parent is for each mask...
            mask = np.zeros(len(path_maps))
            mask[g_ind] = 1
            mask_list.append(torch.tensor(mask, dtype=int))
    
        path_lengths = torch.tensor(path_lengths)

        self.all_paths = all_paths
        self.path_lengths = path_lengths
        self.mask_list = mask_list
            
        return all_paths, path_lengths, mask_list
    
    
    def initialize_training(self, labels, class_weight_dict):
        """Initiates training elements.

        Parameters
        ----------
        labels : str np.ndarray
            Ground truth label list for datapoints in the dataset.
        
        class_weight_dict : str -> 1d numpy.ndarray dict
            Contains weights of each label in the dataset (due to
            unbalanced data).
        
        y_dict : (str -> numpy.ndarray) dict
            Each vertex v maps to an array where a[i] = 1 if vertices[i] is on
            the shortest path from root to v, inclusive of v.
        
        Returns
        -------
        weights : np.ndarray list
            Contains initialized weights for the data to train on.

        labels_new : np.ndarray list
            For each index with label i, contains associated path
            vector of 0s and 1s, where v[j]=1 if vertices[j] is on
            path from root to label i in tree.
        """
        #labels_new = [y_dict[x] for x in labels]
        weights = [class_weight_dict[x] for x in labels]
        return weights #labels_new,


    # Add drawing function down here for ease of use.
    def draw_graph(self):
        # Encode positions of graph - purely for drawing purposes
        pos = hierarchy_pos(self.G, self.root)

        fig = plt.figure(1, figsize=(20,10))
        nx.draw_networkx(self.G, pos=pos, nodelist=self.vertices, node_color='white', with_labels=False, node_size=2000, arrows=False)
        text = nx.draw_networkx_labels(self.G, pos)

        #Uncomment if need to rotate the labels for a more advanced label system.
        for _, t in text.items():
            t.set_rotation(25) 
        plt.show()
