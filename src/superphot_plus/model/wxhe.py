"""The class holding the hierarchical cross entropy loss."""

import torch
import torch.nn as nn
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from superphot_plus.model.taxonomy import Taxonomy


class hier_xe_loss(nn.Module):
    
    def __init__(self, taxo: Taxonomy, unique_labels):
        super().__init__()
        self.all_paths, self.pathlengths, self.mask_list, self.y_dict = taxo.calc_paths_and_masks()
        self.alpha = taxo.alpha
        self.weight_dict = taxo.weight_dict
        self.mapping = taxo.mapping
        self.config = taxo.config
        self.unique_labels = unique_labels

    def __str__(self):
        string = f"masklist: {self.mask_list}\npathlengths: {self.pathlengths}"
        string += f"\nall_paths: {self.all_paths}\nalpha: {self.alpha}\nclass_weights: {self.weight_dict}"
        return string 
    
    """
    Copied from hxe-for-tda by Ashley Villar.
    """
    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-10):
        # Applies the mask
        # Note, this is multiplying torch.tensors, so it just multiplies the elements, not dot product
        masked_vec = vec * mask.float()

        # exp(masked_vec) -> exp(0) for masked-out, exp(vec[i]) for valid elements
        # each exps_i = e^[masked_vec_{i}]
        exps = torch.exp(masked_vec)
        
        # Need to apply masking again because we need to zero out masked entries again as exp(0)=1
        masked_exps = exps * mask.float()

        # Sums the unmasked exponentials across the class dimension and adds epsilon 
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon 

        # Computes softmax only where mask = 1, preserves original values where mask = 0
        final_vec = masked_exps/masked_sums + ((1-mask) * vec)
        return final_vec

    def get_label(self, y):
        # account for tree contraction in mapping of leaves
        new_labels = [None] * y.size(dim=0)
        j = 0
        for ind in y:
            # ind is the index in config.allowed_types of the correct label for this data point
            label = self.config.graph['vertices'][ind]
            new_labels[j] = self.y_dict[label]
            j += 1
        return torch.from_numpy(np.array(new_labels))

    def get_weight(self, y):
        # account for tree contraction in mapping of leaves
        new_weights = [None] * y.size(dim=0)
        j = 0
        for ind in y:
            label = self.config.graph['vertices'][ind]
            new_weights[j] = torch.from_numpy(np.array([self.weight_dict[label][0]]))
            j+= 1
        return torch.from_numpy(np.array(new_weights))

    """
    Modified from hxe-for-tda by Ashley Villar.
    """
    def forward(self, y_pred, y_actual, alpha=0.5):
        final_sum = 0
        
        # sets the first column/entry of y_pred to 1 such that the root node 
	    # always has a fixed probability of 1, or log-prod = 0
        y_pred[:, 0] = 1.0 
        #print("y_pred", y_pred)

        labels = self.get_label(y_actual)
        weights = self.get_weight(y_actual)
        print("labels: ", labels)
        print(labels.size())
        print("weights: ", weights)
        print(weights.size())
        
        # Applies the different parent masks made above to the y_pred to normalize 
	    # child probabilities for each set of children/leaves
        for i,mask in enumerate(self.mask_list):
            y_pred = self.masked_softmax(y_pred, mask)

        # Apply log to the conditional probabilities
        y_pred = y_pred.log()

        # Apply the lambda = exp(-ah(c)) term
        y_pred = y_pred * np.exp(-alpha * (self.pathlengths - 1))
        print("y_pred: ", y_pred)
        print(y_pred.size())


        # y_pred*y_actual.sum -> picks out log probs along hierarchical path + 
        # 					   sums along this path giving us log joint prob of path up tree
        # target_weights * (...) -> This is the multiply by W(c)
        # .mean() -> normalize by the batch size
        inter = (y_pred*labels)
        print("type, inter: ", type(inter))
        print(inter.size())

        final_sum = ((weights * inter).sum(dim=1)).mean()

        return -final_sum