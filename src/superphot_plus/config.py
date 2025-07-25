import dataclasses
import os
from dataclasses import dataclass, field
from typing import List, Optional, ClassVar
from functools import *

import numpy as np
import torch
import yaml
from typing_extensions import Self

from .supernova_class import SupernovaClass as SnClass 
from .hierarchy_class import HierarchyClass as HClass

# pylint: disable=too-many-instance-attributes
@dataclass
class SuperphotConfig:
    """Holds information about the specific training
    configuration of a model. The default values are
    sampled by ray tune for parameter optimization."""

    create_dirs: bool = True
    relative_dirs: bool = True
    
    # data options
    data_dir: str = 'data'
    transient_data_fn: str = "transients"
    
    # sampling options
    sampler_results_fn: str = "sampler_results"
    sampler: str = "dynesty"
    chisq_cutoff: float = 1.2
    
    # plotting options
    figs_dir: str = 'figs'
    metrics_dir: str = 'metrics'
    fit_plots_dir: str = 'fits'
    cm_dir: str = 'confusion_matrices'
    
    # logging options
    logging: bool = False
    log_fn: str = 'results.log'
    plot: bool = False
    
    # classification options
    load_checkpoint: bool = False
    models_dir: str = 'models'
    model_type: str = "LightGBM"
    probs_dir: str = 'probabilities'
    device = torch.device("cpu")
    input_features: Optional[list] = None
    use_redshift_features: bool = False
    fits_per_majority: int = 5
    
    # single-class options
    target_label: Optional[str] = None
    prob_threshhold: Optional[float] = 0.5
    
    # multi-class options
    allowed_types: Optional[list[str]] = None
    
    # MLP parameters
    neurons_per_layer: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    
    # general training
    n_folds: int = 1
    num_epochs: Optional[int] = None
    n_parallel: int = 1
    
    # reproducibility options
    random_seed: int = 42

    # graph/taxonomy parameters for WHXE
    use_hierarchy: bool = False
    alpha: float = None
    class_weights: Optional[dict] = field(default_factory = lambda: {})
    graph: Optional[dict] = field(default_factory = {
        'edges' : None,
        'height' : None, 
        'ignored_leaves': None,
        'root' : None,
        'vertices' : None
    }) 
    
    
    def __post_init__(self):
        """Ensure subdirectory structure exists."""
        if self.relative_dirs:
            self.transient_data_fn = os.path.join(self.data_dir, self.transient_data_fn)
            self.models_dir = os.path.join(self.data_dir, self.models_dir)
            self.sampler_results_fn = os.path.join(self.data_dir, self.sampler_results_fn)
            self.figs_dir = os.path.join(self.data_dir, self.figs_dir)
            
            self.metrics_dir = os.path.join(self.figs_dir, self.metrics_dir)
            self.fit_plots_dir = os.path.join(self.figs_dir, self.fit_plots_dir)
            self.cm_dir = os.path.join(self.figs_dir, self.cm_dir)

            self.log_fn = os.path.join(self.data_dir, self.log_fn)
            self.probs_dir = os.path.join(self.data_dir, self.probs_dir)
            
        self.model_prefix = os.path.join(self.models_dir, f"model_{self.__str__()}")
        self.metrics_prefix = os.path.join(self.metrics_dir, f"metrics_{self.__str__()}")
        self.cm_prefix = os.path.join(self.cm_dir, f"cm_{self.__str__()}")
        self.probs_fn = os.path.join(self.probs_dir, f"probs_{self.__str__()}.csv")
    
        if self.create_dirs:
            for x_dir in [
                self.models_dir, self.figs_dir, self.metrics_dir,
                self.fit_plots_dir, self.cm_dir, self.probs_dir
            ]:
                os.makedirs(x_dir, exist_ok=True)

        if self.n_folds < 1:
            raise ValueError("Number of K-folds must be positive")
        if self.chisq_cutoff <= 0.0:
            raise ValueError("chisq cutoff must be positive")
        
        if not self.use_hierarchy:
            if self.allowed_types is None:
                self.allowed_types = SnClass.all_classes()
            self.allowed_types = [SnClass.canonicalize(x) for x in self.allowed_types]
        else:
            self.allowed_types = [key for key, val in self.class_weights.items()]
            print(self.allowed_types)
            canon_part = partial(HClass.canonicalize, self.allowed_types, None)
            self.allowed_types = [canon_part(label) for label in self.allowed_types]

            # Use Checks for correctness
            if len(self.class_weights) == 0:
                raise ValueError("Class_weights must be manually filled in with weights for the labels.")
            if len(self.graph['vertices']) == 0:
                raise ValueError("Length of vertices cannot be 0.")
            if len(self.graph['edges']) == 0:
                raise ValueError("Length of edges cannot be 0.")
            if self.graph['root'] not in self.graph['vertices']:
                raise ValueError("Root must be a valid vertex.")
            if self.graph['height'] <= 1:
                raise ValueError("Height must be at least 2.")
            for leaf in self.graph['ignored_leaves']:
                if leaf in self.class_weights.keys():
                    raise ValueError("There are counts for this leaf, it should not be ignored.")
                if leaf not in self.graph['vertices']:
                    raise ValueError("These must be valid leaf selections.")
            for vert in self.class_weights.keys():
                if vert not in self.graph['vertices']:
                    print(vert)
                    raise ValueError("All class weight labels must be vertices in the graph.")
            for edge in self.graph['edges']:
                if edge[0] not in self.graph['vertices'] or edge[1] not in self.graph['vertices']:
                    print("(",edge[0], ",", edge[1], ")")
                    raise ValueError("All edges must be pairs of valid vertices in the graph.")
            

    def __str__(self):
        """Return string summary of config for unambiguous file naming.
        Note: does not include filenames, so if contents of files change,
        config str is not unique."""
        string = f"{self.sampler}_{self.model_type}_{str(self.input_features)}_{self.use_redshift_features}"
        string += f"_{self.fits_per_majority}_{self.target_label}_{self.n_folds}_{self.num_epochs}_{self.random_seed}"
        
        if self.model_type == 'MLP':
            string += f"_{self.neurons_per_layer}_{self.num_hidden_layers}_{self.learning_rate}_{self.batch_size}"
            
        return string

    def write_to_file(self, file: str):
        """Save configuration data to a YAML file."""
        args = dataclasses.asdict(self)
        if self.relative_dirs: # undo file add-ons
            args['transient_data_fn'] = self.transient_data_fn.removeprefix(self.data_dir+"/")
            args['models_dir'] = self.models_dir.removeprefix(self.data_dir+"/")
            args['sampler_results_fn'] = self.sampler_results_fn.removeprefix(self.data_dir+"/")
            args['figs_dir'] = self.figs_dir.removeprefix(self.data_dir+"/")
            args['metrics_dir'] = self.metrics_dir.removeprefix(self.figs_dir+"/")
            args['fit_plots_dir'] = self.fit_plots_dir.removeprefix(self.figs_dir+"/")
            args['cm_dir'] = self.cm_dir.removeprefix(self.figs_dir+"/")
            args['log_fn'] = self.log_fn.removeprefix(self.data_dir+"/")
            args['probs_dir'] = self.probs_dir.removeprefix(self.data_dir+"/")

        encoded_string = yaml.dump(args, sort_keys=False, default_flow_style=False)
        with open(file, "w", encoding="utf-8") as file_handle:
            file_handle.write(encoded_string)
            
    def update(self, **kwargs):
        """Update config attributes."""
        for (k,v) in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_file(cls, file: str) -> Self:
        """Load configuration data from a YAML file."""
        with open(file, "r", encoding="utf-8") as file_handle:
            metadata = yaml.safe_load(file_handle)
            return cls(**metadata)
