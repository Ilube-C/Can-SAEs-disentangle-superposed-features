from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for superposition experiments."""
    
    # Data generation parameters
    sparsity_levels: List[float] = None
    sparse_dim: int = 20  # n: sparse dimension
    dense_dim: int = 5    # m: dense dimension (bottleneck)
    num_samples: int = 10000  # N: number of datapoints
    
    # Feature importance parameters
    decay_factor: float = 0.7
    
    # Training parameters
    num_epochs: int = 10
    learning_rate: float = 1e-2
    
    # Random seed
    random_seed: int = 42
    
    # Output paths
    results_dir: str = "results"
    save_plots: bool = True
    
    def __post_init__(self):
        if self.sparsity_levels is None:
            self.sparsity_levels = [0.1, 0.3, 0.7]


# Default configuration
DEFAULT_CONFIG = ExperimentConfig()