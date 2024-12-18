from dataclasses import dataclass
import numpy as np
from enum import Enum

@dataclass
class ExperimentConfig:
    data_size: float = 1.0
    width: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    batch_size: int = 128
    depth: int = 4
    epochs: int = 100

class Distribution(Enum):
    LOG = 1     # Used to generate log distribution as by np.logspace
    DOUBLE = 2  # Used to generate doubling distribution as by generate_doubling_space

def generate_doubling_space(start, end):
    """Keeps doubling the 'start' value until it reaches the 'end'."""
    values = [start]
    while values[-1] * 2 <= end:
        values.append(values[-1] * 2)
    return np.array(values)

class GridGenerator:
    def __init__(self, grid):
        """
        Initializes the Grid object with a given configuration.
        Args:
            grid (dict): A dictionary where the key corresponds to an attribute in 
                            ExperimentConfig and the value is either:
                            - A tuple (min, max, step, Optional[Distribution]) defining a range of values.
                            - A function that takes an ExperimentConfig object as an argument and 
                            computes the value based on this configuration.

                        Example: {
                                    'data_size': (0.1, 1.0, 0.1),                # Linear distribution. 0.1, 0.2, 0.3, ...
                                    'width': (32, 128, 32, Distribution.DOUBLE), # Doubling distribution. 32, 64, 128, ...
                                    'learning_rate': predict_learning_rate       # Function to compute learning rate
                                }
        """
        self.value_grid = {}
        self.function_grid = {}

        for k, v in grid.items():
            if callable(v):
               self.function_grid[k] = v
            else:
                self.value_grid[k] = self._generate_distribution(v)

        self.grid_shape = [len(v) for v in self.value_grid.values()]

    def _generate_distribution(self, value):
        assert 2 < len(value) < 5, "Values should have 3 or 4 values."
        
        if len(value) == 3:
            return np.arange(*value)
        
        start, end, step, distribution = value

        if distribution == Distribution.DOUBLE:
            return generate_doubling_space(start, end)

        if distribution == Distribution.LOG:
            num = int((end-start)/step) + 1
            return np.logspace(np.log10(start), np.log10(end), num=num)
        
        raise ValueError(f"Not supported distribution {distribution}")

    def __repr__(self) -> str:
        grid_info = ", ".join(f"{k}: {len(v)}" for k, v in self.value_grid.items())
        return f"GridGenerator({grid_info})"

    def __len__(self) -> int:
        """Total number of experiments in the grid."""
        return np.prod(self.grid_shape)

    def __getitem__(self, index) -> ExperimentConfig:

        # Convert the 1D index to N-dimensional index
        indices = np.unravel_index(index, self.grid_shape)

        d = {key: values[index] for key, values, index in zip(self.value_grid.keys(), self.value_grid.values(), indices)}
        experiment = ExperimentConfig(**d)

        for key, function in self.function_grid.items():
            experiment.__setattr__(key, function(experiment))

        return experiment

    def __iter__(self):
      for i in range(len(self)):
        yield self[i]
