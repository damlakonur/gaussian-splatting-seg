from typing import Dict, Union

import numpy as np
import torch


class AverageMeter:
    """Computes and stores the average of metrics."""
    
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.data = {}

    def update(self, loss_dict: Dict[str, Union[float, torch.Tensor]]) -> None:
        for key, val in loss_dict.items():
            if val is None:
                continue
            if key not in self.data:
                self.data[key] = []
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.data[key].append(val)

    def finalize(self) -> Dict[str, float]:
        loss_dict = {}
        for key, val in self.data.items():
            loss_dict[key] = np.mean(val)
        return loss_dict

