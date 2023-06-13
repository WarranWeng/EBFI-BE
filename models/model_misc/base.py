"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

from abc import abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        # model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        trained_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        return super().__str__() + "\nTrainable parameters: {} \nAll parameters: {}".format(trained_params, all_params)
