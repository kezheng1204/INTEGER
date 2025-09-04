import numpy as np
import open3d
import random
import torch
import pickle
import yaml

from typing import Tuple, List, Optional, Union, Any

from copy import deepcopy


class TTAModelParameterProtector(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def __enter__(self):
        self.model_param_back = deepcopy(self.model.state_dict())
        self.optimizer_param_back = deepcopy(self.optimizer.state_dict())

    def __exit__(self, type, value, trace):
        self.model.load_state_dict(self.model_param_back)
        self.optimizer.load_state_dict(self.optimizer_param_back)
        del self.model_param_back
        del self.optimizer_param_back



