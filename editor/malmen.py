from typing import Dict
from omegaconf import DictConfig

import torch.nn as nn
from editor.base import BaseEditor

from util import (
    get_module,
    get_shape,
)
import torch
from nets import HyperNet

class MALMEN(BaseEditor):

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        super().__init__(
            config,
            model
        )
        self.nets = {}
        net = nn.ModuleDict({
            str(k): HyperNet(
                *k,
                config.editor.rank ,
                config.editor.n_blocks,
                v,
                config.editor.lr
            )
            for k, v in self.shape_counter.items()
        }).to(config.editor_device)
        for i in range(config.editor.iters):
            self.nets[i] = net
        params = []
        for _, net in self.nets.items():
            params.extend(net.parameters())
        self.opt = torch.optim.Adam(
            params,
            config.editor.meta_lr
        )