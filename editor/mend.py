from typing import Dict
from omegaconf import DictConfig

import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from editor.base import BaseEditor
from util import get_module, get_shape
from nets import HyperNet


from util import (
    get_module,
    get_shape,
)


class MEND(BaseEditor):

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


    def predict_param_shifts(self, net_id) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.model.edit_modules):
            
            shape = get_shape(get_module(self.model, module_name))
            net = self.nets[net_id][str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            param_shift = torch.zeros((net.key_size, net.value_size), device=self.config.editor_device)
            for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size)):
                keys = torch.load(f"{self.config.editor.cache_dir}/{self.config.dataset.name}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}/{module_idx}_{idx}_keys_{net_id}.pth")
                values_grad = torch.load(f"{self.config.editor.cache_dir}/{self.config.dataset.name}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}/{module_idx}_{idx}_values_grad_{net_id}.pth")
                with torch.no_grad():
                    pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
                    param_shift += - net.lr(layer_idx) * pesudo_keys.T @ pesudo_values_grad
            param_shifts[module_name] = param_shift

        return param_shifts
    

    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor], update: bool, net_id: int):
        self.opt.zero_grad()
        for module_idx, module_name in enumerate(self.config.model.edit_modules,):
            shape = get_shape(get_module(self.model, module_name))
            net = self.nets[net_id][str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size)):
                keys = torch.load(f"{self.config.editor.cache_dir}/{self.config.dataset.name}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}/{module_idx}_{idx}_keys_{net_id}.pth")
                values_grad = torch.load(f"{self.config.editor.cache_dir}/{self.config.dataset.name}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}/{module_idx}_{idx}_values_grad_{net_id}.pth")
                pesudo_keys, pesudo_values_grad = net(keys, values_grad, layer_idx)
                param_shift = - net.lr(layer_idx) * pesudo_keys.T @ pesudo_values_grad
                (module_grad * param_shift.to(module_grad.device)).sum().backward()

        clip_grad_norm_(
            self.nets[net_id].parameters(),
            self.config.editor.max_grad_norm
        )
        
        if update == True:
            self.opt.step()
            self.opt.zero_grad()
