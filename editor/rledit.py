from typing import Dict
from omegaconf import DictConfig

import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from nets import HyperNet

from editor.base import BaseEditor
from util import get_module, get_shape

import numpy as np


from itertools import islice

from tqdm import tqdm
import swanlab

from util import (
    get_module,
    get_shape,
    empty_cache,
    cross_entropy,
    kl_div,
    pad_tensor
)





class RLEDIT(BaseEditor):

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
        for i in range(config.editor.iters):
            self.nets[i] = nn.ModuleDict({
                str(k): HyperNet(
                    *k,
                    config.editor.rank // (i+1),
                    config.editor.n_blocks,
                    v,
                    config.editor.lr
                )
                for k, v in self.shape_counter.items()
            }).to(config.editor_device)
        params = []
        for _, net in self.nets.items():
            params.extend(net.parameters())
        self.opt = torch.optim.Adam(
            params,
            config.editor.meta_lr
        )

    def train(self, loader: DataLoader, save=False):
        """
        The training method for RLEdit.
        Model the sequential editing as a Markov Devision Process, and use the Paradigm of Reinforce Learning to solve the question.
        """

        sequence_tuples = []
        max_steps = self.config.num_seq
        time_decay = self.config.editor.time_decay

        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Train", ncols=100, total=max_steps)):
            gen_losses_show = []
            loc_losses_show = []

            sequence_tuples.append(tuples)


            param_shifts_dict = {}
            for net_id in range(self.config.editor.iters):
                tot_loss_e = 0
                tot_loss_loc = 0
                self.cache(tuples["edit_tuples"], net_id)
                param_shifts_dict[net_id] = self.predict_param_shifts(net_id)
                self.model.zero_grad()
                l2_reg_loss = 0
                for _, param_shift in param_shifts_dict[net_id].items():
                    l2_reg_loss += torch.sum(param_shift ** 2)
                l2_reg_loss *= self.config.editor.reg_coef

                
                self.edit_model(param_shifts_dict[net_id], False)


                for _, tuple in enumerate(reversed(sequence_tuples)):
                    loss_e = 0
                    for t in tuple["equiv_tuples"]:
                        if "old_labels" in t:
                            old_labels = t.pop("old_labels")
                        logits = self.model(**t)["logits"]
                        try:
                            t["old_labels"] = old_labels
                        except:
                            pass
                        loss = cross_entropy(logits, t["labels"])
                        loss_e += loss
                    gen_losses_show.append(loss_e.item())
                    tot_loss_e += (loss_e * pow(time_decay, _))

                    if _+1 >= self.config.editor.back_depth:
                        break

                tot_loss_e += l2_reg_loss.to(tot_loss_e.device)
                tot_loss_e.backward()
                self.edit_model(param_shifts_dict[net_id], True)

                for _, tuple in enumerate(reversed(sequence_tuples)):
                    loss_loc = 0
                    for t in tuple["unrel_tuples"]:
                        if "old_labels" in t:
                            old_labels = t.pop("old_labels")
                        with torch.no_grad():
                            refer_logits = self.model(**t)["logits"]
                        self.edit_model(param_shifts_dict[net_id], False)
                        logits = self.model(**t)["logits"]
                        try:
                            t["old_labels"] = old_labels
                        except:
                            pass
                        loss = kl_div(
                            refer_logits,
                            logits,
                            t["labels"]
                        )
                        loss_loc += (self.config.editor.loc_coef * loss)
                        self.edit_model(param_shifts_dict[net_id], True)
                    loc_losses_show += [loss_loc.item()]
                    tot_loss_loc += (loss_loc * pow(time_decay, _))

                    if _+1 >= self.config.editor.back_depth:
                        break
                tot_loss_loc.backward()
                self.edit_model(param_shifts_dict[net_id], False)

            self.update_hypernet(param_shifts_dict, False)

            swanlab.log({
                "gen_loss": np.mean(gen_losses_show),
                "loc_loss": np.mean(loc_losses_show)
            })

        self.opt.step()
        self.opt.zero_grad()
        del param_shifts_dict, tot_loss_e, l2_reg_loss, loss, loss_e, gen_losses_show, tot_loss_loc, loc_losses_show
        torch.cuda.empty_cache()
        if save:
            torch.save(self.net, f"checkpoints/hypernet.pt")
        
    def update_hypernet(self, param_shifts_dict: Dict, update: bool, net_id: int = None):
        for net_id in range(self.config.editor.iters):
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                net = self.nets[net_id][str(shape)]
                layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.editor_device)
                keys = torch.cat([
                    torch.load(f"{self.config.editor.cache_dir}/{self.config.dataset.name}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}/{module_idx}_{idx}_keys_{net_id}.pth")
                    for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size))
                ])
                values_grad = torch.cat([
                    torch.load(f"{self.config.editor.cache_dir}/{self.config.dataset.name}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}/{module_idx}_{idx}_values_grad_{net_id}.pth")
                    for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size))
                ])
                module = get_module(self.model, module_name)
                module_grad = module.weight.grad.to(torch.float32).to(self.config.editor_device)
                param_shift = param_shifts_dict[net_id][module_name].to(self.config.editor_device)
                if isinstance(module, nn.Linear):
                    module_grad = module_grad.T
                with torch.no_grad():
                    mat = torch.linalg.solve(keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device = self.config.editor_device), module_grad)
                    lamda_grad = - net.lamda(layer_idx).exp() * (mat * param_shift).sum()
                value_diffs_grad = keys @ mat
                (lamda_grad * net.lamda(layer_idx)).backward()
                for start_idx in range(0, keys.shape[0], self.config.editor.batch_size):
                    end_idx = start_idx + self.config.editor.batch_size
                    keys_once = keys[start_idx:end_idx]
                    values_grad_once = values_grad[start_idx:end_idx]
                    (pesudo_keys, pesudo_values_grad) = net(
                        keys_once,
                        values_grad_once,
                        layer_idx,
                    )
                    coeffs = - net.lr(layer_idx) * (keys_once * pesudo_keys).sum(-1).unsqueeze(-1)
                    value_diff = coeffs * pesudo_values_grad
                    value_diff = value_diff[:keys.shape[0] - start_idx, :]
                    (value_diffs_grad[start_idx:end_idx] * value_diff).sum().backward(retain_graph=True)
            params = []
            for _, net in self.nets.items():
                params.extend(net.parameters())
            clip_grad_norm_(
                params,
                self.config.editor.max_grad_norm
            )

        if update == True:
            self.opt.step()
            self.opt.zero_grad()

