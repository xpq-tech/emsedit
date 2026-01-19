from omegaconf import DictConfig
from typing import Dict
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import math
from editor.base import BaseEditor

import numpy as np

from itertools import islice

from tqdm import tqdm
import swanlab
from util import get_module, get_shape
from util import (
    cross_entropy,
    empty_cache
)
from nets import HyperNet



class EMSEDIT(BaseEditor):

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
        if self.config.num_seq == 1: ## num_seq = 1 means batch editing, using identical hypernet
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
        else:
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
        self.original_params = {
            module_name: get_module(self.model, module_name).weight.detach().clone()
            for module_name in self.config.model.edit_modules
        }
    def run(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use MEND or MALMEN to complete sequential editing task.
        Just train the hypernet on the original LLM, then freeze it.
        """
        empty_cache(self.config.editor.cache_dir, self.config)
        for _ in range(self.config.editor.n_epochs):
            if self.config.num_seq == 1:
                self.train_batch(train_loader)
            else:
                self.train_seq(train_loader)
            self.reset_model()
            self.original_params = None
            del self.original_params
            if self.config.editor.save_checkpoint:
                for i in range(self.config.editor.iters):
                    torch.save(self.nets[i].state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.dataset.n_edits)}_net_{i}.pth")
                torch.save(self.opt.state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.dataset.n_edits)}_opt.pth")
                print("-----Saved checkpoints-----")
            if self.config.editor.full_curve == True:
                self.sequential_valid_full(valid_loader)
            else:
                self.sequential_valid(valid_loader)

        empty_cache(self.config.editor.cache_dir, self.config)

    def train_seq(self, loader: DataLoader, save=False):
        """
        The training method for LightEdit.

        """

        max_steps = self.config.num_seq

        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Train", ncols=100, total=max_steps)):
            l2_reg_loss = 0
            param_shifts_dict = {}
            for net_id in range(self.config.editor.iters):
                self.cache(tuples["edit_tuples"], net_id)
                param_shifts_dict[net_id] = self.predict_param_shifts(net_id)
                self.model.zero_grad()

                self.edit_model(param_shifts_dict[net_id], False)

            for module_name in self.config.model.edit_modules:
                module = get_module(self.model, module_name)
                l2_reg_loss += torch.sum((module.weight - self.original_params[module_name]) ** 2)
            l2_reg_loss *= self.config.editor.reg_coef

            gen_losses_show = []
            tot_loss = 0
            loss_eq = 0
            for t in tuples["equiv_tuples"]:
                if "old_labels" in t:
                    old_labels = t.pop("old_labels")
                logits = self.model(**t)["logits"]
                try:
                    t["old_labels"] = old_labels
                except:
                    pass
                loss = cross_entropy(logits, t["labels"])
                loss_eq += loss
            gen_losses_show.append(loss_eq.item())

            tot_loss = loss_eq
            l2_reg_loss = l2_reg_loss.to(tot_loss.device)
            tot_loss += self.config.editor.reg_coef * l2_reg_loss
            tot_loss.backward()

            self.update_hypernet(param_shifts_dict, False)

            swanlab.log({
                "gen_loss": float(np.mean(gen_losses_show)),
            })

        self.opt.step()
        self.opt.zero_grad()
        del param_shifts_dict, tot_loss, loss, loss_eq, gen_losses_show, l2_reg_loss
        torch.cuda.empty_cache()


        if save:
            torch.save(self.net, f"checkpoints/hypernet.pt")
        
    def train_batch(self, loader: DataLoader, save=False):
        """
        Original training method for MEND and MALMEN, which is for one-shot knowledge editing.
        """
        max_steps = self.config.editor.max_steps
        if max_steps == -1:
            max_steps = len(loader)
            limited_loader = loader
        else:
            limited_loader = islice(loader, max_steps)
        for _, tuples in enumerate(tqdm(limited_loader, desc = "Train", ncols = 100, total=max_steps)):
            gen_losses = []
            loc_losses = []
            for net_id in range(self.config.editor.iters):
                self.cache(tuples["edit_tuples"], net_id)
                param_shifts = self.predict_param_shifts(net_id)
                self.model.zero_grad()

                self.edit_model(param_shifts, False)
                for t in tuples["equiv_tuples"]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    loss = cross_entropy(logits, t["labels"])
                    loss.backward()
                    gen_losses += [loss.item()]
                l2_reg_loss = 0
                for module_name in self.config.model.edit_modules:
                    module = get_module(self.model, module_name)
                    l2_reg_loss += torch.sum((module.weight - self.original_params[module_name]) ** 2)
                l2_reg_loss *= self.config.editor.reg_coef
                l2_reg_loss.backward()
                self.edit_model(param_shifts, True)


                self.update_hypernet_batch(param_shifts, update=True, net_id=net_id)

            swanlab.log({
                "gen_loss": np.mean(gen_losses),
                "loc_loss": np.mean(loc_losses)
            })
        del param_shifts, loss, loc_losses, gen_losses
        torch.cuda.empty_cache()
        if save:
            for i in range(self.config.editor.iters):
                torch.save(self.nets[i], f"hypernet_{i}.pt")

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

    def update_hypernet_batch(self, param_shifts: Dict[str, torch.FloatTensor], update: bool, net_id: int=None):
        
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
            param_shift = param_shifts[module_name].to(self.config.editor_device)
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
            
        clip_grad_norm_(
            self.nets[net_id].parameters(),
            self.config.editor.max_grad_norm
        )

        if update == True:
            self.opt.step()
            self.opt.zero_grad()