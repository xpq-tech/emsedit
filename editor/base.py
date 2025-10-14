from typing import Dict, List
from omegaconf import DictConfig

from collections import Counter
import numpy as np
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import islice
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import swanlab

from nets import HyperNet
from transformers import AutoTokenizer

from glue_eval.glue_eval import GLUEEval
import json

from model import make_model
from util import (
    get_module,
    get_shape,
    empty_cache,
    TracerDict,
    cross_entropy,
    kl_div,
    succ_ratios,
    pad_tensor
)


class BaseEditor:

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        
        self.config = config
        self.model = model
        
        shape_counter = Counter()
        self.name2idx = {}
        for module_name in config.model.edit_modules:
            shape = get_shape(get_module(model, module_name))
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1
        
        self.shape_counter = shape_counter

        self.tuples_list = []

        print(f"--------Init Editor: {config.editor.name}--------")
        if config.editor.load_checkpoint:
            for i in range(config.editor.iters):
                self.nets[i].load_state_dict(torch.load(f"checkpoints/{config.model.name}_{config.editor.name}_{str(config.dataset.n_edits)}_net_{i}.pth"))
            self.opt.load_state_dict(torch.load(f"checkpoints/{config.model.name}_{config.editor.name}_{str(config.dataset.n_edits)}_opt.pth"))
            print("-----Loaded checkpoints-----")

    def edit_model(
        self,
        param_shifts: Dict[str, torch.FloatTensor],
        is_reverse: bool
    ):
        
        for module_name, param_shift in param_shifts.items():
            module = get_module(self.model, module_name)
            if isinstance(module, nn.Linear):
                param_shift = param_shift.T
            if is_reverse:
                param_shift = - param_shift
            module.weight.data += param_shift.to(module.weight.data.dtype).to(module.weight.device)


    def reset_model(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = make_model(self.config.model, self.config.model_device)


    def cache(self, tuples: List[Dict[str, torch.LongTensor]], net_id: int):
        for idx, t in enumerate(tuples):
            if "old_labels" in t:
                old_labels = t.pop("old_labels")

            with TracerDict(
                self.model,
                self.config,
                t
            ) as tr:
                logits = self.model(**t)["logits"]
                loss = cross_entropy(logits, t["labels"])
                loss.backward()
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                keys = tr[module_name].keys.to(torch.float32).to(self.config.editor_device)
                values_grad = tr[module_name].values_grad.to(torch.float32).to(self.config.editor_device)
                self.nets[net_id][str(shape)].normalizer.update(torch.cat((keys, values_grad), -1))
                dir_path = f"{self.config.editor.cache_dir}/{self.config.dataset.name}_{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(keys, f"{dir_path}/{module_idx}_{idx}_keys_{net_id}.pth")
                torch.save(values_grad, f"{dir_path}/{module_idx}_{idx}_values_grad_{net_id}.pth")
            del loss, keys, values_grad
            torch.cuda.empty_cache()

            try:
                t["old_labels"] = old_labels
            except:
                pass

    def reset_hypernet(self):

        self.nets = {}
        if self.config.num_seq == 1:
            net = nn.ModuleDict({
                str(k): HyperNet(
                    *k,
                    self.config.editor.rank ,
                    self.config.editor.n_blocks,
                    v,
                    self.config.editor.lr
                )
                for k, v in self.shape_counter.items()
            }).to(self.config.editor_device)
            for i in range(self.config.editor.iters):
                self.nets[i] = net
        else:
            for i in range(self.config.editor.iters):
                self.nets[i] = nn.ModuleDict({
                    str(k): HyperNet(
                        *k,
                        self.config.editor.rank // (i+1),
                        self.config.editor.n_blocks,
                        v,
                        self.config.editor.lr
                    )
                    for k, v in self.shape_counter.items()
                }).to(self.config.editor_device)
        params = []
        for _, net in self.nets.items():
            params.extend(net.parameters())
        self.opt = torch.optim.Adam(
            params,
            self.config.editor.meta_lr
        )

    def train(self, loader: DataLoader, save=False):
        """
        Original training method for MEND and MALMEN, which is for one-shot knowledge editing.
        """
        max_steps = self.config.editor.max_steps
        if max_steps == -1:
            max_steps = len(loader)
            limited_loader = loader
        else:
            if max_steps > len(loader):
                max_steps = len(loader)
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
                self.edit_model(param_shifts, True)

                
                for t in tuples["unrel_tuples"]:

                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")

                    with torch.no_grad():
                        refer_logits = self.model(**t)["logits"]

                    self.edit_model(param_shifts, False)
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
                    (self.config.editor.loc_coef * loss).backward()
                    self.edit_model(param_shifts, True)
                    loc_losses += [loss.item()]

                self.update_hypernet(param_shifts, update=True, net_id=net_id)

            swanlab.log({
                "gen_loss": np.mean(gen_losses),
                "loc_loss": np.mean(loc_losses)
            })
        del param_shifts, loss, loc_losses, gen_losses
        torch.cuda.empty_cache()
        if save:
            for i in range(self.config.editor.iters):
                torch.save(self.nets[i], f"hypernet_{i}.pt")


    def valid(self, loader: DataLoader):
        """
        The original valid method for MEND and MALMEN, which just valid the editing of single knowledge.
        """

        for tuples in tqdm(loader, desc = "Valid", ncols = 100):
            for net_id in range(self.config.editor.iters):
                self.cache(tuples["edit_tuples"], net_id)
                param_shifts = self.predict_param_shifts(net_id)
                self.edit_model(param_shifts, False)
            edit_succs, gen_succs, loc_succs = [], [], []
            for k, s in zip(
                ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                [edit_succs, gen_succs, loc_succs]
            ):
                for t in tuples[k]:
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    s += succ_ratios(logits, t["labels"])
                    
            self.edit_model(param_shifts, True)
            
            swanlab.log({
                "ES": np.mean(edit_succs),
                "GS": np.mean(gen_succs),
                "LS": np.mean(loc_succs)
            })
            print({
                "dataset": self.config.dataset.name,
                "model": self.config.model.name,
                "editor": self.config.editor.name,
                "n_edits": self.config.dataset.n_edits,
                "iters": self.config.editor.iters,
                "num_seq": self.config.num_seq,
            })



    def sequential_valid_full(self, loader: DataLoader):
        """
        Valid the entire knowledge sequence, with the full curve showed.
        """

        max_steps = self.config.num_seq
        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

            if self.config.glue_step > 0:
                if _ == 0 or (_+1) % self.config.glue_step == 0:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                    glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                    out_file = f"glue_eval/results/{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{_}_{self.config.dataset.name}"
                    if not os.path.exists(out_file):
                        os.makedirs(out_file, exist_ok=True)
                    out_file = f"{out_file}/glue.json"
                    glue_results = {'edit_num': -1}
                    glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    with open(out_file, "w") as f:
                        json.dump(glue_results, f, indent=4)
            for net_id in range(self.config.editor.iters):
                self.cache(tuples["edit_tuples"], net_id)
                param_shifts = self.predict_param_shifts(net_id)
                self.edit_model(param_shifts, False)
            del param_shifts
            torch.cuda.empty_cache()
            self.tuples_list.append(tuples)
            edit_succs, gen_succs, loc_succs = [], [], []
            for k, s in zip(
                ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                [edit_succs, gen_succs, loc_succs]
            ):
                for tuple in self.tuples_list:
                    for t in tuple[k]:
                        if "old_labels" in t:
                            old_labels = t.pop("old_labels")
                        with torch.no_grad():
                            logits = self.model(**t)["logits"]
                        try:
                            t["old_labels"] = old_labels
                        except:
                            pass
                        if self.config.dataset.name == "counterfact":
                            t["old_labels"] = old_labels
                            s += succ_ratios(logits, t["labels"], t["old_labels"])
                        else:
                            s += succ_ratios(logits, t["labels"])

            swanlab.log({
                "ES": np.mean(edit_succs),
                "GS": np.mean(gen_succs),
                "LS": np.mean(loc_succs)
            })

            self.opt.zero_grad()


    def sequential_valid(self, loader: DataLoader):
        """
        Valid the entire knowledge sequence, with just final results showed.
        """

        max_steps = self.config.num_seq
        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

            if self.config.glue_step > 0:
                if _ == 0 or (_+1) % self.config.glue_step == 0:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                    glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                    out_file = f"glue_eval/results/{self.config.model.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{_}_{self.config.dataset.name}"
                    if not os.path.exists(out_file):
                        os.makedirs(out_file, exist_ok=True)
                    out_file = f"{out_file}/glue.json"
                    glue_results = {'edit_num': -1}
                    glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    with open(out_file, "w") as f:
                        json.dump(glue_results, f, indent=4)
            for net_id in range(self.config.editor.iters):
                self.cache(tuples["edit_tuples"], net_id)
                param_shifts = self.predict_param_shifts(net_id)
                self.edit_model(param_shifts, False)
            del param_shifts
            torch.cuda.empty_cache()
            self.tuples_list.append(tuples)
            if hasattr(self, 'opt'):
                self.opt.zero_grad()

        edit_succs, gen_succs, loc_succs = [], [], []
        for k, s in zip(
            ["edit_tuples", "equiv_tuples", "unrel_tuples"],
            [edit_succs, gen_succs, loc_succs]
        ):
            for tuple in self.tuples_list:
                for t in tuple[k]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    if self.config.dataset.name == "counterfact":
                        t["old_labels"] = old_labels
                        s += succ_ratios(logits, t["labels"], t["old_labels"])
                    else:
                        s += succ_ratios(logits, t["labels"])

        swanlab.log({
            "ES": np.mean(edit_succs),
            "GS": np.mean(gen_succs),
            "LS": np.mean(loc_succs)
        }, print_to_console=True)


    def run(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use MEND or MALMEN to complete sequential editing task.
        Just train the hypernet on the original LLM, then freeze it.
        """
        empty_cache(self.config.editor.cache_dir, self.config)
        for _ in range(self.config.editor.n_epochs):

            self.train(train_loader)
            self.reset_model()
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



    def run_single(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Original run function in MEND and MALMEN, single edit, which means training a hypernet and edit 1 batch of knowledge.
        """

        empty_cache(self.config.editor.cache_dir, self.config)
        self.train(train_loader)
        for _ in range(self.config.editor.n_epochs):
            self.valid(valid_loader)
        if self.config.editor.save_checkpoint:
            for i in range(self.config.editor.iters):
                torch.save(self.nets[i].state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.dataset.n_edits)}_net_{i}.pth")
            torch.save(self.net.state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.dataset.n_edits)}_net.pth")
            torch.save(self.opt.state_dict(), f"checkpoints/{self.config.model.name}_{self.config.editor.name}_{str(self.config.dataset.n_edits)}_opt.pth")


    def run_sequential_retrain_full(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use MEND or MALMEN to complete sequential editing task.
        Before editing the next batch of knowledge, we retrain the hypernet on post-edited LLM in order to keep its ability.
        This function will show the full curve.
        """

        empty_cache(self.config.editor.cache_dir, self.config)
        for _ in range(self.config.editor.n_epochs):
            
            max_steps = self.config.num_seq
            limited_loader = islice(valid_loader, max_steps)

            for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

                if _ % 3 == 0:
                    self.train(train_loader)

                if self.config.glue_step > 0:
                    # if _ == 0 or (_+1) % self.config.glue_step == 0:
                    if (_+1) % self.config.glue_step == 0:
                        tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                        glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                        out_file = f"glue_eval/results/{self.config.model.name}_{_}_{self.config.dataset.name}"
                        if not os.path.exists(out_file):
                            os.makedirs(out_file, exist_ok=True)
                        out_file = f"glue_eval/results/{self.config.model.name}_{_}_{self.config.dataset.name}/glue.json"
                        glue_results = {'edit_num': -1}
                        glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                        with open(out_file, "w") as f:
                            json.dump(glue_results, f, indent=4)
                for net_id in range(self.config.editor.iters):
                    self.cache(tuples["edit_tuples"], net_id)
                    param_shifts = self.predict_param_shifts(net_id)
                    self.edit_model(param_shifts, False)
                self.tuples_list.append(tuples)

                edit_succs, gen_succs, loc_succs = [], [], []
                for k, s in zip(
                    ["edit_tuples", "equiv_tuples", "unrel_tuples"],
                    [edit_succs, gen_succs, loc_succs]
                ):
                    for tuple in self.tuples_list:
                        for t in tuple[k]:
                            # print(t["labels"].shape)
                            # print(t["old_labels"].shape)
                            if "old_labels" in t:
                                old_labels = t.pop("old_labels")
                            with torch.no_grad():
                                logits = self.model(**t)["logits"]
                            try:
                                t["old_labels"] = old_labels
                            except:
                                pass
                            if self.config.dataset.name == "counterfact":
                                t["old_labels"] = old_labels
                                # print(f"logits.shape = {logits.shape}")
                                s += succ_ratios(logits, t["labels"], t["old_labels"])
                            else:
                                s += succ_ratios(logits, t["labels"])

                swanlab.log({
                    "ES": np.mean(edit_succs),
                    "GS": np.mean(gen_succs),
                    "LS": np.mean(loc_succs)
                })

                if _ % 3 == 0:
                    empty_cache(self.config.editor.cache_dir, self.config)
                    self.opt.zero_grad()
                    self.reset_hypernet()

    
    def run_sequential_retrain(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use MEND or MALMEN to complete sequential editing task.
        Before editing the next batch of knowledge, we retrain the hypernet on post-edited LLM in order to keep its ability.
        This function will only show final results.
        """

        max_steps = self.config.num_seq
        limited_loader = islice(valid_loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Valid", ncols=100, total=max_steps)):

            self.train(train_loader)
            if self.config.glue_step > 0:
                if _ == 0 or (_+1) % self.config.glue_step == 0:
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model.name_or_path)
                    glue_eval = GLUEEval(self.model, tokenizer, number_of_tests = 100)
                    out_file = f"glue_eval/results/{self.config.model.name}_{_}_{self.config.dataset.name}"
                    if not os.path.exists(out_file):
                        os.makedirs(out_file, exist_ok=True)
                    out_file = f"glue_eval/results/{self.config.model.name}_{_}_{self.config.dataset.name}/glue.json"
                    glue_results = {'edit_num': -1}
                    glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    with open(out_file, "w") as f:
                        json.dump(glue_results, f, indent=4)
            for net_id in range(self.config.editor.iters):
                self.cache(tuples["edit_tuples"], net_id)
                param_shifts = self.predict_param_shifts(net_id)
                self.edit_model(param_shifts, False)
            self.tuples_list.append(tuples)
            self.opt.zero_grad()
            self.reset_hypernet()

        edit_succs, gen_succs, loc_succs = [], [], []
        for k, s in zip(
            ["edit_tuples", "equiv_tuples", "unrel_tuples"],
            [edit_succs, gen_succs, loc_succs]
        ):
            for tuple in self.tuples_list:
                for t in tuple[k]:
                    if "old_labels" in t:
                        old_labels = t.pop("old_labels")
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    try:
                        t["old_labels"] = old_labels
                    except:
                        pass
                    if self.config.dataset.name == "counterfact":
                        t["old_labels"] = old_labels
                        # print(f"logits.shape = {logits.shape}")
                        s += succ_ratios(logits, t["labels"], t["old_labels"])
                    else:
                        s += succ_ratios(logits, t["labels"])

        swanlab.log({
            "ES": np.mean(edit_succs),
            "GS": np.mean(gen_succs),
            "LS": np.mean(loc_succs)
        })

    def predict_param_shifts(self, net_id) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
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
            value_diffs = torch.empty((0, net.value_size), device = self.config.editor_device)
            for start_idx in range(0, keys.shape[0], self.config.editor.batch_size):
                end_idx = start_idx + self.config.editor.batch_size
                keys_once = pad_tensor(keys[start_idx:end_idx], self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor.batch_size, 0)
                with torch.no_grad():
                    (pesudo_keys, pesudo_values_grad) = net(
                        keys_once,
                        values_grad_once,
                        layer_idx,
                    )
                    coeffs = - net.lr(layer_idx) * (keys_once * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diffs = torch.cat((value_diffs, coeffs * pesudo_values_grad))
            with torch.no_grad():
                mat = keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device=self.config.editor_device)
            value_diffs = value_diffs[:keys.shape[0], :]
            param_shift = torch.linalg.solve(mat, keys.T @ value_diffs)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)
            
        return param_shifts

    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor], update: bool, net_id: int=None):
        
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
                keys_once = pad_tensor(keys[start_idx:end_idx], self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor.batch_size, 0)
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