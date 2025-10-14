from typing import Tuple
import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):

    def __init__(self, size: int, eps = 1e-8):
        super().__init__()

        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("mean", torch.zeros((size)))
        self.register_buffer("var", torch.zeros((size)))
        self.register_buffer("std", torch.ones((size))) 
        self.eps = eps


    def update(self, x: torch.FloatTensor):
        m = x.shape[0]
        if m == 0: 
            return
    
        x_mean = x.mean(0)
        x_var = x.var(0, unbiased=False)
   
        n_new = self.n + m
        delta = x_mean - self.mean
        new_mean = self.mean + delta * m / n_new
       
        M2 = self.var * self.n + x_var * m + delta**2 * self.n * m / n_new
 
        self.var = M2 / n_new.clamp_min(1)  
        self.mean = new_mean
        self.n = n_new
        self.std = (M2 / (n_new - 1 + self.eps)).sqrt()


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return (x - self.mean) / (self.std + torch.finfo(x.dtype).eps)


class HyperNetBlock(nn.Module):

    def __init__(self, size: int, rank: int, n_modules: int):
        super().__init__()

        self.A = nn.Parameter(torch.randn(size, rank))
        self.B = nn.Parameter(torch.zeros(rank, size))
        self.bias = nn.Parameter(torch.zeros(size))
        
        self.scale = nn.Embedding(n_modules, size)
        self.shift = nn.Embedding(n_modules, size)
        
        self.scale.weight.data.fill_(1)
        self.shift.weight.data.fill_(0)


    def forward(
        self,
        y: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> torch.FloatTensor:

        x = y @ self.A @ self.B + self.bias
        x = x.clamp(0)

        x = self.scale(module_idx) * x + self.shift(module_idx)
        x = x + y

        return x


class HyperNet(nn.Module):

    def __init__(
        self,
        key_size: int,
        value_size: int,
        rank: int,
        n_blocks: int,
        n_modules: int,
        lr: float
    ):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        self.normalizer = RunningMeanStd(key_size + value_size)
        self.blocks = nn.ModuleList([
            HyperNetBlock(key_size + value_size, rank, n_modules)
            for _ in range(n_blocks)
        ])

        self.lr = nn.Embedding(n_modules, 1)
        self.lamda = nn.Embedding(n_modules, 1)
        
        self.lr.weight.data.fill_(lr)
        self.lamda.weight.data.fill_(0)


    def forward(
        self,
        keys: torch.FloatTensor,
        values_grad: torch.FloatTensor,
        module_idx: torch.LongTensor
    ) -> Tuple[torch.FloatTensor]:

        hidden_states = torch.cat((keys, values_grad), -1)
        hidden_states = self.normalizer(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, module_idx)
        return hidden_states.split([self.key_size, self.value_size], -1)