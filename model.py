import os

# You can specify the GPU you are using here
# os.environ['CUDA_VISIBLE_DEVICES'] =  '0'

from omegaconf import DictConfig

import torch

import transformers


from util import get_module


def make_model(config: DictConfig, device: str):

    device_map = 'auto' if device=="auto" else None
    model_class = getattr(transformers, config.class_name)
    model = model_class.from_pretrained(config.name_or_path, device_map=device_map, torch_dtype = torch.bfloat16 if config.half else torch.float32)
    if device_map != 'auto':
        model.to(device)


    for param in model.parameters():
        param.requires_grad = False
        
    for module_name in config.edit_modules:
        module = get_module(model, module_name)
        module.weight.requires_grad = True
        
    return model