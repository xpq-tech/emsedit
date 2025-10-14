import hydra
from omegaconf import DictConfig, OmegaConf
import importlib
from data.base import make_loader
from model import make_model
import swanlab
import torch

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
  
    swanlab.init(
        project = f"{config.dataset.name}_{config.model.name}",
        name = f"{config.editor.name}_{str(config.dataset.n_edits)}",
        config = OmegaConf.to_container(config, resolve = True)
    )
    
    data_module = importlib.import_module(f"data.{config.dataset.name}")
    data_class = getattr(data_module, f"{config.dataset.name.upper()}Dataset")

    train_loader, valid_loader = make_loader(config, data_class)

    model = make_model(config.model, config.model_device)

    editor_module = importlib.import_module(f"editor.{config.editor.name}")
    editor_class = getattr(editor_module, config.editor.name.upper())
    editor = editor_class(config, model)

    editor.run(train_loader, valid_loader)
    swanlab.log({
                "max_memory_allocated": torch.cuda.max_memory_allocated(),
                "max_memory_reserved": torch.cuda.max_memory_reserved(),
            })


if __name__ == "__main__":
    main()