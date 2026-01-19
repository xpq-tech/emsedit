# EMSEdit
This is repository for the paper [EMSEdit: Efficient Multi-Step Meta-Learning-based Model Editing](https://arxiv.org/abs/2508.04012), WWW2026.

## Setup Instructions

### 1. Environment Preparation
Create and activate a conda environment:
```shell
conda create -n emsedit python==3.10
conda activate emsedit
```

### 2. Install Dependencies
Install required packages:
```shell
pip install -r requirements.txt
```

### 3. Model Setup
Download your LLM models and configure their paths in the corresponding YAML files under `config/model/`.

## Running Experiments

Execute experiments using the following command structure:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py \
    dataset=[zsre|counterfact|ripple-effect] \
    model=[llama-3-instruct|gptj|gemma-2] \
    editor=[emsedit|rledit|malmen|ultraedit|mend] \
    num_seq=[number_of_sequences] \
    dataset.n_edits=[batch_size_per_sequence]
```

### Parameter Explanation:

| Parameter         | Description                                                                 | Examples                     |
|-------------------|-----------------------------------------------------------------------------|------------------------------|
| `dataset`         | Dataset for editing and evaluation                                         | `zsre`, `counterfact`, `ripple-effect` |
| `model`           | Base LLM for editing                                                       | `llama-3-instruct`, `gptj`, `gemma-2` |
| `editor`          | Editing method to use                                                      | `emsedit`, `rledit`, `malmen`, `ultraedit`, `mend`|
| `num_seq`         | Number of editing sequences (e.g., 400 for 400×n edits)                    | 400 (for sequential edits)   |
| `dataset.n_edits` | Batch size per sequence (e.g., 1024 for 1×1024 batch edits)                | 1024 (for batch edits)       |

### Usage Examples:

1. For sequential editing (400x20 sequences):
```shell
CUDA_VISIBLE_DEVICES=0 python main.py dataset=zsre model=llama-3-instruct editor=smedit num_seq=400 dataset.n_edits=20
```

2. For batch editing (single sequence with 1024 edits):
```shell
CUDA_VISIBLE_DEVICES=0 python main.py dataset=counterfact model=gptj editor=rledit num_seq=1 dataset.n_edits=1024
```

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{li2025emsedit,
  title={EMSEdit: Efficient Multi-Step Meta-Learning-based Model Editing},
  author={Xiaopeng Li, Shasha Li, Xi Wang, Shezheng Song, Bin Ji, Shangwen Wang, Jun Ma, Xiaodong Liu, Mina Liu and Jie Yu},
  journal={arXiv preprint arXiv:2508.04012},
  year={2025}
}
```