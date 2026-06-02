# SR-GM
This is the Pytorch implementation of "**Decoupling and Damping: Structurally-Regularized Gradient Matching for Multimodal Graph Condensation**"
## Requirements
Please see `requirements.txt`
```
click==8.3.1
deeprobust==0.2.11
hydra-core==1.3.2
matplotlib==3.7.3
networkx==3.3
numpy==1.24.0
ogb==1.3.6
omegaconf==2.3.0
pandas==3.0.0
scikit_learn==1.8.0
scipy==1.17.0
torch==2.1.1+cu118
torch_geometric==2.7.0
torch_scatter==2.1.2+pt21cu118
torch_sparse==0.6.18+pt21cu118
torchvision==0.16.1+cu118
tqdm==4.67.1
```

## Datasets
Features for Ele-fashion and Goodreads-NC are generated via [CLIP](https://proceedings.mlr.press/v139/radford21a), while those for Amazon-Sports and Amazon-Cloth are generated via [ImageBind](https://openaccess.thecvf.com/content/CVPR2023/html/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.html). Add the `data/` folder to the root directory and execute the main script to start graph condensation.

## Instructions
### Run the code
1. Configuration

For Ele-fashion and Goodreads-NC, first find the configuration in `configs/defaults.yaml`. 

To **start a new condensation**, set the parameters as follows:
```
save: False
training: True
```
**Note**: Enabling `save`(True) will store the condensed graph in `saved-ours/srgm/`, **overwriting the previous condensed graph**.

For Amazon-Sports and Amazon-Cloth, find the configuration in `configs/amazon_review.yaml` and follow the same instructions. 

2. Command

For Ele-fashion and Goodreads-NC, please run the following command:
```
python sr_mt_ind.py
```
For Amazon-Sports and Amazon-Cloth, please run the following command:
```
python sr_mt_trans.py
```

### Reproduce the performance
The condensed graphs can be found in the `saved-ours/srgm/` folder; you can directly load them to test the performance.
1. Configuration

To **load the condensed graphs and test the performance**, set the parameters as follows:
```
training: False
```
Next, configure the `dataset` and `reduction_rate` to test different condensed graphs at varying condensation ratios. The table below lists the available configuration choices and their corresponding datasets:

| Dataset | `dataset` Parameter | Supported `reduction_rate` |
|-|:-:|:-:|
| Goodreads-NC | books_nc | 0.00025, 0.0005 |
| Ele-fashion | ele-fashion | 0.001, 0.003, 0.005 |
| Amazon-Sports | nc_ar_sport | 0.001, 0.003, 0.005 |
| Amazon-Cloth | nc_ar_cloth | 0.001, 0.003, 0.005 |

2. Command

For Ele-fashion and Goodreads-NC, please run the following command:
```
python sr_mt_ind.py
```
For Amazon-Sports and Amazon-Cloth, please run the following command:
```
python sr_mt_trans.py
```

3. Example

Dataset: Goodreads-NC, Ratio: 0.0005 (0.05%)

Change the configuration in `configs/defaults.yaml`:
```
dataset: books_nc
reduction_rate: 0.0005
```
And run the command:
```
python sr_mt_ind.py
```

[![DOI:10.5281/zenodo.20503552](https://zenodo.org)](https://doi.org)
