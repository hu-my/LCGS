# LCGS
_________________

This repo contains the reference source code in PyTorch for the IJCAI 2022 paper [Learning Continuous Graph Structure with Bilevel Programming for Graph Neural Networks](https://www.ijcai.org/proceedings/2022/0424.pdf)

### Dependencies

The code is built with following libraries:

- python 3.7
- [PyTorch](https://pytorch.org/) 1.7.1
- scipy 1.6
- scikit-learn 0.24.1
- networkx 2.6.3

#### Installation
```
conda create -n LCGS python=3.7
source activate LCGS
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install scipy
pip install scikit-learn
pip install networkx
```

#### Dataset prepare
Please download the dataset from [GCN](https://github.com/tkipf/gcn), then change the filepath in ```utils.py``` for your path.

### Usage

The main script is in the file ```main.py```. The options are
```
--dataset: the evaulation dataset. Available datasets are cora, citeseer and pubmed. Default Cora.
--seed: the random seed. Default 1.
--gpu: the gpu device number (must be a single number). Default 0.
--name: the save name for checkpoint. Default None.
```

For experiments with normal graph scenario on Cora run
```
python main.py --dataset cora --name {if if you want to specify the save name}
```

Our codebase is developed based on the [LDS](https://github.com/lucfra/LDS-GNN) from paper [Learning Discrete Structures for Graph Neural Networks](http://proceedings.mlr.press/v97/franceschi19a.html).
