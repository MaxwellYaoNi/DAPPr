<br />
<p align="center">
    <h1 align="center">
        Possibilistic Predictive Uncertainty for Deep Learning (ICML 2026)
    </h1>

  <p align="center">
    <p align="center">
    <a href="https://scholar.google.com/citations?user=oGD-WMQAAAAJ"><strong>Yao Ni</strong></a>
    ,
    <a href="https://jeremiehoussineau.com/"><strong>Jeremie Houssineau</strong></a>
    ,
    <a href="https://personal.ntu.edu.sg/asysong/home.html"><strong>Yew Soon Ong</strong></a>
    ,
    <a href="https://www.koniusz.com/"><strong>Piotr Koniusz</strong></a>
    </p>

  
  <p align="center">
    <a href='https://arxiv.org/abs/2605.00600'>
      <img src='https://img.shields.io/badge/Paper-arXiv-80261B?style=flat&logo=Googledocs&logoColor=white' alt='Paper arXiv'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Slides-2AA26C?style=flat&logo=Slides&logoColor=white' alt='Slides'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Poster-2AA26C?style=flat&logo=Packt&logoColor=white' alt='Slides'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Video-Youtube-FA1D1D?style=flat&logo=youtube&logoColor=white' alt='Video Youtube'>
    </a>
  </p>
<br/>

## Usage Guide
[`DAPPr.py`](DAPPr.py) provides a drop-in replacement for cross entropy and uncertainty scores for testing.
```diff
import torch.nn.functional as F
+from DAPPr import DAPPr_loss, DAPPr_uncertainty

# Training
for x, labels in train_loader:
    logits = model(x)
-   loss = F.cross_entropy(logits, labels)
+   loss = DAPPr_loss(logits, labels, lamb=2e-4)

# Testing: uncertainty estimation
for x, labels in test_loader:
    logits = model(x)
+   uncertainty = DAPPr_uncertainty(logits)
+   AU = uncertainty["AU"]  # aleatoric uncertainty
+   EU = uncertainty["EU"]  # epistemic uncertainty
```
`lamb` controls the regularization strength. In our experiments, we use either warm-up or linear scheduling for this value.

## Environment
The code was tested with Python 3.11.7. Required Python packages are listed in [`requirements.txt`](requirements.txt).

## Datasets
Please follow [data/README.md](data/README.md) to download and prepare the datasets.

## Training for DAPPr
```bash
# CUB-200-2011
python3 train.py --dataset=CUB

# Stanford Dogs
python3 train.py --dataset=StanfordDogs 

# Tiny-ImageNet
python3 train.py --dataset=TinyImageNet --epochs=100 --lr=5e-3 --lamb=5e-3 --lamb_schedule=linear
```

## Optional HDF5 Storage
The code supports both image-folder and HDF5 loading. Image-folder loading is the default.

If your filesystem is slow or has a file-count limit, follow [HDF5 conversion instructions](data/README.md#3-optional-convert-to-hdf5) to convert image folders to HDF5. Then add `--hdf5`:
```diff
python3 train.py --dataset=CUB \
+ --hdf5
```

## Running time 
Reference running time on one V100 32GB GPU using image folders or HDF5 storage:
Dataset|Image Folders|HDF5
:-|:-:|:-:
CUB-200-2011|1h50m|**1h20m**
StanfordDogs|2h40m|**2h10m**
TinyImageNet|10h40m|**10h20m**

## Commands for EDL
```bash
python3 train.py --dataset=CUB --method=EDL --lamb=1e-5
python3 train.py --dataset=StanfordDogs --method=EDL --lamb=1e-5
python3 train.py --dataset=TinyImageNet --method=EDL --epochs=100 --lr=5e-3 --lamb=2e-4 --lamb_schedule=linear
```

## Notes
**Hyperparameter:** For new datasets, we recommend tuning `--lamb` in the range `1e-5` to `1e-2`. `linear` schedules usually require a larger value than `warmup`.

**Entropy for OOD:** For OOD detection, entropy is often stronger than epistemic uncertainty in our experiments. The code supports both.

## Cite
```
@article{ni2026possibilistic,
  title={Possibilistic Predictive Uncertainty for Deep Learning},
  author={Ni, Yao and Houssineau, Jeremie and Ong, Yew Soon and Koniusz, Piotr},
  journal={arXiv preprint arXiv:2605.00600},
  year={2026}
}
```

