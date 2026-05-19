# Dataset Preparation

Run all commands from the project root.
```
mkdir -p data/datasets
```
If you prefer to download manually, place the files at the paths shown below (after the `→`).

## 1. OOD datasets

### ImageNet-O
[ImageNet-O](https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar) `→` `data/datasets/imagenet-o`
```
curl -L -o imagenet-o.tar "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
tar -xf imagenet-o.tar && mv imagenet-o data/datasets/
```

### DTD
[DTD](https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz) `→` `data/datasets/dtd`
```
curl -L -o dtd-r1.0.1.tar.gz "https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz"
tar -xzf dtd-r1.0.1.tar.gz && mv dtd data/datasets/
```

### Places365
[Places365 val_256](https://data.csail.mit.edu/places/places365/val_256.tar) `→` `data/datasets/Places365/val_256`
```
curl -L -o val_256.tar "https://data.csail.mit.edu/places/places365/val_256.tar"
tar -xf val_256.tar && mkdir -p data/datasets/Places365 && mv val_256 data/datasets/Places365/
```

## 2. ID datasets
### 2.1 CUB-200-2011
[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1) `→` `data/datasets/CUB-200-2011`
```
curl -L -o CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
mkdir -p data/datasets/CUB_200_2011 && tar -xzf CUB_200_2011.tgz -C data/datasets/CUB_200_2011
```

### 2.2 Stanford Dogs
[Standord Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) `→` `data/datasets/StanfordDogs`
```
curl -L -o images.tar "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
mkdir -p data/datasets/StanfordDogs && tar -xf images.tar -C data/datasets/StanfordDogs/
```

### 2.3 Tiny-ImageNet
[Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) `→` `data/datasets/tiny-imagenet-200`
```
curl -L -o tiny-imagenet-200.zip "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
unzip tiny-imagenet-200.zip && mv tiny-imagenet-200 data/datasets/
```

## 3. (Optional) Convert to HDF5.
Recommended if your system has a file count limit (e.g. on HPC clusters).
```
python3 data/convert_to_hdf5.py --src data/datasets/imagenet-o --dst data/datasets/imagenet-o.hdf5
python3 data/convert_to_hdf5.py --src data/datasets/dtd/images --dst data/datasets/dtd/images.hdf5
python3 data/convert_to_hdf5.py --src data/datasets/Places365/val_256 --dst data/datasets/Places365/val_256.hdf5
python3 data/convert_to_hdf5.py --src data/datasets/CUB_200_2011/CUB_200_2011/images --dst data/datasets/CUB_200_2011/CUB_200_2011/images.hdf5
python3 data/convert_to_hdf5.py --src data/datasets/StanfordDogs/Images --dst data/datasets/StanfordDogs/Images.hdf5
python3 data/convert_to_hdf5.py --src data/datasets/tiny-imagenet-200/train --dst data/datasets/tiny-imagenet-200/train.hdf5
python3 data/convert_to_hdf5.py --src data/datasets/tiny-imagenet-200/val/images --dst data/datasets/tiny-imagenet-200/val/images.hdf5
```

## 4. Acknowledgement
The CUB-200-2011 and Stanford Dogs split files in `data/splits` are from [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)