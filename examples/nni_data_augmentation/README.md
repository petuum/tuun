# Tuun data augmentation
This example shows how Tuun can be used to search data augmentation policies. The model used in this example is [Resnet18](https://github.com/bkj/basenet/blob/819db22359e77ae6b4d0424d1379b84999fa20ac/examples/cifar/cifar10.py#L125) in the [Basenet](https://github.com/bkj/basenet) repo.
## Environment setup
### Python environment
#### Conda(recommended)
```shell
conda create -n augmentation python=3.6 pip -y
conda activate augmentation
```
#### Python venv(requires python>=3.6)
```shell
python -m venv env
source env/bin/activate
```
### Install dependencies
First `cd` into this folder, then
```shell
pip install -r requirements.txt
```
#### Install PyTorch
##### Conda
```shell
# Note that cuda version is set to 9.0 here, you may change it to fit your needs
conda install -y pytorch torchvision cuda90 -c pytorch
```
##### Python venv
```shell
# Make sure cuda is installed correctly as required by PyTorch, then
pip install torch torchvision
```

#### Install [Tuun](https://github.com/petuum/tuun)
```shell
pip install -r $(pwd)/../../requirements/requirements_dev.txt
python $(pwd)/../../tuun/probo/models/stan/compile_models.py -m gp_distmat_fixedsig
# Include Tuun in PYTHONPATH.
export PYTHONPATH=$(pwd)/../../tuun:$PYTHONPATH
```

#### Install [NNI](https://github.com/microsoft/nni)
```shell
python -m pip install --upgrade nni
```

## Run NNI Experiments

Specify the code directory for Tuun `nni_config.yml` by changing `codeDir`, then
```shell
nnictl create --config nni_config.yml
```
