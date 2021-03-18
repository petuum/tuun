## Tuun data augmentation
##### Installation
###### This example
```shell
conda create -n augmentation python=3.6 pip -y
conda activate augmentation
pip install -r requirements.txt
## note cuda is 9.0 here; you can change it to fit your cuda version
conda install -y pytorch torchvision cuda90 -c pytorch
python setup.py install
```

###### [Tuun](https://github.com/petuum/tuun)
```shell
cd ..
pip install -r requirements/requirements_dev.txt
python tuun/probo/models/stan/compile_models.py -m gp_distmat_fixedsig
```
Include Tuun in PYTHONPATH.
```shell
export PYTHONPATH="{TUUN_REPO_PARENT_DIR_PATH}/tuun:$PYTHONPATH"
```

###### [NNI](https://github.com/microsoft/nni)
```shell
python -m pip install --upgrade nni
```

#### Run NNI Experiments

Specify the code directory for Tuun `nni_config.yml` by changing `codeDir`,
then run
```shell
nnictl create --config nni_config.yml
```
