## TUUN showcase

#### Basenet

Classes wrapping basic `pytorch` functionality.
  
  - Train for an epoch
  - Eval for an epoch
  - Predict
  - Learning rate schedules

##### Installation

###### Basenet
```
conda create -n basenet_env python=3.6 pip -y
source activate basenet_env

pip install -r requirements.txt

## note cuda is 9.0 here; you can change it to fit your cuda version
conda install -y pytorch torchvision cuda90 -c pytorch
python setup.py install
```

###### Tuun (containing installing NNI)
```
git clone git@github.com:petuum/tuun.git
cd tuun
pip install -r requirements/requirements_dev.txt
python tuun/probo/models/stan/compile_models.py -m gp_distmat_fixedsig
```
After installing Tuun, we need to add Tuun to the PYTHONPATH.
```
export PYTHONPATH="{TUUN_REPO_PARENT_DIR_PATH}/tuun:$PYTHONPATH"
```



#### Run the NNI Experiments

```
conda activate basenet_env # activate conda
# optional you could do 
# source activate basenet_env
# if you have an older version of conda
cd cifar
```
To run a single training without autotuning, you can directly run
```
python cifar10.py
```



`auto_config_tuun.yml` is the configuration file. Before running, remember to specify the code directory for Tuun in your configuration file by changing `codeDir`.
In order to run the NNI experiments using Tuun, 
```
nnictl create --config auto_config_tuun.yml
```
