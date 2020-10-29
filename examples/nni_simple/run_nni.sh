python3 -m pip install -e .
nnictl package install .
nnictl create --config examples/nni_simple/config.yml
