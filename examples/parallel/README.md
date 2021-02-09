<p align="center"><img src="docs/images/tuun_logo.svg" width=280 /></p>

<p align="center">
    &emsp;&emsp;
    <a href="https://petuum.github.io/tuun">Documentation</a> and
    <a href="https://github.com/petuum/tuun/tree/master/examples">Examples</a>
</p>

This repo gives an example as well as instruction about how to run **NNI** with **Tuun** in parallel. As other tuners, it is very straightforward to run Tuun in NNI for parallel tuning. The user just needs to follow NNI instruction and we still provide examples with mnist-pytorch for how to run it in parallel. 


## Parallel on a Single Node

To run paralleling search with NNI on a single Node, the user only needs to change the `trialConcurrency`  to be more than one. For the experiments using GPU, if the user wants to have only one GPU for one experiment the user can set `gpuNum`  and `maxTrialNumPerGpu` to be one. The configuration file `config_tuun.yml` is provided as an example.

## Parallel on Multiple Nodes
To run paralleling search on multiple nodes, we first need to specify `nniManagerIp`, which is the IP address of the machine with NNI manager (NNICTL) that launches NNI experiment. Typically, it is the IP address for your local machine which you install and run NNI. Then you need to provide the IP address of your worker machines in `machineList` as well as the username and passwords for these machines if you use a password authentication of SSH for these machines. If you prefer a key authentication of SSH, please see [here](https://nni.readthedocs.io/en/stable/TrainingService/RemoteMachineMode.html). Usually the user needs to prepare a virtual environment in each node, where the required dependencies (like required in `requirements.txt`) and nni library are installed, then in the configuration file the user needs to use `preCommand` to specify and run a python environment on your remote machine before running the trial script. Similar to the parallel run on a single node, the user then still needs to change the `trialConcurrency`  to be more than one. One important thing the user needs to keep in mind is that all the training data and scripts will be firstly held on your local machine and uploaded to the `/tmp` folder on your remote machines by the NNI manager. So, the user does not need to copy these files to the remote machines but has to remember to enable the access permission of the `/tmp` folder on all the remote machines. When all is done, the user can start to run the parallel job with the modified configuration file.

The configuration file `config_remote.yml` is provided as an example. For more detailed instruction, please visit [this doc page](https://nni.readthedocs.io/en/stable/TrainingService/RemoteMachineMode.html) in NNI.  

