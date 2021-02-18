import os, sys
import subprocess
import time
# experiments
pwd = 'baoyu'

# tuner
tuners = ['random']
for tuner in tuners:
    for i in range(5):
        cmd0 = 'fuser -k 7007/tcp'
        os.system('echo {}|{}'.format(pwd,cmd0))
        cmd1 = 'nnictl create --config config_{}.yml --port=7007'.format(tuner)
        print(cmd1)
        os.system(cmd1)
        while True:
            time.sleep(100)
            l = 4
            experiment = subprocess.check_output("nnictl experiment list | tail -{} | head -1".format(l),shell=True).decode("utf-8").split('\n')[0]  
            status = experiment.split('Status: ')[1].split(' ')[0]
            port = experiment.split('Port: ')[1].split(' ')[0]
            #status = subprocess.check_output("nnictl experiment list | tail -{} | head -1 | awk '/:/ {{print $6}}'".format(l),shell=True).decode("utf-8").split('\n')[0]            
            #print(status) 
            #port = subprocess.check_output("nnictl experiment list | tail -{} | head -1 | awk '/:/ {{print $8}}'".format(l),shell=True).decode("utf-8").split('\n')[0] 
            #print(port)
            print(l,experiment, port,status)
            while port != '7007':
                l = l+1
                experiment = subprocess.check_output("nnictl experiment list | tail -{} | head -1".format(l),shell=True).decode("utf-8").split('\n')[0]  
                status = experiment.split('Status: ')[1].split(' ')[0]
                port = experiment.split('Port: ')[1].split(' ')[0] 
                #status = subprocess.check_output("nnictl experiment list | tail -{} | head -1 | awk '/:/ {{print $6}}'".format(l),shell=True).decode("utf-8").split('\n')[0]            
                #print(status)
                #port = subprocess.check_output("nnictl experiment list | tail -{} | head -1 | awk '/:/ {{print $8}}'".format(l),shell=True).decode("utf-8").split('\n')[0]
                #print(port)
            print(l,experiment, port,status)
            if status == 'DONE' or status == 'NO_MORE_TRIAL':
                break
        nniid = subprocess.check_output("nnictl experiment list | tail -{} | head -1 | awk '/:/ {{print $2}}'".format(l),shell=True).decode("utf-8").split('\n')[0] 
        cmd2 = 'nnictl experiment export {} --filename=result/cifar10_{}{}.json --type json'.format(nniid, tuner, i)
        os.system(cmd2)

