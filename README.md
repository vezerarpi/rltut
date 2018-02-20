# rltut

Reinforcement Learning Tutorial.

## Getting started

 - See our [intro presentation](https://docs.google.com/presentation/d/1rJo2nzRS3zimhr6R29wbSg9i5QKd0XYPhU6Ov4sRc6U)
 - Install [Docker](https://www.docker.com/get-docker), Git, Python3
 - Download/clone this repository
 - Use the `run` python script to build & run a notebook server
   - `./run build`
   - `./run notebook`
     - Choose a password for your notebook server
     - Open http://localhost:6767


## Deployment

Running a multi-user server on Azure:

    $ docker run --rm -v ${HOME}:/root -v `pwd`:/work -w /work -it azuresdk/azure-cli-python
    az login
    az account set -s SUBSCRIPTION
    az group create --name rltut --location westeurope
    az configure --defaults group=rltut
    az vm create --name rltut-0 --image UbuntuLTS --ssh-key-value /root/.ssh/id_rsa.pub --admin-username ubuntu --size Standard_D64_v3
    az vm open-port --name rltut-0 --port 80
    az vm extension set --vm-name rltut-0 --name DockerExtension --publisher Microsoft.Azure.Extensions

    # Take note of the IP returned from az vm create:
    ssh ubuntu@IPADDRESS
    git clone https://github.com/vezerarpi/rltut.git
    cd rltut
    ./run build
    ./run prepare alfred betty charlie ...
    ./run start

    # To clean up
    az group delete --name rltut
