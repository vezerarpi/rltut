# rltut

Reinforcement Learning Tutorial (Draft)

## Self-hosting

 - Install Docker
 - Download/clone this repo
 - Use the `run` python script to build & run a notebook server
   - `./run build`
   - `./run notebook`

## Deployment

    $ docker run --rm -v ${HOME}:/root -v `pwd`:/work -w /work -it azuresdk/azure-cli-python
    az login
    az account set -s 1fcc571a-0c5d-4bba-af8c-88239654d71b
    az group create --name rltut --location westeurope
    az configure --defaults group=rltut
    az vm create --name rltut-0 --image UbuntuLTS --ssh-key-value /root/.ssh/id_rsa.pub --admin-username ubuntu --size Standard_D64_v3
    az vm open-port --name rltut-0 --port 80
    az vm extension set --vm-name rltut-0 --name DockerExtension --publisher Microsoft.Azure.Extensions

    # Take note of the IP returned from az vm create:
    ssh ubuntu@IPADDRESS
    docker ps
