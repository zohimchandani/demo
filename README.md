Login via ssh 

cd /lustre/zchandani

docker pull nvcr.io/nvidia/cuquantum-appliance:24.03-x86_64

Start an interactive job without a container allocating -N nodes 

srun -N 1 --ntasks-per-node 1 -p benchmark-cg1 --job-name SAG --mpi=pmix --network=sharp --pty /bin/bash

enroot import --output cudaQ10.sqsh 'docker://nvcr.io#nvidia/quantum/cuda-quantum:cu12-0.10.0'

enroot import --output cuq.sqsh 'docker://nvcr.io#nvidia/cuquantum-appliance:24.11.0-x86_64'



exit

Launch a containerized job on -N nodes 

srun -N 4 --ntasks-per-node 1 -p benchmark-cg1 --job-name SAG --container-image=/lustre/zchandani/cudaQ10.sqsh --container-writable --container-mounts=/dev/infiniband:/dev/infiniband,/lustre/zchandani/demo:/demo /bin/bash /demo/hello_world.sh

srun -N 4 --ntasks-per-node 1 -p benchmark-cg1 --job-name SAG --container-image=/lustre/zchandani/qiskit-25.sqsh --container-writable --container-mounts=/dev/infiniband:/dev/infiniband,/lustre/zchandani/demo:/demo /bin/bash /demo/hello_world.sh

python3 /demo/hello_world.py

1 node = 1 GH 480 + 96 = 576 GB 

set env variables in terminal:  export CUDAQ_MAX_CPU_MEMORY_GB=480

filtcmd() { awk -W interactive 'BEGIN{f=0}$2=="DeprecationWarning:" || $2=="PendingDeprecationWarning:"{f=2}f==0 && $3!="unclosed"{print $0}f>0{f--}'; } - ignore any deprecation warnings 


squeue
             59741 benchmark      SAG zchandan  R       3:30      2 lego-cg1-qs-[146-147]

in head node on ganon: ssh zchandani@lego-cg1-qs-146


comment out export LD_LIBRARY_PATH=/home/cudaq/hpcx-2.21/ucx/mt/lib/:/home/cudaq/hpcx-2.21/ucx/mt/lib/ucx/:$LD_LIBRARY_PATH for cudaq 


srun -N 4 --ntasks-per-node 1 -p benchmark-cg1 --job-name SAG --container-image=/lustre/zchandani/qiskit-25.sqsh --container-writable --container-mounts=/dev/infiniband:/dev/infiniband,/lustre/zchandani/demo:/demo /bin/bash /demo/hello_world.sh