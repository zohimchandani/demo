Login 

cd /lustre/zchandani


srun -N 1 --ntasks-per-node 1 -p benchmark-cg1 --job-name SAG --mpi=pmix --network=sharp --pty /bin/bash

enroot import --output cudaQ09.sqsh ‘docker://nvcr.io#nvidia/quantum/cuda-quantum:cu12-0.9.1’
enroot import --output cudaQ10.sqsh ‘docker://nvcr.io#nvidia/quantum/cuda-quantum:cu12-0.10.0’

srun -N 1 --ntasks-per-node 1 -p benchmark-cg1 --job-name SAG --container-image=/lustre/zchandani/demo/cudaQ10.sqsh --container-writable --container-mounts=/dev/infiniband:/dev/infiniband,/lustre/zchandani/demo:/demo --pty /bin/bash

1 node = 1gh 480 + 96 

zchandani@lego-cg1-qs-139:/demo$ export CUDAQ_MAX_CPU_MEMORY_GB=480
