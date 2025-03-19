unset `export | grep UCX | sed 's/=.*//g' | sed 's/.*UCX/UCX/g'`
unset `export | grep MPI | grep -v SLURM | sed 's/=.*//g' | sed 's/.*UCX/UCX/g'`
export PMIX_MCA_gds=hash
export LD_LIBRARY_PATH=/home/cudaq/hpcx-2.21/ucx/mt/lib/:/home/cudaq/hpcx-2.21/ucx/mt/lib/ucx/:$LD_LIBRARY_PATH
export PATH=/home/cudaq/hpcx-2.21/ucx/mt/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export OMPI_MCA_pml=ucx
export OMPI_MCA_opal_cuda_support=true
export QISKIT_FUSION_MAX_QUBITS=5
export UBACKEND_USE_FABRIC_HANDLE=0
export CUDAQ_MGPU_FUSE=5
export CUDAQ_MAX_CPU_MEMORY_GB=480

filtcmd() { awk -W interactive 'BEGIN{f=0}$2=="DeprecationWarning:" || $2=="PendingDeprecationWarning:"{f=2}f==0 && $3!="unclosed"{print $0}f>0{f--}'; }

##NQUBITS=36
##SHORS_NQUBITS=36
NQUBITS=8
SHORS_NQUBITS=8

if [ $PMIX_RANK -lt 1 ]; then
    echo "################################################################"
    echo "################################################################"
    echo "######                   QAOA DEMO                         #####"
    echo "################################################################"
    echo "################################################################"
fi
cuquantum-benchmarks circuit --frontend qiskit --backend cusvaer --benchmark qaoa --nqubits $NQUBITS --nfused 5 --precision single --cusvaer-global-index-bits 1 --cusvaer-p2p-device-bits 0 --ngpus 1 --cusvaer-comm-plugin-type mpi_openmpi --nwarmups 0 --nrepeats 1 --nshots 1000 |& filtcmd

if [ $PMIX_RANK -lt 1 ]; then
    echo "################################################################"
    echo "################################################################"
    echo "######                   QFT DEMO                          #####"
    echo "################################################################"
    echo "################################################################"
fi
cd /workspace/qedc-nvidia/quantum-fourier-transform/qiskit/
cp qft_benchmark.py /lustre/mmodani/SAG/.
python3 -m mpi4py qft_benchmark.py -b cusvaer -s 1000 -n $NQUBITS -c 1 |& filtcmd

if [ $PMIX_RANK -lt 1 ]; then
    echo "################################################################"
    echo "################################################################"
    echo "######                 SHORS DEMO                          #####"
    echo "################################################################"
    echo "################################################################"
fi
cd /workspace/qedc-nvidia/shors/qiskit/
cp shors_benchmark.py /lustre/mmodani/SAG/.
python3 -m mpi4py shors_benchmark.py -b cusvaer -s 1000 -n $SHORS_NQUBITS -c 1 |& filtcmd


cd /workspace
if [ $PMIX_RANK -lt 1 ]; then
    echo "################################################################"
    echo "################################################################"
    echo "######                 NOISE DEMO                          #####"
    echo "################################################################"
    echo "################################################################"
    time python3 cudaq_noise.py $NQUBITS 20
else
    python3 cudaq_noise.py $NQUBITS 20
fi
cp cudaq_noise.py /lustre/mmodani/SAG/.
