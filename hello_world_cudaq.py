import os 
os.environ["CUDAQ_MAX_CPU_MEMORY_GB"] = "NONE"
print("CUDAQ_MAX_CPU_MEMORY_GB:", os.environ.get("CUDAQ_MAX_CPU_MEMORY_GB"))

import cudaq

cudaq.set_target('nvidia', option='mgpu,fp32')

# cudaq.set_target('nvidia')

cudaq.mpi.initialize()

print('mpi initialized?', cudaq.mpi.is_initialized())

n_qubits = 37

@cudaq.kernel
def kernel(n_qubits:int):
    
    qubits = cudaq.qvector(n_qubits)
    
    x(qubits)
    h(qubits)    
    y(qubits)

print('starting calculation')

# expectation_value = cudaq.observe(kernel, cudaq.spin.z(0), n_qubits)
counts = cudaq.sample(kernel, n_qubits)

print("num qubits", n_qubits, ', counts', counts.most_probable())

cudaq.mpi.finalize()