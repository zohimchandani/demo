import cudaq
cudaq.set_target('nvidia', option='mgpu,fp32')

# cudaq.set_target('nvidia')

# import os 
# os.environ["CUDAQ_MAX_CPU_MEMORY_GB"] = "NONE"
# os.environ["CUDAQ_MAX_GPU_MEMORY_GB"] = "NONE"
# print("CUDAQ_MAX_CPU_MEMORY_GB:", os.environ.get("CUDAQ_MAX_CPU_MEMORY_GB"))
# print("CUDAQ_MAX_GPU_MEMORY_GB:", os.environ.get("CUDAQ_MAX_GPU_MEMORY_GB"))


cudaq.mpi.initialize()

print('mpi initialized?', cudaq.mpi.is_initialized())

n_qubits = 34

@cudaq.kernel
def kernel(n_qubits:int):
    
    qubits = cudaq.qvector(n_qubits)
    
    x(qubits)
    h(qubits)    
    y(qubits)

print('starting exp val calculation')

expectation_value = cudaq.observe(kernel, cudaq.spin.z(0), n_qubits).expectation()

print("num qubits", n_qubits, 'expectation_value', expectation_value)

cudaq.mpi.finalize()