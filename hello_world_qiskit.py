from qiskit import transpile

from qiskit import QuantumCircuit

from cusvaer.backends import StatevectorSimulator

n_qubits = 37

def create_ghz_circuit(n_qubits):
    ghz = QuantumCircuit(n_qubits)
    ghz.h(0)
    for qubit in range(n_qubits - 1):
        ghz.cx(qubit, qubit + 1)
    ghz.measure_all()
    return ghz

circuit = create_ghz_circuit(n_qubits)

print('starting')

simulator = StatevectorSimulator()
simulator.set_options(cusvaer_max_cpu_memory_mb = 457764, precision='single')
circuit = transpile(circuit, simulator)
job = simulator.run(circuit, shots=1024)
result = job.result()

if result.mpi_rank == 0:
    print(result.get_counts())