import cudaq
import sys
from mpi4py import MPI

cudaq.set_target('nvidia', option="mgpu,fp32")

# Get the number of qubits to simulate and number of shots
qubit_count = int(sys.argv[1]) if 1 < len(sys.argv) else 2
shots = int(sys.argv[2]) if 2 < len(sys.argv) else 5

# CUDA-Q supports several different models of noise. In this
# case, we will examine the modeling of depolarization noise. This
# depolarization will result in the qubit state decaying into a mix
# of the basis states, |0> and |1>, with a user provided probability.

# We will begin by defining an empty noise model that we will add
# our depolarization channel to.
noise = cudaq.NoiseModel()

# We define a depolarization channel setting the probability
# of the qubit state being scrambled to `1.0`.
depolarization = cudaq.DepolarizationChannel(0.01)

# We will apply the channel to any Y-gate on the qubits. In other words,
# for each Y-gate on our qubits, the qubit will have a `1.0`
# probability of decaying into a mixed state.
noise.add_all_qubit_channel('y', depolarization)

qubit_count = int(sys.argv[1]) if 1 < len(sys.argv) else 2

# Now we define our simple kernel function and allocate
@cudaq.kernel
def kernel():
    qubits = cudaq.qvector(qubit_count)
    # First we apply a Y-gate to the qubits.
    # This will bring all qubits into the |1> state
    # With noise, the state of each qubit will decay
    for i in range(0, qubit_count):
        y(qubits[i])
    mz(qubits)


# Without noise, the qubit should still be in the |1> state.
if MPI.COMM_WORLD.Get_rank() == 0:
    print("Running on",qubit_count,"qubits without noise.",shots,"shots.")
    print("Expected result, all qubits from all",shots,"shots in |1> state")
    print("Actual result:")
counts = cudaq.sample(kernel, shots_count=shots)
if MPI.COMM_WORLD.Get_rank() == 0:
    print(counts)

# With noise, the measurements should be a roughly 50/50
# mix between the |0> and |1> states.
if MPI.COMM_WORLD.Get_rank() == 0:
    print("")
    print("Running on",qubit_count,"qubits with noise.",shots,"shots.")
    print("Expected result, in some samples noise results in qubits in |0>")
noisy_counts = cudaq.sample(kernel, shots_count=shots, noise_model=noise)
if MPI.COMM_WORLD.Get_rank() == 0:
    print(noisy_counts)
