
# !pip install openfermionpyscf==0.5 matplotlib==3.8.4 scipy==1.13.0 -q

import cudaq
from cudaq import spin 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
np.random.seed(42)

cudaq.set_target("nvidia")

# Number of hydrogen atoms.
hydrogen_count = 10

# Distance between the atoms in Angstroms.
bond_distance = 0.7474

# Define a linear chain of Hydrogen atoms
geometry = [('H', (0, 0, i * bond_distance)) for i in range(hydrogen_count)]

print('preparing hamiltonian')

# hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(geometry, 'sto-3g', 1, 0)
# electron_count = data.n_electrons
# qubit_count = 2 * data.n_orbitals

hamiltonian = spin.x(0)
electron_count = hydrogen_count
qubit_count = 2 * electron_count

print('hamiltonian prepared')


@cudaq.kernel
def kernel(thetas: list[float]):

    qubits = cudaq.qvector(qubit_count)

    for i in range(electron_count):
        x(qubits[i])

    cudaq.kernels.uccsd(qubits, thetas, electron_count, qubit_count)


parameter_count = cudaq.kernels.uccsd_num_parameters(electron_count,
                                                     qubit_count)


# Define a function to minimize
def cost(theta):

    exp_val = cudaq.observe(kernel, hamiltonian, theta).expectation()

    return exp_val


exp_vals = []

def callback(xk):
    vals = cost(xk)
    exp_vals.append(vals)
    print('exp_vals', vals)


# Initial variational parameters.
x0 = np.random.normal(0, np.pi, parameter_count)

# Use the scipy optimizer to minimize the function of interest
result = minimize(cost,
                  x0,
                  method='COBYLA',
                  callback=callback,
                  options={'maxiter': 40})

plt.plot(exp_vals)
plt.xlabel('Epochs')
plt.ylabel('Energy')
plt.title('VQE')
plt.show()


