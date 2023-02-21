from qiskit.algorithms import VQE
from qiskit_nature.algorithms import (GroundStateEigensolver,NumPyMinimumEigensolverFactory)
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
import matplotlib.pyplot as plt
import numpy as np
from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit.algorithms.optimizers import SLSQP
from qiskit.opflow import TwoQubitReduction
from qiskit import BasicAer, Aer
from qiskit.utils import QuantumInstance

def qubit_operator(dist):
    # Define the H_2_O Molecule
    mol = Molecule(geometry=[ ["O", [0.0, 0.0, 0.0]] , ["H", [0.0, 0.0, dist]] , ["H", [0.0, dist, dist]] ], multiplicity=1,charge=0)

    # we must use a driver for each distance
    d = ElectronicStructureMoleculeDriver(molecule=mol,basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)

    # Run the driver and get the details
    DETAILS = d.run()
    number_of_particles     =    (DETAILS.get_property("ParticleNumber").num_particles)
    number_of_spin_orbitals = int(DETAILS.get_property("ParticleNumber").num_spin_orbitals)

    # Define the problem at hand, Use an approximation..
    prb = ElectronicStructureProblem( d, [ FreezeCoreTransformer( freeze_core=True, remove_orbitals=None) ] )

    second_quintized_operators = prb.second_q_ops()  # Get 2nd Quantized Operators
    number_of_spin_orbitals = prb.num_spin_orbitals
    number_of_particles     = prb.num_particles

    mapper = ParityMapper()  # Set Mapper
    hamiltonian = second_quintized_operators[0]  # Get Hamiltonian

    # the converter will map the hamiltonian
    converter = QubitConverter(mapper)
    qubit_op = converter.convert(hamiltonian)

    return qubit_op, number_of_particles, number_of_spin_orbitals, prb, converter

# This is the classical solver, exactly the same as qiskit.org
def exact_solver(problem, converter):
    solver = NumPyMinimumEigensolverFactory()
    calc = GroundStateEigensolver(converter, solver)
    result = calc.solve(problem)
    return result


# Lets start 

backend = BasicAer.get_backend("statevector_simulator")
distances = np.arange(0.3, 3.0, 0.2)
#initiate all the energy arrays needed
exact_en = []
vqe_en   = []

# set the classical optimizer
optimizer = SLSQP(maxiter=1000)

for dist in distances:
    (qubit_op, number_of_particles, number_of_spin_orbitals,problem, converter) = qubit_operator(dist)

    result = exact_solver(problem,converter)
    exact_en.append(result.total_energy[0].real)

    # INITIAL STATE -> Hartree Fock State
    init_state = HartreeFock(number_of_spin_orbitals, number_of_particles, converter)
    # For the variational form , lets use the UCCSD
    var_form = UCCSD(converter,number_of_particles,number_of_spin_orbitals,initial_state=init_state)

    # proceed with the calculations of the energy
    vqe = VQE(var_form, optimizer, quantum_instance=backend)
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    vqe_result = problem.interpret(vqe_calc).total_energy[0].real
    vqe_en.append(vqe_result)

    #print the results
    print(f"Interatomic Distance: {np.round(dist, 2)}",
          f"VQE Result: {vqe_result}",
          f"Exact Energy: {exact_en[-1]}")
    
print('All energies have been evaluated.')


# plot all the diagramms
plt.plot(distances, exact_en, label="Exact Energy")
plt.plot(distances, vqe_en, label="VQE Energy")
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.show()