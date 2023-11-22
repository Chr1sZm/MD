from ase import Atoms
from ase.visualize import view
from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.build import bulk

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

crystalstructure = 'sc'

def calculate_energy(lattice_constant):
    """takes the lattice constant and returns the potential energy of a Cu fcc lattice"""

    cubic_structure = bulk('Cu', crystalstructure, a=lattice_constant)
    emt_potential = EMT()
    cubic_structure.set_calculator(emt_potential)

    # Optimize the structure
    # dyn = BFGS(cubic_structure, trajectory=None)
    # dyn.run(fmax=0.01)

    return cubic_structure.get_potential_energy()


# task 3.2
def main():
    lattice_constants = np.linspace(1., 4., 20)  # range of the lattice constant
    energies = [calculate_energy(lattice_constant) for lattice_constant in lattice_constants]

    coefficients = np.polyfit(lattice_constants, energies, 2)
    polynomial = np.poly1d(coefficients)

    fitted_energies = polynomial(lattice_constants)

    l_min = -coefficients[1] / (2 * coefficients[0])  # l_min = -p_1/(2*p_2)
    min_energy = polynomial(l_min)

    plt.figure(figsize=(3, 3))
    plt.plot(lattice_constants, energies, 'x', color='black', label='Energies')
    plt.plot(lattice_constants, fitted_energies, '--', color='orange', label='2nd order polynomial fit')
    plt.xlabel('Lattice Constant (Å)')
    plt.ylabel('Potential Energy (eV)')
    plt.text(1.45, 47, '$(l_0, W_0)$ = (%.2f~\AA, %.3f~eV)' % (l_min, min_energy))
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fname='energy_lattice_parameter_{}.png'.format(crystalstructure))
    plt.show()


if __name__ == '__main__':
    main()

