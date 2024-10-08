# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Dihedral potential unit tests."""

import collections

import hoomd
import numpy
import pytest

PotentialTestCase = collections.namedtuple(
    'PotentialTestCase',
    ['potential', 'params', 'r1', 'r2', 'r3', 'r4', 'energy', 'force'],
)

potential_tests = []
# dihedral.BendingTorsion
potential_tests += [
    # test potential when k_phi = 0 
    PotentialTestCase(
        hoomd.azplugins.dihedral.BendingTorsion,
        dict(k_phi=0, a0=1.5,a1=6.2,a2=1.7,a3=3.0,a4=7.2),
        (1.0,0.0,0.0),
        (1.0,0.5,0.0),
        (0.7,0.3,-0.2),
        (0.0,0.4,-0.6),
        (0,0,0,0,0,0),
        (0,0,0,0),
    ),

    # test potential when the first of the angles is 180
    PotentialTestCase(
        hoomd.azplugins.dihedral.BendingTorsion,
        dict(k_phi=0, a0=1.5,a1=6.2,a2=1.7,a3=3.0,a4=7.2),#0, 1.5, 6.2, 1.7, 3.0);
        (0.0,0.0,0.0),
        (0.0,0.5,0.0),
        (0.0,1.0,0.0),
        (0.5,1.4,0.0),
        (0,0,0,0,0,0),
        (0,0,0,0),
    ),
    # test potential when the second of the angles is 180
    PotentialTestCase(
        hoomd.azplugins.dihedral.BendingTorsion,
        dict(k_phi=0, a0=1.5,a1=6.2,a2=1.7,a3=3.0,a4=7.2),#0, 1.5, 6.2, 1.7, 3.0);
        (-0.5,-0.5,0.0),
        (0.0,0.0,0.0),
        (0.0,0.5,0.0),
        (0.0,1.0,0.0),
        (0,0,0,0,0,0),
        (0,0,0,0),
    ),
    # test potential at phi = 0
    PotentialTestCase(
        hoomd.azplugins.dihedral.BendingTorsion,
        dict(k_phi=0, a0=1.5,a1=6.2,a2=1.7,a3=3.0,a4=7.2),#0, 1.5, 6.2, 1.7, 3.0);
        (-0.5,-0.5,0.0),
        (0.0,0.0,0.0),
        (0.0,0.5,0.0),
        (-0.5,1.0,0.0),
        (0,0,0,0,0,0),
        (0,0,0,0),
    ),
    # test potential at phi = 180
    PotentialTestCase(
        hoomd.azplugins.dihedral.BendingTorsion,
        dict(k_phi=0, a0=1.5,a1=6.2,a2=1.7,a3=3.0,a4=7.2),#0, 1.5, 6.2, 1.7, 3.0);
        (-0.5,-0.5,0.0),
        (0.0,0.0,0.0),
        (0.0,0.5,0.0),
        (0.5,1.0,0.0),
        (0,0,0,0,0,0),
        (0,0,0,0),
    ),
    # test potential at phi = 90
    PotentialTestCase(
        hoomd.azplugins.dihedral.BendingTorsion,
        dict(k_phi=0, a0=1.5,a1=6.2,a2=1.7,a3=3.0,a4=7.2),#0, 1.5, 6.2, 1.7, 3.0);
        (-0.5,-0.5,0.0),
        (0.0,0.0,0.0),
        (0.0,0.5,0.0),
        (0.0,1.0,0.5),
        (0,0,0,0,0,0),
        (0,0,0,0),
    ),
]

@pytest.mark.parametrize(
    'potential_test', potential_tests, ids=lambda x: x.potential.__name__
)
def test_energy_and_force(
    simulation_factory, bonded_four_particle_snapshot_factory, potential_test
):
    """Test energy and force evaluation."""
    # make 2 particle test configuration
    sim = simulation_factory(
        bonded_four_particle_snapshot_factory(r1=potential_test.r1,
                                              r2=potential_test.r2,
                                              r3=potential_test.r3,
                                              r4=potential_test.r4)
    )

    # setup dummy NVE integrator
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    integrator.methods = [nve]

    # setup pair potential
    potential = potential_test.potential()
    potential.params['A-A'] = potential_test.params
    integrator.forces = [potential]

    # calculate energies and forces
    sim.operations.integrator = integrator
    sim.run(0)

    # test that parameters are still correct after attach runs
    assert potential.params['A-A'] == potential_test.params

    # test that the energies match reference values, half goes to each particle
    energies = potential.energies
    e = potential_test.energy
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(energies, [0.5 * e, 0.5 * e], decimal=4)

    # test that the forces match reference values, should be directed along x
    forces = potential.forces
    f = potential_test.force
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_almost_equal(
            forces, [[-f, 0, 0], [f, 0, 0]], decimal=4
        )
