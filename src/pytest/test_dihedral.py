# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Dihedral potential unit tests."""

import collections

import hoomd
import numpy
import pytest

# dihedral.BendingTorsion
import itertools
# Test parameters include the class, dict of class keyword arguments, dict of 
# bond params, system theta in degrees, system phi in degrees, force array, and energy.
dihedral_test_parameters = [
    (
        hoomd.azplugins.dihedral.BendingTorsion,
        dict(),
        dict(k_phi=10.0, a0=2.41, a1=-2.95, a2=0.36, a3=1.33, a4=0.0),
        45,
        90,
        [[0,1.206551022, -2.91287184],
         [0,-1.206551022,2.91287184],
         [0,-1.206551022,-2.91287184],
         [0,1.206551022, 2.91287184],],
        9.74261005,
    ),
    (
        hoomd.azplugins.dihedral.BendingTorsion,
        dict(),
        dict(k_phi=1.0, a0=1, a1=0.0, a2=0.0, a3=0.0, a4=0.0),
        0,
        135,
        [[-0.265165043,-0.265165043,0.0],
         [0.265165043,0.265165043,0.0],
         [-0.265165043,0.265165043,0.0],
         [0.265165043,-0.265165043,0.0],],
        0.125,
    ),
]

@pytest.fixture(scope='session')
def dihedral_snapshot_factory(device):#, phi_deg=45, theta_deg=90):

    def make_snapshot(d=1.0, phi_deg=45, theta_deg=90, particle_types=['A'], L=20):
        phi_rad = phi_deg * (numpy.pi / 180)
        # the central particles are along the x-axis, so phi is determined from
        # the angle in the yz plane.
        theta_rad = theta_deg * (numpy.pi / 180)

        snapshot = hoomd.Snapshot(device.communicator)
        N = 4
        if snapshot.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            snapshot.configuration.box = box
            snapshot.particles.N = N
            snapshot.particles.types = particle_types
            # shift particle positions slightly in z so MPI tests pass
            snapshot.particles.position[:] = [
                [
                    d*numpy.sin(-(theta_rad-numpy.pi/2)),
                    d*numpy.cos(-(theta_rad-numpy.pi/2))*numpy.cos(phi_rad/2),
                    d*numpy.cos(-(theta_rad-numpy.pi/2))*numpy.sin(phi_rad/2) + 0.1,
                ],
                [0.0, 0.0, 0.1],
                [d, 0.0, 0.1],
                [   
                    d + d*numpy.sin(theta_rad-numpy.pi/2),
                    d*numpy.cos(theta_rad-numpy.pi/2)*numpy.cos(phi_rad/2),
                    -d*numpy.cos(theta_rad-numpy.pi/2)*numpy.sin(phi_rad/2) + 0.1,
                ],
            ]
            print(snapshot.particles.position[:])
            print(theta_deg)
            print(phi_deg)

            snapshot.dihedrals.N = 1
            snapshot.dihedrals.types = ['A-A-A-A']
            snapshot.dihedrals.typeid[0] = 0
            snapshot.dihedrals.group[0] = (0, 1, 2, 3)

        return snapshot

    return make_snapshot#(phi_deg=phi_deg, theta_deg=theta_deg)


@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, phi, theta, force_array, energy',
                         dihedral_test_parameters)
def test_before_attaching(dihedral_cls, dihedral_args, params, phi, theta, force_array, energy):
    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params
    for key in params:
        potential.params['A-A-A-A'][key] == pytest.approx(params[key])


@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, phi, theta, force_array, energy',
                         dihedral_test_parameters)
def test_after_attaching(dihedral_snapshot_factory, simulation_factory,
                         dihedral_cls, dihedral_args, params, phi, theta, force_array, energy):
    phi_deg = phi
    phi_rad = phi * numpy.pi/180
    theta_deg = theta
    theta_rad = theta * numpy.pi/180
    snapshot = dihedral_snapshot_factory(d=0.969, L=5)
    sim = simulation_factory(snapshot)

    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)
    for key in params:
        assert potential.params['A-A-A-A'][key] == pytest.approx(params[key])


@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, phi, theta, force_array, energy',
                         dihedral_test_parameters)
def test_forces_and_energies(dihedral_snapshot_factory, simulation_factory,
                             dihedral_cls, dihedral_args, params, phi, theta, force_array,
                             energy):
    phi_deg = phi
    phi_rad = phi_deg * (numpy.pi / 180)
    theta_deg = theta
    theta_rad = theta_deg * (numpy.pi / 180)
    snapshot = dihedral_snapshot_factory(phi_deg=phi_deg, theta_deg=theta_deg)
    sim = simulation_factory(snapshot)

    # the dihedral angle is in yz plane, thus no force along x axis
    # force_array = force * numpy.asarray(
    #     [0, numpy.sin(-phi_rad / 2),
    #      numpy.cos(-phi_rad / 2)])
    force_array = numpy.asarray(force_array)
    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)

    sim_energies = potential.energies
    sim_forces = potential.forces
    print(sim_forces)
    print(force_array)
    if sim.device.communicator.rank == 0:
        assert sum(sim_energies) == pytest.approx(energy, rel=1e-2, abs=1e-5)
        numpy.testing.assert_allclose(sim_forces[0],
                                      force_array[0],
                                      rtol=1e-2,
                                      atol=1e-5)
        numpy.testing.assert_allclose(sim_forces[1],
                                      force_array[1],
                                      rtol=1e-2,
                                      atol=1e-5)
        numpy.testing.assert_allclose(sim_forces[2],
                                      force_array[2],
                                      rtol=1e-2,
                                      atol=1e-5)
        numpy.testing.assert_allclose(sim_forces[3],
                                      force_array[3],
                                      rtol=1e-2,
                                      atol=1e-5)


@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, phi, theta, force_array, energy',
                         dihedral_test_parameters)
def test_kernel_parameters(dihedral_snapshot_factory, simulation_factory,
                           dihedral_cls, dihedral_args, params, phi, theta, force_array, energy):
    phi_deg = 45
    phi_deg = phi
    phi_rad = phi * numpy.pi/180
    theta_deg = theta
    theta_rad = theta * numpy.pi/180
    snapshot = dihedral_snapshot_factory(phi_deg=phi_deg)
    sim = simulation_factory(snapshot)

    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)

    hoomd.conftest.autotuned_kernel_parameter_check(instance=potential,
                                     activate=lambda: sim.run(1))


# # Test Logging
# @pytest.mark.parametrize(
#     'cls, expected_namespace, expected_loggables',
#     zip((hoomd.md.dihedral.Dihedral, hoomd.md.dihedral.Periodic, hoomd.md.dihedral.Table,
#          hoomd.md.dihedral.OPLS), itertools.repeat(('hoomd.md', 'dihedral')),
#         itertools.repeat(hoomd.conftest.expected_loggable_params)))
# def test_logging(cls, expected_namespace, expected_loggables):
#     hoomd.conftest.logging_check(cls, expected_namespace, expected_loggables)


# Test Pickling
@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, phi, theta, force_array, energy',
                         dihedral_test_parameters)
def test_pickling(simulation_factory, dihedral_snapshot_factory, dihedral_cls,
                  dihedral_args, params, phi, theta, force_array, energy):
    phi_deg = 45
    phi_deg = phi
    phi_rad = phi * numpy.pi/180
    theta_deg = theta
    theta_rad = theta * numpy.pi/180
    snapshot = dihedral_snapshot_factory(phi_deg=phi_deg)
    sim = simulation_factory(snapshot)
    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params

    hoomd.conftest.pickling_check(potential)
    integrator = hoomd.md.Integrator(0.05, forces=[potential])
    sim.operations.integrator = integrator
    sim.run(0)
    hoomd.conftest.pickling_check(potential)

# @pytest.mark.parametrize(
#     'potential_test', potential_tests, ids=lambda x: x.potential.__name__
# )
# def test_energy_and_force(
#     simulation_factory, bonded_four_particle_snapshot_factory, potential_test
# ):
#     """Test energy and force evaluation."""
#     # make 2 particle test configuration
#     sim = simulation_factory(
#         bonded_four_particle_snapshot_factory(r1=potential_test.r1,
#                                               r2=potential_test.r2,
#                                               r3=potential_test.r3,
#                                               r4=potential_test.r4)
#     )

#     # setup dummy NVE integrator
#     integrator = hoomd.md.Integrator(dt=0.001)
#     nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
#     integrator.methods = [nve]

#     # setup pair potential
#     potential = potential_test.potential()
#     potential.params['A-A'] = potential_test.params
#     integrator.forces = [potential]

#     # calculate energies and forces
#     sim.operations.integrator = integrator
#     sim.run(0)

#     # test that parameters are still correct after attach runs
#     assert potential.params['A-A'] == potential_test.params

#     # test that the energies match reference values, half goes to each particle
#     energies = potential.energies
#     e = potential_test.energy
#     if sim.device.communicator.rank == 0:
#         numpy.testing.assert_array_almost_equal(energies, [0.5 * e, 0.5 * e], decimal=4)

#     # test that the forces match reference values, should be directed along x
#     forces = potential.forces
#     f = potential_test.force
#     if sim.device.communicator.rank == 0:
#         numpy.testing.assert_array_almost_equal(
#             forces, [[-f, 0, 0], [f, 0, 0]], decimal=4
#         )
