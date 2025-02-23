# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""SLLOD equations of motion."""

import itertools

import numpy

import hoomd
from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.md.methods import Thermostatted
from hoomd.filter import ParticleFilter
from hoomd.md.methods.thermostats import Thermostat

class MTTKSLLOD(Thermostat):
    r"""The Nosé-Hoover thermostat.

    Controls the system temperature using velocity rescaling with the
    Nosé-Hoover thermostat.

    Args:
        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\mathrm{time}]`

    The translational thermostat has a momentum :math:`\xi` and position
    :math:`\eta`. The rotational thermostat has momentum
    :math:`\xi_{\mathrm{rot}}` and position :math:`\eta_\mathrm{rot}`. Access
    these quantities using `translational_dof` and `rotational_dof`.

    Note:
        The coupling constant `tau` should be set within a
        reasonable range to avoid abrupt fluctuations in the kinetic temperature
        and to avoid long time to equilibration. The recommended value for most
        systems is :math:`\tau = 100 \delta t`.

    See Also:
        `G. J. Martyna, D. J. Tobias, M. L. Klein 1994
        <https://dx.doi.org/10.1063/1.467468>`_ and `J. Cao, G. J. Martyna 1996
        <https://dx.doi.org/10.1063/1.470959>`_.

    .. rubric:: Examples:

    .. code-block:: python

        mttk = hoomd.md.methods.thermostats.MTTK(
            kT=1.5,
            tau=simulation.operations.integrator.dt * 100,
        )
        simulation.operations.integrator.methods[0].thermostat = mttk

    Attributes:
        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\mathrm{energy}]`.

            .. rubric:: Examples:

            .. code-block:: python

                mttk.kT = 1.0

            .. code-block:: python

                mttk.kT = hoomd.variant.Ramp(A=1.0, B=2.0, t_start=0, t_ramp=1_000_000)

        tau (float): Coupling constant for the thermostat
            :math:`[\mathrm{time}]`

            .. rubric:: Example:

            .. code-block:: python

                mttk.tau = 0.2

        translational_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\xi`,
            :math:`\eta`)

            Save and restore the thermostat degrees of freedom when continuing
            simulations:

            .. rubric:: Examples:

            Save before exiting:

            .. code-block:: python

                numpy.save(
                    file=path / "translational_dof.npy",
                    arr=mttk.translational_dof,
                )

            Load when continuing:

            .. code-block:: python

                mttk = hoomd.md.methods.thermostats.MTTK(
                    kT=1.5,
                    tau=simulation.operations.integrator.dt * 100,
                )
                simulation.operations.integrator.methods[0].thermostat = mttk

                mttk.translational_dof = numpy.load(file=path / "translational_dof.npy")


        rotational_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat (:math:`\xi_\mathrm{rot}`,
            :math:`\eta_\mathrm{rot}`)

            Save and restore the thermostat degrees of freedom when continuing
            simulations:

            .. rubric:: Examples:

            Save before exiting:

            .. code-block:: python

                numpy.save(
                    file=path / "rotational_dof.npy",
                    arr=mttk.rotational_dof,
                )

            Load when continuing:

            .. code-block:: python

                mttk = hoomd.md.methods.thermostats.MTTK(
                    kT=1.5,
                    tau=simulation.operations.integrator.dt * 100,
                )
                simulation.operations.integrator.methods[0].thermostat = mttk

                mttk.rotational_dof = numpy.load(file=path / "rotational_dof.npy")
    """

    def __init__(self, kT, tau):
        super().__init__(kT)
        param_dict = ParameterDict(
            tau=float(tau),
            translational_dof=(float, float),
            rotational_dof=(float, float),
        )
        param_dict.update(dict(translational_dof=(0, 0), rotational_dof=(0, 0)))
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = _azplugins.MTTKThermostatSLLOD(
            self.kT, group, self._thermo, self._simulation.state._cpp_sys_def, self.tau
        )

    @hoomd.logging.log(requires_run=True)
    def energy(self):
        """Energy the thermostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`.

        .. rubric:: Example:

        .. code-block:: python

            logger.add(obj=mttk, quantities=['energy'])
        """
        return self._cpp_obj.getThermostatEnergy(self._simulation.timestep)

    def thermalize_dof(self):
        r"""Set the thermostat momenta to random values.

        `thermalize_dof` sets a random value for the momentum :math:`\xi`.
        When `hoomd.md.Integrator.integrate_rotational_dof` is `True`,
        it also sets a random value for the rotational thermostat
        momentum :math:`\xi_{\mathrm{rot}}`. Call `thermalize_dof` to set a new
        random state for the thermostat.

        .. rubric:: Example

        .. code-block:: python

            mttk.thermalize_dof()

        .. important::
            You must call `Simulation.run` before `thermalize_dof`.

        .. seealso:: `State.thermalize_particle_momenta`
        """
        if not self._attached:
            raise RuntimeError(
                "Call Simulation.run(0) before attempting to thermalize the "
                "MTTK thermostat."
            )
        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeThermostat(self._simulation.timestep)



class ConstantVolumeSLLOD(Thermostatted):
    r"""Constant volume dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.

        thermostat (hoomd.md.methods.thermostats.Thermostat): Thermostat to
            control temperature. Setting this to ``None`` samples a constant
            energy (NVE, microcanonical) dynamics. Defaults to ``None``.

    `ConstantVolume` numerically integrates the translational degrees of freedom
    using Velocity-Verlet and the rotational degrees of freedom with a scheme
    based on `Kamberaj 2005`_.

    When set, the `thermostat` rescales the particle velocities to model a
    canonical (NVT) ensemble. Use no thermostat (``thermostat = None``) to
    perform constant energy integration.

    See Also:
        `hoomd.md.methods.thermostats`.

    .. rubric:: Examples:

    NVE integration:

    .. code-block:: python

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        simulation.operations.integrator.methods = [nve]

    NVT integration:

    .. code-block:: python

        nvt = hoomd.md.methods.ConstantVolume(
            filter=hoomd.filter.All(),
            thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5),
        )
        simulation.operations.integrator.methods = [nvt]

    {inherited}

    ----------

    **Members defined in** `ConstantVolume`:

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

    .. _Kamberaj 2005: https://dx.doi.org/10.1063/1.1906216
    """

    __doc__ = __doc__.replace("{inherited}", Thermostatted._doc_inherited)

    _doc_inherited = (
        Thermostatted._doc_inherited
        + """
    ----------

    **Members inherited from**
    `ConstantVolume <hoomd.md.methods.ConstantVolume>`:

    .. py:attribute:: filter

        Subset of particles on which to apply this method.
        `Read more... <hoomd.md.methods.ConstantVolume.filter>`
    """
    )

    def __init__(self, filter, thermostat=None):
        super().__init__()
        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter, thermostat=OnlyTypes(Thermostat, allow_none=True)
        )
        param_dict.update(dict(filter=filter, thermostat=thermostat))
        # set defaults
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        # initialize the reflected cpp class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = _azplugins.TwoStepConstantVolumeSLLOD
            thermo_cls = _azplugins.ComputeThermoSLLOD
        else:
            cls = _azplugins.TwoStepConstantVolumeSLLOD
            thermo_cls = _azplugins.ComputeThermoSLLODGPU

        group = self._simulation.state._get_group(self.filter)
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._thermo = thermo_cls(cpp_sys_def, group)

        if self.thermostat is None:
            self._cpp_obj = cls(cpp_sys_def, group, None)
        else:
            self.thermostat._set_thermo(self.filter, self._thermo)
            self.thermostat._attach(self._simulation)
            self._cpp_obj = cls(cpp_sys_def, group, self.thermostat._cpp_obj)
        super()._attach_hook()

