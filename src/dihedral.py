# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

r"""Dihedral forces.

Dihedral force classes apply a force and virial on every particle in the
simulation state commensurate with the potential energy:

.. math::

    U_\mathrm{dihedral} = \sum_{(i,j,k,l) \in \mathrm{dihedrals}}
    U_{ijkl}(\phi)

Each dihedral is defined by an ordered quadruplet of particle tags in the
`hoomd.State` member ``dihedral_group``. HOOMD-blue does not construct dihedral
groups, users must explicitly define dihedrals in the initial condition.

.. image:: md-dihedral.svg
    :alt: Definition of the dihedral bond between particles i, j, k, and l.

In the dihedral group (i,j,k,l), :math:`\phi` is the signed dihedral angle
between the planes passing through (:math:`\vec{r}_i, \vec{r}_j, \vec{r}_k`) and
(:math:`\vec{r}_j, \vec{r}_k, \vec{r}_l`).

.. rubric Per-particle energies and virials

Dihedral force classes assign 1/4 of the potential energy to each of the
particles in the dihedral group:

.. math::

    U_m = \frac{1}{4} \sum_{(i,j,k,l) \in \mathrm{dihedrals}}
    U_{ijkl}(\phi) [m=i \lor m=j \lor m=k \lor m=l]

and similarly for virials.

Important:
    There are multiple conventions pertaining to the dihedral angle in the
    literature. HOOMD-blue utilizes the convention where :math:`\phi = \pm \pi`
    in the anti-parallel stretched state ( /\\/ ) and :math:`\phi = 0` in the
    parallel compact state ( \|_\| ).
"""

"""Dihedral potentials"""

from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.md import dihedral

import numpy

'''
class Dihedral(Force):
    """Base class dihedral force.

    `Dihedral` is the base class for all dihedral forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    # Module where the C++ class is defined. Reassign this when developing an
    # external plugin.
    _ext_module = _md

    def __init__(self):
        super().__init__()

    def _attach_hook(self):
        # check that some dihedrals are defined
        if self._simulation.state._cpp_sys_def.getDihedralData().getNGlobal(
        ) == 0:
            self._simulation.device._cpp_msg.warning(
                "No dihedrals are defined.\n")

        # create the c++ mirror class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_class = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_class = getattr(self._ext_module, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_class(self._simulation.state._cpp_sys_def)
'''

class Combined_BT(dihedral.Dihedral):
    r"""Combined bending-torsion proper dihedral force.

    `Combined_BT` computes forces, virials, and energies on all dihedrals in the
    simulation state with:

    .. math::

        U(\theta_{i-1},\theta_{i},\phi_{i}) = 
                  k_{\phi} sin^{3}\left(\theta_{i-1}\right) sin^{3}\left(\theta_{i}\right)
                  \[ \sum_{n=0}^{4} a_{n}cos^{n}\left(\phi_{i}\right) = 1 \]

    :math:`a_n` are the force coefficients.

    Attributes:
        params (`TypeParameter` [``dihedral type``, `dict`]):
            The parameter of the Combined_BT dihedrals for each particle type.
            The dictionary has the following keys:

            * ``k_phi`` (`float`, **required**) -  force constant of the
              dihedral :math:`[]`

            * ``a0`` (`float`, **required**) -  force constant of the
              first term :math:`[\mathrm{energy}]`

            * ``a1`` (`float`, **required**) -  force constant of the
              second term :math:`[\mathrm{energy}]`

            * ``a2`` (`float`, **required**) -  force constant of the
              third term :math:`[\mathrm{energy}]`

            * ``a3`` (`float`, **required**) -  force constant of the
              fourth term :math:`[\mathrm{energy}]`

    Examples::

        combined_bt = azplugins.dihedral.Combined_BT()
        opls.params['A-A-A-A'] = dict(k_phi=1.0, a0=1.0, a1=-1.0, a2=1.0, a3=1.0)
    """

    _ext_module = _azplugins
    _cpp_class_name = "Combined_BTDihedralForceCompute"

    def __init__(self):
        super().__init__()
        # check that some dihedrals are defined
        params = TypeParameter(
            'params', 'dihedral_types',
            TypeParameterDict(k_phi=float,
                              a0=float,
                              a1=float,
                              a2=float,
                              a3=float,
                              len_keys=1))
        self._add_typeparam(params)
