// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>
#include <vector>

/*! \file DihedralBendingTorsionForceCompute.h
    \brief Declares a class for computing Bending-Torsion dihedrals
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef AZPLUGINS_DIHEDRAL_BENDING_TORSION_FORCE_COMPUTE_H__
#define AZPLUGINS_DIHEDRAL_BENDING_TORSION_FORCE_COMPUTE_H__

namespace hoomd
    {
namespace azplugins
    {
struct dihedral_bending_torsion_params
    {
    Scalar k_phi;
    Scalar a0;
    Scalar a1;
    Scalar a2;
    Scalar a3;
    Scalar a4;

#ifndef __HIPCC__
    dihedral_bending_torsion_params() : k_phi(0.), a0(0.), a1(0.), a2(0.), a3(0.), a4(0.) { }

    dihedral_bending_torsion_params(pybind11::dict v)
        : k_phi(v["k_phi"].cast<Scalar>()), a0(v["a0"].cast<Scalar>()),
            a1(v["a1"].cast<Scalar>()), a2(v["a2"].cast<Scalar>()),
            a3(v["a3"].cast<Scalar>()), a4(v["a4"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k_phi"] = k_phi;
        v["a0"] = a0;
        v["a1"] = a1;
        v["a2"] = a2;
        v["a3"] = a3;
        v["a4"] = a4;
        return v;
        }
#endif
    } __attribute__((aligned(32)));
//! Computes combined bending-torsion proper dihedral forces on each particle
/*! Combined bending-torsion proper dihedral forces are computed on every particle in
        the simulation.

    The dihedrals which forces are computed on are accessed from ParticleData::getDihedralData
    \ingroup computes
*/
class PYBIND11_EXPORT DihedralBendingTorsionForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    DihedralBendingTorsionForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~DihedralBendingTorsionForceCompute();

    //! Set the parameters
    virtual void setParams(std::string type, pybind11::dict params);

    /// Get the parameters for a specified type
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    GPUArray<dihedral_bending_torsion_params> m_params;

    //!< Dihedral data to use in computing dihedrals
    std::shared_ptr<DihedralData> m_dihedral_data;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace azplugins
    } // end namespace hoomd

#endif
