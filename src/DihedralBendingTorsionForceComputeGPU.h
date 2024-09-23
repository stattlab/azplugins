// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "DihedralBendingTorsionForceCompute.h"
#include "DihedralBendingTorsionForceGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file DihedralBendingTorsionForceComputeGPU.h
    \brief Declares the DihedralBendingTorsionForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef AZPLUGINS_DIHEDRAL_BENDING_TORSION_FORCE_COMPUTE_GPU_H__
#define AZPLUGINS_DIHEDRAL_BENDING_TORSION_FORCE_COMPUTE_GPU_H__

namespace azplugins
    {
//! Computes combined bending-torsion-style dihedral potentials on the GPU
/*! Calculates the combined bending-torsion type dihedral force on the GPU

    The GPU kernel for calculating this can be found in DihedralBendingTorsionForceComputeGPU.cu
    \ingroup computes
*/
class PYBIND11_EXPORT DihedralBendingTorsionForceComputeGPU : public DihedralBendingTorsionForceCompute
    {
    public:
    //! Constructs the compute
    DihedralBendingTorsionForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~DihedralBendingTorsionForceComputeGPU() { }

    private:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size

    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace azplugins

#endif
