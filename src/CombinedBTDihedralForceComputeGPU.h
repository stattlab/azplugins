// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "CombinedBTDihedralForceCompute.h"
#include "CombinedBTDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file CombinedBTDihedralForceComputeGPU.h
    \brief Declares the CombinedBTDihedralForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef AZPLUGINS_COMBINEDBT_DIHEDRAL_FORCE_COMPUTE_GPU_H__
#define AZPLUGINS_COMBINEDBT_DIHEDRAL_FORCE_COMPUTE_GPU_H__

namespace azplugins
    {
//! Computes combined bending-torsion-style dihedral potentials on the GPU
/*! Calculates the combined bending-torsion type dihedral force on the GPU

    The GPU kernel for calculating this can be found in CombinedBTDihedralForceComputeGPU.cu
    \ingroup computes
*/
class PYBIND11_EXPORT CombinedBTDihedralForceComputeGPU : public CombinedBTDihedralForceCompute
    {
    public:
    //! Constructs the compute
    CombinedBTDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~CombinedBTDihedralForceComputeGPU() { }

    private:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size

    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace azplugins

#endif
