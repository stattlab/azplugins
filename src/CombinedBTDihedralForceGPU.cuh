// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*! \file CombinedBTDihedralForceGPU.cuh
    \brief Declares GPU kernel code for calculating the combined bending-torsion proper
   dihedral forces. Used by CombinedBTDihedralForceComputeGPU.
*/

#ifndef AZPLUGINS_COMBINEDBT_DIHEDRAL_GPU_CUH_
#define AZPLUGINS_COMBINEDBT_DIHEDRAL_GPU_CUH_

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

namespace azplugins
    {
namespace gpu
    {
//! Kernel driver that computes combined bending-torsion dihedral forces for CombinedBTDihedralForceComputeGPU
hipError_t gpu_compute_combinedBT_dihedral_forces(Scalar4* d_force,
                                            Scalar* d_virial,
                                            const size_t virial_pitch,
                                            const unsigned int N,
                                            const Scalar4* d_pos,
                                            const BoxDim& box,
                                            const group_storage<4>* tlist,
                                            const unsigned int* dihedral_ABCD,
                                            const unsigned int pitch,
                                            const unsigned int* n_dihedrals_list,
                                            const Scalar4* d_params,
                                            const unsigned int n_dihedral_types,
                                            const int block_size,
                                            const int warp_size);

    } // end namespace gpu
    } // end namespace azplugins

#endif
