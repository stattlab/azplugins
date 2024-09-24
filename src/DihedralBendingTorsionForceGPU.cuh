// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*! \file DihedralBendingTorsionForceGPU.cuh
    \brief Declares GPU kernel code for calculating the combined bending-torsion proper
   dihedral forces. Used by DihedralBendingTorsionForceComputeGPU.
*/

#ifndef AZPLUGINS_DIHEDRAL_BENDING_TORSION_GPU_CUH_
#define AZPLUGINS_DIHEDRAL_BENDING_TORSION_GPU_CUH_

#include "DihedralBendingTorsion.h"
#include "hip/hip_runtime.h"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {
//! Kernel driver that computes combined bending-torsion dihedral forces for
//! DihedralBendingTorsionForceComputeGPU
hipError_t gpu_compute_bending_torsion_dihedral_forces(Scalar4* d_force,
                                                  Scalar* d_virial,
                                                  const size_t virial_pitch,
                                                  const unsigned int N,
                                                  const Scalar4* d_pos,
                                                  const BoxDim& box,
                                                  const group_storage<4>* tlist,
                                                  const unsigned int* dihedral_ABCD,
                                                  const unsigned int pitch,
                                                  const unsigned int* n_dihedrals_list,
                                                  dihedral_bending_torsion_params* d_params,
                                                  const unsigned int n_dihedral_types,
                                                  const int block_size,
                                                  const int warp_size);

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd

#endif
