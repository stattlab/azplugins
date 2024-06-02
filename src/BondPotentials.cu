// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file BondPotentials.cu
 * \brief Defines the driver functions for computing bonded forces on the GPU
 *
 * Each bond potential evaluator needs to have an explicit instantiation of the
 * compute_bond_potential.
 */

#include "BondPotentials.cuh"

namespace azplugins
    {
namespace gpu
    {

//! Kernel driver for double well bond potential
template cudaError_t compute_bond_potential<azplugins::detail::BondEvaluatorDoubleWell>(
    const bond_args_t& bond_args,
    const typename azplugins::detail::BondEvaluatorDoubleWell::param_type* d_params,
    unsigned int* d_flags);

    } // end namespace gpu
    } // end namespace azplugins
