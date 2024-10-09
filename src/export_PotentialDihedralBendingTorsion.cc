// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

// Adapted from hoomd/md/export_PotentialBond.cc.inc of HOOMD-blue.
// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
// #include "hoomd/md/PotentialBond.h"
#include "DihedralBendingTorsionForceCompute.cc"

// #define EVALUATOR_CLASS BondEvaluator@_evaluator@
#define EXPORT_FUNCTION export_DihedralBendingTorsionForceCompute
// clang-format on

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    hoomd::azplugins::detail::export_DihedralBendingTorsionForceCompute(m);
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
