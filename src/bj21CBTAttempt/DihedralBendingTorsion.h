// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"

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

    } // namespace azplugins
    } // namespace hoomd
