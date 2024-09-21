// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"

namespace hoomd
    {
namespace azplugins
    {
namespace md
    {
struct combined_bt_params
    {
    Scalar k_phi;
    Scalar a0;
    Scalar a1;
    Scalar a2;
    Scalar a3;
    Scalar a4;

#ifndef __HIPCC__
    combined_bt_params() : k_phi(0.), a0(0.), a1(0.), a2(0.), a3(0.), a4(0.) { }

    combined_bt_params(pybind11::dict v)
        : k_phi(v["k_phi"].cast<Scalar>()), a0(v["a0"].cast<Scalar>()),
          a1(v["a1"].cast<Scalar>()), a2(v["a2"].cast<Scalar>()),
          a3(v["a3"].cast<Scalar>()), a4(v["a0"].cast<Scalar>()) {};
        // {
        // if (k <= 0)
        //     {
        //     throw std::runtime_error("Periodic improper K must be greater than 0.");
        //     }
        // if (d != 1 && d != -1)
        //     {
        //     throw std::runtime_error("Periodic improper d must be -1 or 1.");
        //     }
        // if (chi_0 < 0 || chi_0 >= Scalar(2 * M_PI))
        //     {
        //     throw std::runtime_error("Periodic improper chi_0 must be in the range [0, 2pi).");
        //     }
        // }

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

    } // namespace md
    } // namespace azplugins
    } // namespace hoomd
