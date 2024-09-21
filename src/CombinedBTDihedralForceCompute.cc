// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "CombinedBTDihedralForceCompute.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! \file CombinedBTDihedralForceCompute.cc
    \brief Contains code for the CombinedBTDihedralForceCompute class
*/

namespace hoomd
    {
namespace azplugins
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
CombinedBTDihedralForceCompute::CombinedBTDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing CombinedBTDihedralForceCompute" << endl;

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();

    // check for some silly errors a user could make
    if (m_dihedral_data->getNTypes() == 0)
        {
        throw runtime_error("No dihedral types in the system.");
        }

    // allocate the parameters
    GPUArray<dihedral_combinedbt_params> params(m_dihedral_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    }

CombinedBTDihedralForceCompute::~CombinedBTDihedralForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CombinedBTDihedralForceCompute" << endl;
    }

/*! \param type Type of the dihedral to set parameters for
    \param k_phi Overall Force parameter in CombinedBT-style dihedral
    \param a0 Force parameter in CombinedBT-style dihedral
    \param a1 Force parameter in CombinedBT-style dihedral
    \param a2 Force parameter in CombinedBT-style dihedral
    \param a3 Force parameter in CombinedBT-style dihedral
    \param a4 Force parameter in CombinedBT-style dihedral

*/
// void CombinedBTDihedralForceCompute::setParams(unsigned int type,
//                                          Scalar k_phi,
//                                          Scalar a0,
//                                          Scalar a1,
//                                          Scalar a2,
//                                          Scalar a3,
//                                          Scalar a4)
//     {
//     // make sure the type is valid
//     if (type >= m_dihedral_data->getNTypes())
//         {
//         throw runtime_error("Invalid dihedral type.");
//         }

//     // set parameters in m_params
//     ArrayHandle<Scalar6> h_params(m_params, access_location::host, access_mode::readwrite);
//     h_params.data[type] = make_scalar6(k_phi, a0, a1, a2, a3, a4);
//     }

void CombinedBTDihedralForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    // make sure the type is valid
    auto typ = m_dihedral_data->getTypeByName(type);
    dihedral_combinedbt_params _params(params);
    // setParams(typ, _params.k_phi, _params.a0, _params.a1, _params.a2, _params.a3, _params.a4);
    ArrayHandle<dihedral_combinedbt_params> h_params(m_params,
                                                   access_location::host,
                                                   access_mode::readwrite);
    h_params.data[typ] = _params;
    }

pybind11::dict CombinedBTDihedralForceCompute::getParams(std::string type)
    {
    auto typ = m_dihedral_data->getTypeByName(type);
    // // make sure the type is valid
    // if (typ >= m_dihedral_data->getNTypes())
    //     {
    //     throw runtime_error("Invalid dihedral type.");
    //     }
    // ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);
    // auto val = h_params.data[typ];
    pybind11::dict params;
    // // note: the values stored in params are precomputed k/2 values
    // params["k1"] = val.x * 2;
    // params["k2"] = val.y * 2;
    // params["k3"] = val.z * 2;
    // params["k4"] = val.w * 2;
    ArrayHandle<dihedral_combinedbt_params> h_params(m_params,
                                                   access_location::host,
                                                   access_mode::read);
    // return params;
    return h_params.data[typ].asDict();
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void CombinedBTDihedralForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // access the force and virial tensor arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // access parameter data
    ArrayHandle<dihedral_combinedbt_params> h_params(m_params, access_location::host,
                                            access_mode::read);

    // Zero data for force calculation before computation
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // there are enough other checks on the input data, but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    size_t virial_pitch = m_virial.getPitch();

    // From LAMMPS OPLS dihedral implementation
    unsigned int i1, i2, i3, i4, n, dihedral_type;
    Scalar3 vb1, vb2, vb3, vb2m;

    // this volatile is not strictly needed, but it works around a compiler bug on Mac arm64
    // with Apple clang version 13.0.0 (clang-1300.0.29.30)
    // without the volatile, the x component of f2 is always computed the same as the y component
    volatile Scalar4 f1, f2, f3, f4;
    Scalar ax, ay, az, bx, by, bz, rasq, rbsq, rgsq, rg, rginv, ra2inv, rb2inv, rabinv;
    Scalar df, df1, ddf1, fg, hg, fga, hgb, gaa, gbb;
    Scalar dtfx, dtfy, dtfz, dtgx, dtgy, dtgz, dthx, dthy, dthz;
    Scalar c, s, p, sx2, sy2, sz2, cos_term, e_dihedral;
    Scalar k_phi, a0, a1, a2, a3, a4;
    Scalar dihedral_virial[6];
    Scalar angle_012_virial[6];
    Scalar angle_123_virial[6];

    // get a local copy of the simulation box
    const BoxDim& box = m_pdata->getBox();

    // iterate through each dihedral
    const unsigned int numDihedrals = (unsigned int)m_dihedral_data->getN();
    for (n = 0; n < numDihedrals; n++)
        {
        // lookup the tag of each of the particles participating in the dihedral
        const ImproperData::members_t& dihedral = m_dihedral_data->getMembersByIndex(n);
        assert(dihedral.tag[0] < m_pdata->getNGlobal());
        assert(dihedral.tag[1] < m_pdata->getNGlobal());
        assert(dihedral.tag[2] < m_pdata->getNGlobal());
        assert(dihedral.tag[3] < m_pdata->getNGlobal());

        // i1 to i4 are the tags
        i1 = h_rtag.data[dihedral.tag[0]];
        i2 = h_rtag.data[dihedral.tag[1]];
        i3 = h_rtag.data[dihedral.tag[2]];
        i4 = h_rtag.data[dihedral.tag[3]];

        // throw an error if this angle is incomplete
        if (i1 == NOT_LOCAL || i2 == NOT_LOCAL || i3 == NOT_LOCAL || i4 == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "dihedral.combinedbt: dihedral " << dihedral.tag[0] << " " << dihedral.tag[1] << " "
                << dihedral.tag[2] << " " << dihedral.tag[3] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in dihedral calculation");
            }

        assert(i1 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i2 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i3 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i4 < m_pdata->getN() + m_pdata->getNGhosts());

        // 1st bond

        vb1.x = h_pos.data[i1].x - h_pos.data[i2].x;
        vb1.y = h_pos.data[i1].y - h_pos.data[i2].y;
        vb1.z = h_pos.data[i1].z - h_pos.data[i2].z;

        // 2nd bond

        vb2.x = h_pos.data[i3].x - h_pos.data[i2].x;
        vb2.y = h_pos.data[i3].y - h_pos.data[i2].y;
        vb2.z = h_pos.data[i3].z - h_pos.data[i2].z;

        // 3rd bond

        vb3.x = h_pos.data[i4].x - h_pos.data[i3].x;
        vb3.y = h_pos.data[i4].y - h_pos.data[i3].y;
        vb3.z = h_pos.data[i4].z - h_pos.data[i3].z;

        // apply periodic boundary conditions
        vb1 = box.minImage(vb1);
        vb2 = box.minImage(vb2);
        vb3 = box.minImage(vb3);

        vb2m.x = -vb2.x;
        vb2m.y = -vb2.y;
        vb2m.z = -vb2.z;
        vb2m = box.minImage(vb2m);

        // c,s calculation

        ax = vb1.y * vb2m.z - vb1.z * vb2m.y;
        ay = vb1.z * vb2m.x - vb1.x * vb2m.z;
        az = vb1.x * vb2m.y - vb1.y * vb2m.x;
        bx = vb3.y * vb2m.z - vb3.z * vb2m.y;
        by = vb3.z * vb2m.x - vb3.x * vb2m.z;
        bz = vb3.x * vb2m.y - vb3.y * vb2m.x;

        rasq = ax * ax + ay * ay + az * az;
        rbsq = bx * bx + by * by + bz * bz;
        rgsq = vb2m.x * vb2m.x + vb2m.y * vb2m.y + vb2m.z * vb2m.z;
        rg = sqrt(rgsq);

        rginv = ra2inv = rb2inv = 0.0;
        if (rg > 0)
            rginv = 1.0 / rg;
        if (rasq > 0)
            ra2inv = 1.0 / rasq;
        if (rbsq > 0)
            rb2inv = 1.0 / rbsq;
        rabinv = sqrt(ra2inv * rb2inv);

        c = (ax * bx + ay * by + az * bz) * rabinv;
        s = rg * rabinv * (ax * vb3.x + ay * vb3.y + az * vb3.z);

        if (c > 1.0)
            c = 1.0;
        if (c < -1.0)
            c = -1.0;

        // get values for k1/2 through k4/2
        // ----- The 1/2 factor is already stored in the parameters --------
        dihedral_type = m_dihedral_data->getTypeByIndex(n);
        k_phi = h_params.data[dihedral_type].k_phi;
        a0 = h_params.data[dihedral_type].a0;
        a1 = h_params.data[dihedral_type].a1;
        a2 = h_params.data[dihedral_type].a2;
        a3 = h_params.data[dihedral_type].a3;
        a4 = h_params.data[dihedral_type].a4;


        /**

        fg = vb1.x * vb2m.x + vb1.y * vb2m.y + vb1.z * vb2m.z;
        hg = vb3.x * vb2m.x + vb3.y * vb2m.y + vb3.z * vb2m.z;
        fga = fg * ra2inv * rginv;
        hgb = hg * rb2inv * rginv;
        gaa = -ra2inv * rg;
        gbb = rb2inv * rg;

        dtfx = gaa * ax;
        dtfy = gaa * ay;
        dtfz = gaa * az;
        dtgx = fga * ax - hgb * bx;
        dtgy = fga * ay - hgb * by;
        dtgz = fga * az - hgb * bz;
        dthx = gbb * bx;
        dthy = gbb * by;
        dthz = gbb * bz;

        sx2 = df * dtgx;
        sy2 = df * dtgy;
        sz2 = df * dtgz;

        f1.x = df * dtfx;
        f1.y = df * dtfy;
        f1.z = df * dtfz;
        f1.w = e_dihedral;

        f2.x = sx2 - f1.x;
        f2.y = sy2 - f1.y;
        f2.z = sz2 - f1.z;
        f2.w = e_dihedral;

        f4.x = df * dthx;
        f4.y = df * dthy;
        f4.z = df * dthz;
        f4.w = e_dihedral;

        f3.x = -sx2 - f4.x;
        f3.y = -sy2 - f4.y;
        f3.z = -sz2 - f4.z;
        f3.w = e_dihedral;

        // Apply force to each of the 4 atoms
        h_force.data[i1].x = h_force.data[i1].x + f1.x;
        h_force.data[i1].y = h_force.data[i1].y + f1.y;
        h_force.data[i1].z = h_force.data[i1].z + f1.z;
        h_force.data[i1].w = h_force.data[i1].w + f1.w;
        h_force.data[i2].x = h_force.data[i2].x + f2.x;
        h_force.data[i2].y = h_force.data[i2].y + f2.y;
        h_force.data[i2].z = h_force.data[i2].z + f2.z;
        h_force.data[i2].w = h_force.data[i2].w + f2.w;
        h_force.data[i3].x = h_force.data[i3].x + f3.x;
        h_force.data[i3].y = h_force.data[i3].y + f3.y;
        h_force.data[i3].z = h_force.data[i3].z + f3.z;
        h_force.data[i3].w = h_force.data[i3].w + f3.w;
        h_force.data[i4].x = h_force.data[i4].x + f4.x;
        h_force.data[i4].y = h_force.data[i4].y + f4.y;
        h_force.data[i4].z = h_force.data[i4].z + f4.z;
        h_force.data[i4].w = h_force.data[i4].w + f4.w;

        // Compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // upper triangular version of virial tensor
        dihedral_virial[0] = 0.25 * (vb1.x * f1.x + vb2.x * f3.x + (vb3.x + vb2.x) * f4.x);
        dihedral_virial[1] = 0.25 * (vb1.y * f1.x + vb2.y * f3.x + (vb3.y + vb2.y) * f4.x);
        dihedral_virial[2] = 0.25 * (vb1.z * f1.x + vb2.z * f3.x + (vb3.z + vb2.z) * f4.x);
        dihedral_virial[3] = 0.25 * (vb1.y * f1.y + vb2.y * f3.y + (vb3.y + vb2.y) * f4.y);
        dihedral_virial[4] = 0.25 * (vb1.z * f1.y + vb2.z * f3.y + (vb3.z + vb2.z) * f4.y);
        dihedral_virial[5] = 0.25 * (vb1.z * f1.z + vb2.z * f3.z + (vb3.z + vb2.z) * f4.z);

        for (int k = 0; k < 6; k++)
            {
            h_virial.data[virial_pitch * k + i1] += dihedral_virial[k];
            h_virial.data[virial_pitch * k + i2] += dihedral_virial[k];
            h_virial.data[virial_pitch * k + i3] += dihedral_virial[k];
            h_virial.data[virial_pitch * k + i4] += dihedral_virial[k];
            }
        }
        */
        }
    }

namespace detail
    {
void export_CombinedDihedralForceCompute(pybind11::module& m)
    {
    pybind11::class_<CombinedBTDihedralForceCompute,
                     ForceCompute,
                     std::shared_ptr<CombinedBTDihedralForceCompute>>(m,
                                                    "CombinedBTDihedralForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &CombinedBTDihedralForceCompute::setParamsPython)
        .def("getParams", &CombinedBTDihedralForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
