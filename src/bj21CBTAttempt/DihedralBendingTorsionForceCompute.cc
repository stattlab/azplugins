// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "DihedralBendingTorsionForceCompute.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! \file DihedralBendingTorsionForceCompute.cc
    \brief Contains code for the DihedralBendingTorsionForceCompute class
*/

namespace hoomd
    {
namespace azplugins
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
DihedralBendingTorsionForceCompute::DihedralBendingTorsionForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing DihedralBendingTorsionForceCompute" << endl;

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();

    // check for some silly errors a user could make
    if (m_dihedral_data->getNTypes() == 0)
        {
        throw runtime_error("No dihedral types in the system.");
        }

    // allocate the parameters
    GPUArray<dihedral_bending_torsion_params> params(m_dihedral_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    }

DihedralBendingTorsionForceCompute::~DihedralBendingTorsionForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying DihedralBendingTorsionForceCompute" << endl;
    }

/*! \param type Type of the dihedral to set parameters for
    \param k_phi Overall Force parameter in Bending-Torsion-style dihedral
    \param a0 Force parameter in Bending-Torsion-style dihedral
    \param a1 Force parameter in Bending-Torsion-style dihedral
    \param a2 Force parameter in Bending-Torsion-style dihedral
    \param a3 Force parameter in Bending-Torsion-style dihedral
    \param a4 Force parameter in Bending-Torsion-style dihedral

*/
void DihedralBendingTorsionForceCompute::setParams(std::string type, pybind11::dict params)
    {
    // make sure the type is valid
    auto typ = m_dihedral_data->getTypeByName(type);
    dihedral_bending_torsion_params _params(params);
    ArrayHandle<dihedral_bending_torsion_params> h_params(m_params,
                                                   access_location::host,
                                                   access_mode::readwrite);
    h_params.data[typ] = _params;
    }

pybind11::dict DihedralBendingTorsionForceCompute::getParams(std::string type)
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
    ArrayHandle<dihedral_bending_torsion_params> h_params(m_params,
                                                   access_location::host,
                                                   access_mode::read);
    // return params;
    return h_params.data[typ].asDict();
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void DihedralBendingTorsionForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // access the force and virial tensor arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // access parameter data
    ArrayHandle<dihedral_bending_torsion_params> h_params(m_params, access_location::host, 
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
    volatile Scalar4 F0, F1, F2, F3;
    Scalar C11, C12, C22, C23, C33, C13, D12, D23;
    Scalar3 dcostheta012_dr0, dcostheta012_dr1, dcostheta012_dr2;
    Scalar3 dcostheta123_dr1, dcostheta123_dr2, dcostheta123_dr3;
    Scalar3 dcosphi0123_dr0, dcosphi0123_dr1, dcosphi0123_dr2, dcosphi0123_dr3;
    Scalar3 F0_theta_012, F1_theta_012, F2_theta_012, F3_theta_012;
    Scalar3 F0_theta_123, F1_theta_123, F2_theta_123, F3_theta_123;
    Scalar3 F0_phi_0123, F1_phi_0123, F2_phi_0123, F3_phi_0123;
    Scalar dV_dtheta012, dV_dtheta123, dV_dcosphi0123, dtheta012_dcostheta012, dtheta123_dcostheta123;
    Scalar e_dihedral;
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
                << "dihedral.bendingtorsion: dihedral " << dihedral.tag[0] << " " << dihedral.tag[1] << " "
                << dihedral.tag[2] << " " << dihedral.tag[3] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in dihedral calculation");
            }

        assert(i1 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i2 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i3 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i4 < m_pdata->getN() + m_pdata->getNGhosts());

        // get the dihedral parameters according to the type
        dihedral_type = m_dihedral_data->getTypeByIndex(n);
        const dihedral_bending_torsion_params& param = h_params.data[dihedral_type];
        k_phi = param.k_phi;
        a0 = param.a0;
        a1 = param.a1;
        a2 = param.a2;
        a3 = param.a3;
        a4 = param.a4;

        /**        
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
         */

        // 1st bond
        vb1.x = h_pos.data[i2].x - h_pos.data[i1].x;
        vb1.y = h_pos.data[i2].y - h_pos.data[i1].y;
        vb1.z = h_pos.data[i2].z - h_pos.data[i1].z;
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

        vb1m.x = -vb1.x;
        vb1m.y = -vb1.y;
        vb1m.z = -vb1.z;
        vb1m = box.minImage(vb1m);

        vb2m.x = -vb2.x;
        vb2m.y = -vb2.y;
        vb2m.z = -vb2.z;
        vb2m = box.minImage(vb2m);

        // simplifiers
        C11 = vb1.x * vb1.x + vb1.y * vb1.y + vb1.z * vb1.z;
        C12 = vb1.x * vb2.x + vb1.y * vb2.y + vb1.z * vb2.z;
        C22 = vb2.x * vb2.x + vb2.y * vb2.y + vb2.z * vb2.z;
        C23 = vb2.x * vb3.x + vb2.y * vb3.y + vb2.z * vb3.z;
        C33 = vb3.x * vb3.x + vb3.y * vb3.y + vb3.z * vb3.z;
        C13 = vb1.x * vb3.x + vb1.y * vb3.y + vb1.z * vb3.z;
        D12 = C11*C22-C12*C12;
        D23 = C22*C33-C23*C23;
        // to speed up computation
        Scalar root_D12D23_inv = 1/sqrt(D12*D23);
        Scalar root_C11C22_inv = 1/sqrt(C11*C22);
        Scalar root_C22C33_inv = 1/sqrt(C22*C33);

        // cosines of angles
        Scalar cos_theta_012 = C12/sqrt(C22*C11); //a=2
        Scalar cos_theta_123 = C23/sqrt(C33*C22); //a=3
        Scalar theta_012 = acos(cos_theta_012);
        Scalar theta_123 = acos(cos_theta_123);
        Scalar sin_theta_012 = sin(theta_012);
        Scalar sin_theta_123 = sin(theta_123);
        // cosine of dihedral
        // cos_phi_0123 = -(C23*C12-C13*C22)/sqrt(D23*D12);  //a = 3
        Scalar cos_phi_0123 = -(C23*C12-C13*C22)*root_D12D23_inv;  //a = 3

        // common partial derivatives
        Scalar prefactor = k_phi*sin_theta_012*sin_theta_012*sin_theta_012*sin_theta_123*sin_theta_123*sin_theta_123;
        Scalar torsion_e = a0 + a1*cos_phi_0123 + a2*cos_phi_0123*cos_phi_0123 + a3*cos_phi_0123*cos_phi_0123*cos_phi_0123 + a4*cos_phi_0123*cos_phi_0123*cos_phi_0123*cos_phi_0123;
        dV_dtheta012 = prefactor*3*cos_theta_012/sin_theta_012*torsion_e;
        dV_dtheta123 = prefactor*3*cos_theta_123/sin_theta_123*torsion_e;
        dV_dcosphi0123 = prefactor*(torsion_e-a0)/cos_phi_0123;

        dtheta012_dcostheta012 = -1/sin_theta_012;
        dtheta123_dcostheta123 = -1/sin_theta_123;

        //Force on particle 0
        //angle. a = 0, acting on theta_a+2
        dcostheta012_dr0 = root_C11C22_inv*((C12/C11)*vb1-vb2);
        F0_theta_012 = -dV_dtheta012*dtheta012_dcostheta012*dcostheta012_dr0;
        F0_theta_123 = make_scalar3(0,0,0);
        //dihedral. a = 0, acting on phi_a+3
        // dcosphi0123_dr0 = -1/sqrt(D23*D12)*(-C23*vb2+C22*vb3-1/D12*(C23*C12-C13*C22)*(-C22*vb1+C12*vb2));
        dcosphi0123_dr0 = -root_D12D23_inv*(-C23*vb2+C22*vb3-1/D12*(C23*C12-C13*C22)*(-C22*vb1+C12*vb2));
        F0_phi_0123 = -dV_dcosphi0123*dcosphi0123_dr0;
        F0.x = F0_theta_012.x + F0_theta_123.x + F0_phi_0123.x;
        F0.y = F0_theta_012.y + F0_theta_123.y + F0_phi_0123.y;
        F0.z = F0_theta_012.z + F0_theta_123.z + F0_phi_0123.z;

        //Force on particle 1
        //angle. a = 1, acting on theta_a+1
        dcostheta012_dr1 = root_C11C22_inv*((C12/C22)*vb2-(C12/C11)*vb1+vb2-vb1);
        F1_theta_012 = -dV_dtheta012*dtheta012_dcostheta012*dcostheta012_dr1;
        //angle. a = 1, acting on theta_a+2
        dcostheta123_dr1 = root_C22C33_inv*((C23/C22)*vb2-vb3);
        F1_theta_123 = -dV_dtheta123*dtheta123_dcostheta123*dcostheta123_dr1;
        //dihedral. a = 1, acting on phi_a+2
        // dcosphi0123_dr1 = -1/sqrt(D23*D12)*(-C12*vb3+C23*vb2-C23*vb1-C22*vb3+2*C13*vb2-1/D12*(C23*C12-C13*C22)*(C22*vb1-C11*vb2-C12*vb2+C12*vb1)-1/D23*(C23*C12-C13*C22)*(-C33*vb2+C23*vb3));
        dcosphi0123_dr1 = -root_D12D23_inv*(-C12*vb3+C23*vb2-C23*vb1-C22*vb3+2*C13*vb2-1/D12*(C23*C12-C13*C22)*(C22*vb1-C11*vb2-C12*vb2+C12*vb1)-1/D23*(C23*C12-C13*C22)*(-C33*vb2+C23*vb3));
        F1_phi_0123 = -dV_dcosphi0123*dcosphi0123_dr1;
        F1.x = F1_theta_012.x + F1_theta_123.x + F1_phi_0123.x;
        F1.y = F1_theta_012.y + F1_theta_123.y + F1_phi_0123.y;
        F1.z = F1_theta_012.z + F1_theta_123.z + F1_phi_0123.z;

        //Force on particle 2
        //angle. a = 2, acting on theta_a
        dcostheta012_dr2 = -root_C11C22_inv*((C12/C22)*vb2-vb1);
        F2_theta_012 = -dV_dtheta012*dtheta012_dcostheta012*dcostheta012_dr2;
        //angle. a = 2, acting on theta_a+1
        dcostheta123_dr2 = root_C22C33_inv*((C23/C33)*vb3-(C23/C22)*vb2+vb3-vb2);
        F2_theta_123 = -dV_dtheta123*dtheta123_dcostheta123*dcostheta123_dr2;
        //dihedral. a = 2, acting on phi_a+1
        // dcosphi0123_dr2 = -1/sqrt(D23*D12)*(C12*vb3-C12*vb2+C23*vb1+C22*vb1-2*C13*vb2-1/D12*(C23*C12-C13*C22)*(C11*vb2-C12*vb1)-1/D23*(C23*C12-C13*C22)*(C33*vb2-C22*vb3-C23*vb3+C23*vb2));
        dcosphi0123_dr2 = -root_D12D23_inv*(C12*vb3-C12*vb2+C23*vb1+C22*vb1-2*C13*vb2-1/D12*(C23*C12-C13*C22)*(C11*vb2-C12*vb1)-1/D23*(C23*C12-C13*C22)*(C33*vb2-C22*vb3-C23*vb3+C23*vb2));
        F2_phi_0123 = -dV_dcosphi0123*dcosphi0123_dr2;
        F2.x = F2_theta_012.x + F2_theta_123.x + F2_phi_0123.x;
        F2.y = F2_theta_012.y + F2_theta_123.y + F2_phi_0123.y;
        F2.z = F2_theta_012.z + F2_theta_123.z + F2_phi_0123.z;

        //Force on particle 3
        //angle. a = 3, acting on theta_a
        F3_theta_012 = make_scalar3(0,0,0);
        dcostheta123_dr3 = -root_C22C33_inv*((C23/C33)*vb3-vb2);
        F3_theta_123 = -dV_dtheta123*dtheta123_dcostheta123*dcostheta123_dr3;
        //dihedral. a = 3, acting on phi_a
        // dcosphi0123_dr3 = -1/sqrt(D23*D12)*(C12*vb2-C22*vb1-1/D23*(C23*C12-C13*C22)*(C22*vb3-C23*vb2));
        dcosphi0123_dr3 = -root_D12D23_inv*(C12*vb2-C22*vb1-1/D23*(C23*C12-C13*C22)*(C22*vb3-C23*vb2));
        F3_phi_0123 = -dV_dcosphi0123*dcosphi0123_dr3;
        F3.x = F3_theta_012.x + F3_theta_123.x + F3_phi_0123.x;        
        F3.y = F3_theta_012.y + F3_theta_123.y + F3_phi_0123.y;        
        F3.z = F3_theta_012.z + F3_theta_123.z + F3_phi_0123.z;        

        /**
        c = (ax * bx + ay * by + az * bz) * rabinv;
        s = rg * rabinv * (ax * vb3.x + ay * vb3.y + az * vb3.z);

        if (c > 1.0)
            c = 1.0;
        if (c < -1.0)
            c = -1.0;
         */

        // calculate the potential p = k_phi * sin^3(theta_i-1) * sin^3(theta_i) * sum (n=0,4) (a_n*cos^n(phi) )

        // Compute 1/4 of energy to assign to each of 4 atoms in the dihedral
        e_dihedral = 0.25 * prefactor * torsion_e;
        F0.w = e_dihedral;
        F1.w = e_dihedral;
        F2.w = e_dihedral;
        F3.w = e_dihedral;

        //For the virials. From my reading, I believe I should do for each atom the dihedral + for atoms 1-3 the first angle virial + for atoms 2-4 the second angle virial;
        angle_012_virial[0] = (1. / 3.) * (vb1.x * F0_theta_012.x + vb2.x * F2_theta_012.x);
        angle_012_virial[1] = (1. / 3.) * (vb1.y * F0_theta_012.x + vb2.y * F2_theta_012.x);
        angle_012_virial[2] = (1. / 3.) * (vb1.z * F0_theta_012.x + vb2.z * F2_theta_012.x);
        angle_012_virial[3] = (1. / 3.) * (vb1.y * F0_theta_012.y + vb2.y * F2_theta_012.y);
        angle_012_virial[4] = (1. / 3.) * (vb1.z * F0_theta_012.y + vb2.z * F2_theta_012.y);
        angle_012_virial[5] = (1. / 3.) * (vb1.z * F0_theta_012.z + vb2.z * F2_theta_012.z);

        angle_123_virial[0] = (1. / 3.) * (vb2.x * F1_theta_123.x + vb3.x * F3_theta_123.x);
        angle_123_virial[1] = (1. / 3.) * (vb2.y * F1_theta_123.x + vb3.y * F3_theta_123.x);
        angle_123_virial[2] = (1. / 3.) * (vb2.z * F1_theta_123.x + vb3.z * F3_theta_123.x);
        angle_123_virial[3] = (1. / 3.) * (vb2.y * F1_theta_123.y + vb3.y * F3_theta_123.y);
        angle_123_virial[4] = (1. / 3.) * (vb2.z * F1_theta_123.y + vb3.z * F3_theta_123.y);
        angle_123_virial[5] = (1. / 3.) * (vb2.z * F1_theta_123.z + vb3.z * F3_theta_123.z);
    
        dihedral_virial[0] = 0.25 * (vb1.x * F0_phi_0123.x + vb2.x * F2_phi_0123.x + (vb3.x+ vb2.x) * F3_phi_0123.x);
        dihedral_virial[1] = 0.25 * (vb1.y * F0_phi_0123.x + vb2.y * F2_phi_0123.x + (vb3.y+ vb2.y) * F3_phi_0123.x);
        dihedral_virial[2] = 0.25 * (vb1.z * F0_phi_0123.x + vb2.z * F2_phi_0123.x + (vb3.z+ vb2.z) * F3_phi_0123.x);
        dihedral_virial[3] = 0.25 * (vb1.y * F0_phi_0123.y + vb2.y * F2_phi_0123.y + (vb3.y+ vb2.y) * F3_phi_0123.y);
        dihedral_virial[4] = 0.25 * (vb1.z * F0_phi_0123.y + vb2.z * F2_phi_0123.y + (vb3.z+ vb2.z) * F3_phi_0123.y);
        dihedral_virial[5] = 0.25 * (vb1.z * F0_phi_0123.z + vb2.z * F2_phi_0123.z + (vb3.z+ vb2.z) * F3_phi_0123.z);

        // Apply force to each of the 4 atoms
        h_force.data[i1].x = h_force.data[i1].x + F0.x;
        h_force.data[i1].y = h_force.data[i1].y + F0.y;
        h_force.data[i1].z = h_force.data[i1].z + F0.z;
        h_force.data[i1].w = h_force.data[i1].w + F0.w;
        h_force.data[i2].x = h_force.data[i2].x + F1.x;
        h_force.data[i2].y = h_force.data[i2].y + F1.y;
        h_force.data[i2].z = h_force.data[i2].z + F1.z;
        h_force.data[i2].w = h_force.data[i2].w + F1.w;
        h_force.data[i3].x = h_force.data[i3].x + F2.x;
        h_force.data[i3].y = h_force.data[i3].y + F2.y;
        h_force.data[i3].z = h_force.data[i3].z + F2.z;
        h_force.data[i3].w = h_force.data[i3].w + F2.w;
        h_force.data[i4].x = h_force.data[i4].x + F3.x;
        h_force.data[i4].y = h_force.data[i4].y + F3.y;
        h_force.data[i4].z = h_force.data[i4].z + F3.z;
        h_force.data[i4].w = h_force.data[i4].w + F3.w;

        for (int k = 0; k < 6; k++)
            {
            h_virial.data[virial_pitch * k + i1] += angle_012_virial[k] + dihedral_virial[k];
            h_virial.data[virial_pitch * k + i2] += angle_012_virial[k] + angle_123_virial[k] + dihedral_virial[k];
            h_virial.data[virial_pitch * k + i3] += angle_012_virial[k] + angle_123_virial[k] + dihedral_virial[k];
            h_virial.data[virial_pitch * k + i4] += angle_123_virial[k] + dihedral_virial[k];
            }
        }

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

namespace detail
    {
void export_DihedralBendingTorsionForceCompute(pybind11::module& m)
    {
    pybind11::class_<DihedralBendingTorsionForceCompute,
                     ForceCompute,
                     std::shared_ptr<DihedralBendingTorsionForceCompute>>(m,
                                                    "DihedralBendingTorsionForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &DihedralBendingTorsionForceCompute::setParams)
        .def("getParams", &DihedralBendingTorsionForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
