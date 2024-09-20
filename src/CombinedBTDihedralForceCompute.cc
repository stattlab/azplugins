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
    GPUArray<combined_bt_params> params(m_dihedral_data->getNTypes(), m_exec_conf);
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
    combined_bt_params _params(params);
    // setParams(typ, _params.k_phi, _params.a0, _params.a1, _params.a2, _params.a3, _params.a4);
    ArrayHandle<combined_bt_params> h_params(m_params,
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
    ArrayHandle<combined_bt_params> h_params(m_params,
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
    ArrayHandle<combined_bt_params> h_params(m_params, access_location::host, 
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

        // calculate the potential p = k_phi * sin^3(theta_i-1) * sin^3(theta_i) * sum (n=0,4) (a_n*cos^n(phi) )
        // and df = dp/dc


        // n = 0
        p = a0
        // df = 

        // n = 1
        running_cos = c
        p += a1 * running_cos
        // df = 

        // n = 2
        running_cos = running_cos * c
        p += a2 * running_cos
        // df = 

        // n = 3
        running_cos = running_cos * c
        p += a3 * running_cos
        // df = 

        // n = 4
        running_cos = running_cos * c
        p += a4 * running_cos
        // df = 

        // prefactor
            // normal unit vector for testing
        r_v1  = sqrt(vb1.x * vb1.x + vb1.y * vb1.y + vb1.z * vb1.z)
        r_v2m = sqrt(vb2m.x * vb2m.x + vb2m.y * vb2m.y + vb2m.z * vb2m.z)
        r_v3  = sqrt(vb3.x * vb3.x + vb3.y * vb3.y + vb3.z * vb3.z)
        sin_0 = sqrt(rasq)/r_v1/r_v2m
        sin_1 = sqrt(rbsq)/r_v2m/r_v3
        pref = k_phi * sin_0 * sin_0 * sin_0 * sin_1 * sin_1 * sin_1

        p = pref * p
        // df = 


        /**
        // cos(phi) term
        ddf1 = c;
        df1 = s;
        cos_term = ddf1;

        p = k1 * (1.0 + cos_term);
        df = k1 * df1;
        // cos(2*phi) term
        ddf1 = cos_term * c - df1 * s;
        df1 = cos_term * s + df1 * c;
        cos_term = ddf1;

        p += k2 * (1.0 - cos_term);
        df += -2.0 * k2 * df1;

        // cos(3*phi) term
        ddf1 = cos_term * c - df1 * s;
        df1 = cos_term * s + df1 * c;
        cos_term = ddf1;

        p += k3 * (1.0 + cos_term);
        df += 3.0 * k3 * df1;

        // cos(4*phi) term
        ddf1 = cos_term * c - df1 * s;
        df1 = cos_term * s + df1 * c;
        cos_term = ddf1;

        p += k4 * (1.0 - cos_term);
        df += -4.0 * k4 * df1;
        */

        // Compute 1/4 of energy to assign to each of 4 atoms in the dihedral
        e_dihedral = 0.25 * p;

        //For each bead;
        //Caclualte all partials;
        // So lets start with just r0; F0; and then generalize;
        vb1 = r0-r1;
        vb2m = r1-r2;
        // vc = r2 - r3;
        vb3m.x = -vb3.x;
        vb3m.y = -vb3.y;
        vb3m.z = -vb3.z;
        vb3m = box.minImage(vb3m);

        Caa = vb1.x * vb1.x + vb1.y * vb1.y + vb1.z * vb1.z;
        Cab = vb1.x * vb2m.x + vb1.y * vb2m.y + vb1.z * vb2m.z;
        Cbb = vb2m.x * vb2m.x + vb2m.y * vb2m.y + vb2m.z * vb2m.z ;
        //Caa = np.dot(vb1,vb1);
        //Cab = np.dot(vb1,vb2m);
        //Cbb = np.dot(vb2m,vb2m);
        Dab = Caa*Cbb-Cab*Cab;
        Cbc = vb2m.x * vb3m.x + vb2m.y * vb3m.y + vb2m.z * vb3m.z;
        // Cbc = np.dot(vb2m,vb3m);
        Ccc = vb3m.x * vb3m.x + vb3m.y * vb3m.y + vb3m.z * vb3m.z;
        // Ccc=np.dot(vb3m,vb3m);
        Cac = vb1.x * vb3m.x + vb1.y * vb3m.y + vb1.z * vb3m.z;
        // Cac = np.dot(vb1,vb3m);
        Dbc = Cbb*Ccc-Cbc*Cbc;

        rb1 = sqrt(Caa);
        rb2 = sqrt(Cbb);
        rb3 = sqrt(Ccc);

        cos_theta_012 = Cab/(rb1*rb2);
        cos_theta_123 = Cbc/(rb2*rb3);
        theta_012 = acos(cos_theta_012);
        theta_123 = acos(cos_theta_123);
        /**
         * 
        Scalar3 d123, d012;
        d123.x = vb3m.y * vb2m.z - vb3m.z * vb2m.y;
        d123.y = vb3m.z * vb2m.x - vb3m.x * vb2m.z;
        d123.z = vb2m.x * vb3m.y - vb3m.y * vb2m.x;
        d012.x = vb2m.y * vb1.z - vb2m.z * vb1.y;
        d012.y = vb2m.z * vb1.x - vb2m.x * vb1.z;
        d012.z = vb2m.x * vb1.y - vb2m.y * vb1.x;

        // d123 = np.cross(vb3m,vb2m);
        // d012 = np.cross(vb2m,vb1);
        // at some point need to replace the norm of a parallel line = 0 to = .0001;
        cos_phi_0123 = -np.dot(d123,d012)/(np.linalg.norm(d123)*np.linalg.norm(d012));
         */
        cos_phi_0123 = c
        phi_0123 = acos(cos_phi_0123);
        // V,_,_,_ = V_CBT(r0,r1,r2,r3); p calculated above
    
        //Start by establishing what I need for r0;
        dVdtheta012 = 3*(sin(theta_012)*sin(theta_012))*cos_theta_012*p;
        dtheta012dcostheta012 = 1/sin(theta_012);
        da2dr0 = 2*vb1;
        dabdr0 = vb2m;
        dcostheta012dr0 = 1/rb1/rb2*(dabdr0)-cos_theta_012/2/rb1/rb1*(da2dr0);
        
        F0_theta_012 = -dVdtheta012*dtheta012dcostheta012*dcostheta012dr0;
        F0_theta_123 = 0;
        dVdcosphi_0123 = a1 + 2*a2*cos_phi_0123 + 3*a3*cos_phi_0123*cos_phi_0123 + 4*a4*cos_phi_0123*cos_phi_0123*cos_phi_0123;
        dVdcosphi_0123 = dVdcosphi_0123*p;
        dcosphidr0 = -(sqrt(1/Dab/Dbc))*(Cbc*vb2m-Cbb*vb3m-1/Dbc*(Cab*Cbc-Cac*Cbb)*(Cbb*vb1-Cab*vb2m));
        // dcosphidr0 = -(Dab*Dbc)**(-1/2)*(Cbc*vb2m-Cbb*vb3m-1/Dbc*(Cab*Cbc-Cac*Cbb)*(Cbb*vb1-Cab*vb2m));
        F0_phi_0123 = -dVdcosphi_0123*dcosphidr0;
        F0 = F0_theta_012 + F0_theta_123 + F0_phi_0123;
    
        //Next, let's do F1, where we must consider new things because we are part of two angle potentials instead of just one,
        da2dr1 = -2*vb1;
        db2dr1 = 2*vb2m;
        dabdr1 = vb2m-vb1;
        dcostheta012dr1 = 1/rb1/rb2*(dabdr1)-cos_theta_012/2*((1/rb1/rb1)*(da2dr1)+1/rb2/rb2*db2dr1);
        F1_theta_012 = -dVdtheta012*dtheta012dcostheta012*dcostheta012dr1;
        //dV 123 nonspecific terms;
        dVdtheta123=3*(sin(theta_123)*sin(theta_123))*cos_theta_123*p;
        dtheta123dcostheta123 = 1/sin(theta_123);
        
        da2dr1_123 = 2*vb2m;
        dabdr1_123 = vb3m;
        dcostheta123dr1 = 1/rb2/rb3*(dabdr1_123)-cos_theta_123/2/(rb2*rb2)*(da2dr1_123);
        F1_theta_123 = -dVdtheta123*dtheta123dcostheta123*dcostheta123dr1;
    
        dcosphidr1 = -1/sqrt(Dab*Dbc)*(Cbc*vb1 - Cbc*vb2m+Cab*vb3m+Cbb*vb3m-2*Cac*vb2m-1/Dbc*(Cab*Cbc-Cac*Cbb)*(Ccc*vb2m-Cbc*vb3m)-1/Dab*(Cab*Cbc-Cac*Cbb)*(Caa*vb2m-Cbb*vb1-Cab*vb1+Cab*vb2m));
        F1_phi_0123 = -dVdcosphi_0123*dcosphidr1;
        F1 = F1_phi_0123+F1_theta_012+F1_theta_123;
    
        //F2;
        da2dr2 = 0;
        db2dr2 = -2*vb2m;
        dabdr2 = -vb1;
        dcostheta012dr2 = 1/rb1/rb2*(dabdr2)-cos_theta_012/2*((1/rb1/rb1)*(da2dr2)+1/rb2/rb2*db2dr2);
        F2_theta_012 = -dVdtheta012*dtheta012dcostheta012*dcostheta012dr2;
        da2dr2_123 = -2*vb2m;
        db2dr2_123 = 2*vb3m;
        dabdr2_123 = vb3m-vb2m;
        dcostheta123dr2 = 1/rb2/rb3*(dabdr2_123)-cos_theta_123/2*((1/rb2/rb2)*(da2dr2_123)+1/(rb3*rb3)*db2dr2_123);
        F2_theta_123 = -dVdtheta123*dtheta123dcostheta123*dcostheta123dr2;
        dcosphidr2 = -(sqrt(1/Dab/Dbc))*(-Cbc*vb1+Cab*vb2m-Cab*vb3m-Cbb*vb1+2*Cac*vb2m-1/Dbc*(Cab*Cbc-Cac*Cbb)*(Cbb*vb3m-Ccc*vb2m-Cbc*vb2m+Cbc*vb3m)-1/Dab*(Cab*Cbc-Cac*Cbb)*(-Caa*vb2m+Cab*vb1));
        // dcosphidr2 = -(Dab*Dbc)**(-1/2)*(-Cbc*vb1+Cab*vb2m-Cab*vb3m-Cbb*vb1+2*Cac*vb2m-1/Dbc*(Cab*Cbc-Cac*Cbb)*(Cbb*vb3m-Ccc*vb2m-Cbc*vb2m+Cbc*vb3m)-1/Dab*(Cab*Cbc-Cac*Cbb)*(-Caa*vb2m+Cab*vb1));
        F2_phi_0123 = -dVdcosphi_0123*dcosphidr2;
    
        F2 = F2_theta_012+F2_theta_123+F2_phi_0123;
    
        //F3;
        F3_theta_012 = 0;
    
        da2dr3_123 = 0;
        db2dr3_123 = -2*vb3m;
        dabdr3_123 = -vb2m;
        dcostheta123dr3 = 1/rb2/rb3*(dabdr3_123)-cos_theta_123/2*((1/rb2/rb2)*(da2dr3_123)+1/rb3/rb3*db2dr3_123);
        F3_theta_123 = -dVdtheta123*dtheta123dcostheta123*dcostheta123dr3;
    
        dcosphidr3 = -(sqrt(1/Dab/Dbc))*(-Cab*vb2m+Cbb*vb1-1/Dbc*(Cab*Cbc-Cac*Cbb)*(-Cbb*vb3m+Cbc*vb2m));
        // dcosphidr3 = -(Dab*Dbc)**(-1/2)*(-Cab*vb2m+Cbb*vb1-1/Dbc*(Cab*Cbc-Cac*Cbb)*(-Cbb*vb3m+Cbc*vb2m));
        F3_phi_0123 = -dVdcosphi_0123*dcosphidr3;
        F3 = F3_theta_012+F3_phi_0123+F3_theta_123;
        // F = [F0,F1,F2,F3];
        //cbt_eng;
        cbt_eng = 0.25*p //I believe this is correct since there is no way to seperate the different angles in the potential;
    
        //For the virials. From my reading, I believe I should do for each atom the dihedral + for atoms 1-3 the first angle virial + for atoms 2-4 the second angle virial;
        angle_012_v[0] = (1. / 3.) * (vb1[0] * F0_theta_012[0] + vb2m[0] * F2_theta_012[0]);
        angle_012_v[1] = (1. / 3.) * (vb1[1] * F0_theta_012[0] + vb2m[1] * F2_theta_012[0]);
        angle_012_v[2] = (1. / 3.) * (vb1[2] * F0_theta_012[0] + vb2m[2] * F2_theta_012[0]);
        angle_012_v[3] = (1. / 3.) * (vb1[1] * F0_theta_012[1] + vb2m[1] * F2_theta_012[1]);
        angle_012_v[4] = (1. / 3.) * (vb1[2] * F0_theta_012[1] + vb2m[2] * F2_theta_012[1]);
        angle_012_v[5] = (1. / 3.) * (vb1[2] * F0_theta_012[2] + vb2m[2] * F2_theta_012[2]);
    
        angle_123_v[0] = (1. / 3.) * (vb2m[0] * F1_theta_123[0] + vb3m[0] * F3_theta_123[0]);
        angle_123_v[1] = (1. / 3.) * (vb2m[1] * F1_theta_123[0] + vb3m[1] * F3_theta_123[0]);
        angle_123_v[2] = (1. / 3.) * (vb2m[2] * F1_theta_123[0] + vb3m[2] * F3_theta_123[0]);
        angle_123_v[3] = (1. / 3.) * (vb2m[1] * F1_theta_123[1] + vb3m[1] * F3_theta_123[1]);
        angle_123_v[4] = (1. / 3.) * (vb2m[2] * F1_theta_123[1] + vb3m[2] * F3_theta_123[1]);
        angle_123_v[5] = (1. / 3.) * (vb2m[2] * F1_theta_123[2] + vb3m[2] * F3_theta_123[2]);
    
        dihedral_virial[0] = 0.25 * (vb1[0] * F0_phi_0123[0] + vb2m[0] * F2_phi_0123[0] + (vb3m[0]+ vb2m[0]) * F3_phi_0123[0]);
        dihedral_virial[1] = 0.25 * (vb1[1] * F0_phi_0123[0] + vb2m[1] * F2_phi_0123[0] + (vb3m[1]+ vb2m[1]) * F3_phi_0123[0]);
        dihedral_virial[2] = 0.25 * (vb1[2] * F0_phi_0123[0] + vb2m[2] * F2_phi_0123[0] + (vb3m[2]+ vb2m[2]) * F3_phi_0123[0]);
        dihedral_virial[3] = 0.25 * (vb1[1] * F0_phi_0123[1] + vb2m[1] * F2_phi_0123[1] + (vb3m[1]+ vb2m[1]) * F3_phi_0123[1]);
        dihedral_virial[4] = 0.25 * (vb1[2] * F0_phi_0123[1] + vb2m[2] * F2_phi_0123[1] + (vb3m[2]+ vb2m[2]) * F3_phi_0123[1]);
        dihedral_virial[5] = 0.25 * (vb1[2] * F0_phi_0123[2] + vb2m[2] * F2_phi_0123[2] + (vb3m[2]+ vb2m[2]) * F3_phi_0123[2]);

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
