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
    Scalar3 delta_ante, delta_crnt, delta_post;

    // this volatile is not strictly needed, but it works around a compiler bug on Mac 
    // arm64 with Apple clang version 13.0.0 (clang-1300.0.29.30) that occurs with the 
    // OPLS force compute.
    // Without the volatile, the x component of f_phi_aj may always computed the same as
    // the y component
    volatile Scalar3 f_theta_ante_ai, f_theta_ante_aj, f_theta_ante_ak;
    volatile Scalar3 f_theta_post_aj, f_theta_post_ak, f_theta_post_al;
    volatile Scalar3 f_phi_ai, f_phi_aj, f_phi_ak, f_phi_al;
    Scalar e_dihedral;
    Scalar k_phi, a0, a1, a2, a3, a4;
    Scalar dihedral_virial[6];
    Scalar theta_ante_virial[6];
    Scalar theta_post_virial[6];

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

        // 1st bond
        delta_ante.x = h_pos.data[i2].x - h_pos.data[i1].x;
        delta_ante.y = h_pos.data[i2].y - h_pos.data[i1].y;
        delta_ante.z = h_pos.data[i2].z - h_pos.data[i1].z;
        // 2nd bond
        delta_crnt.x = h_pos.data[i3].x - h_pos.data[i2].x;
        delta_crnt.y = h_pos.data[i3].y - h_pos.data[i2].y;
        delta_crnt.z = h_pos.data[i3].z - h_pos.data[i2].z;
        // 3rd bond
        delta_post.x = h_pos.data[i4].x - h_pos.data[i3].x;
        delta_post.y = h_pos.data[i4].y - h_pos.data[i3].y;
        delta_post.z = h_pos.data[i4].z - h_pos.data[i3].z;

        // apply periodic boundary conditions
        delta_ante = box.minImage(delta_ante);
        delta_crnt = box.minImage(delta_crnt);
        delta_post = box.minImage(delta_post);

        // Computation of forces
        Scalar c_self_ante, c_self_crnt, c_self_post;
        Scalar c_cros_ante, c_cros_acrs, c_cros_post;
        Scalar c_prod, d_ante, d_post;
        Scalar norm_phi, norm_theta_ante, norm_theta_post;
        Scalar cosine_phi, cosine_phi_sq, cosine_theta_ante, cosine_theta_post;
        Scalar sine_theta_ante_sq, sine_theta_post_sq;
        Scalar sine_theta_ante, sine_theta_post;
        Scalar prefactor_phi;
        Scalar ratio_phi_ante, ratio_phi_post;
        Scalar factor_phi_ai_ante, factor_phi_ai_crnt, factor_phi_ai_post;
        Scalar factor_phi_aj_ante, factor_phi_aj_crnt, factor_phi_aj_post;
        Scalar factor_phi_ak_ante, factor_phi_ak_crnt, factor_phi_ak_post;
        Scalar factor_phi_al_ante, factor_phi_al_crnt, factor_phi_al_post;
        Scalar prefactor_theta_ante, ratio_theta_ante_ante, ratio_theta_ante_crnt;
        Scalar prefactor_theta_post, ratio_theta_post_crnt, ratio_theta_post_post;
        Scalar energy;

        /* The formula for combined bending-torsion potential (see file "dihedral.py") 
        * contains in its expression not only the dihedral angle \f[\phi\f] but also 
        * \f[\theta_{i-1}\f] (theta_ante below) and \f[\theta_{i}\f]
        * (theta_post below)---the adjacent bending angles. The forces for the
        * particles ai, aj, ak, al have components coming from the derivatives of the 
        * potential with respect to all three angles.
        * This function is organized in 5 parts
        * PART 1 - Computes force factors common to all the derivatives for the four 
        *           particles
        * PART 2 - Computes the force components due to the derivatives of dihedral
        *           angle Phi
        * PART 3 - Computes the force components due to the derivatives of bending angle
        *           Theta_Ante
        * PART 4 - Computes the force components due to the derivatives of bending angle
        *           Theta_Post
        * PART 5 - Applies the forces and virial tensor from this dihedral
        * Below we will respect this structure */


        /* PART 1 - COMPUTES FORCE FACTORS COMMON TO ALL DERIVATIVES FOR THE FOUR PARTICLES */

        /* Computation of the cosine of the dihedral angle. The scalar ("dot") product method
        * is used. c_*_* cumulate the scalar products of the differences of particles
        * positions while c_prod, d_ante and d_post are differences of products of scalar
        * terms that are parts of the derivatives of forces */
        c_self_ante = dot(delta_ante, delta_ante);
        c_self_crnt = dot(delta_crnt, delta_crnt);
        c_self_post = dot(delta_post, delta_post);
        c_cros_ante = dot(delta_ante, delta_crnt);
        c_cros_acrs = dot(delta_ante, delta_post);
        c_cros_post = dot(delta_crnt, delta_post);
        c_prod      = c_cros_ante * c_cros_post - c_self_crnt * c_cros_acrs;
        d_ante      = c_self_ante * c_self_crnt - c_cros_ante * c_cros_ante;
        d_post      = c_self_post * c_self_crnt - c_cros_post * c_cros_post;

        /*  When three consecutive beads align, we obtain values close to zero.
        *	Here we avoid small values to prevent round-off errors. */
        if (d_ante < FLOAT_EPS)
        {
            d_ante = FLOAT_EPS;
        }
        if (d_post < FLOAT_EPS)
        {
            d_post = FLOAT_EPS;
        }

        /* Computations of cosines and configuration geometry */
        norm_phi           = 1.0 / std::sqrt(d_ante * d_post);
        norm_theta_ante    = 1.0 / std::sqrt(c_self_ante * c_self_crnt);
        norm_theta_post    = 1.0 / std::sqrt(c_self_crnt * c_self_post);
        cosine_phi         = c_prod * norm_phi;
        cosine_theta_ante  = c_cros_ante * norm_theta_ante;
        cosine_theta_post  = c_cros_post * norm_theta_post;
        sine_theta_ante_sq = 1 - cosine_theta_ante * cosine_theta_ante;
        sine_theta_post_sq = 1 - cosine_theta_post * cosine_theta_post;

        /*	It is possible that cosine_theta is slightly bigger than 1.0 due to round-off errors. */
        if (sine_theta_ante_sq < 0.0)
        {
            sine_theta_ante_sq = 0.0;
        }
        if (sine_theta_post_sq < 0.0)
        {
            sine_theta_post_sq = 0.0;
        }
        sine_theta_ante = std::sqrt(sine_theta_ante_sq);
        sine_theta_post = std::sqrt(sine_theta_post_sq);

        /* PART 2 - COMPUTES FORCE COMPONENTS DUE TO DERIVATIVES TO DIHEDRAL ANGLE PHI */

        /*      Computation of ratios */
        ratio_phi_ante = c_prod / d_ante;
        ratio_phi_post = c_prod / d_post;

        /*       Computation of the prefactor */
        cosine_phi_sq = cosine_phi*cosine_phi;
        prefactor_phi = -k_phi * norm_phi
                        * (a1 + a2 * 2.0 * cosine_phi
                        + a3 * 3.0 * (cosine_phi_sq) + 4 * a4 * cosine_phi_sq * cosine_phi)
                        * sine_theta_ante_sq * sine_theta_ante * sine_theta_post_sq * sine_theta_post;

        /* Computation of factors (important for gaining speed). Factors factor_phi_*  
        * are coming from the derivatives of the torsion angle (phi) with respect to the
        * beads ai, aj, al, ak, (four) coordinates and they are multiplied in the force 
        * computations with the differences of the particles positions stored in 
        * parameters delta_ante, delta_crnt, delta_post. For formulas see file 
        * "dihedral.py" */

        factor_phi_ai_ante = ratio_phi_ante * c_self_crnt;
        factor_phi_ai_crnt = -c_cros_post - ratio_phi_ante * c_cros_ante;
        factor_phi_ai_post = c_self_crnt;
        factor_phi_aj_ante = -c_cros_post - ratio_phi_ante * (c_self_crnt + c_cros_ante);
        factor_phi_aj_crnt = c_cros_post + c_cros_acrs * 2.0
                            + ratio_phi_ante * (c_self_ante + c_cros_ante) + ratio_phi_post * c_self_post;
        factor_phi_aj_post = -(c_cros_ante + c_self_crnt) - ratio_phi_post * c_cros_post;
        factor_phi_ak_ante = c_cros_post + c_self_crnt + ratio_phi_ante * c_cros_ante;
        factor_phi_ak_crnt = -(c_cros_ante + c_cros_acrs * 2.0) - ratio_phi_ante * c_self_ante
                            - ratio_phi_post * (c_self_post + c_cros_post);
        factor_phi_ak_post = c_cros_ante + ratio_phi_post * (c_self_crnt + c_cros_post);
        factor_phi_al_ante = -c_self_crnt;
        factor_phi_al_crnt = c_cros_ante + ratio_phi_post * c_cros_post;
        factor_phi_al_post = -ratio_phi_post * c_self_crnt;

        /* Computation of forces due to the derivatives of dihedral angle phi*/
        f_phi_ai.x = prefactor_phi
                    * (factor_phi_ai_ante * delta_ante.x + factor_phi_ai_crnt * delta_crnt.x
                        + factor_phi_ai_post * delta_post.x);
        f_phi_aj.x = prefactor_phi
                    * (factor_phi_aj_ante * delta_ante.x + factor_phi_aj_crnt * delta_crnt.x
                        + factor_phi_aj_post * delta_post.x);
        f_phi_ak.x = prefactor_phi
                    * (factor_phi_ak_ante * delta_ante.x + factor_phi_ak_crnt * delta_crnt.x
                        + factor_phi_ak_post * delta_post.x);
        f_phi_al.x = prefactor_phi
                    * (factor_phi_al_ante * delta_ante.x + factor_phi_al_crnt * delta_crnt.x
                        + factor_phi_al_post * delta_post.x);

        f_phi_ai.y = prefactor_phi
                        * (factor_phi_ai_ante * delta_ante.y + factor_phi_ai_crnt * delta_crnt.y
                            + factor_phi_ai_post * delta_post.y);
        f_phi_aj.y = prefactor_phi
                    * (factor_phi_aj_ante * delta_ante.y + factor_phi_aj_crnt * delta_crnt.y
                        + factor_phi_aj_post * delta_post.y);
        f_phi_ak.y = prefactor_phi
                    * (factor_phi_ak_ante * delta_ante.y + factor_phi_ak_crnt * delta_crnt.y
                        + factor_phi_ak_post * delta_post.y);
        f_phi_al.y = prefactor_phi
                    * (factor_phi_al_ante * delta_ante.y + factor_phi_al_crnt * delta_crnt.y
                        + factor_phi_al_post * delta_post.y);

        f_phi_ai.z = prefactor_phi
                        * (factor_phi_ai_ante * delta_ante.z + factor_phi_ai_crnt * delta_crnt.z
                            + factor_phi_ai_post * delta_post.z);
        f_phi_aj.z = prefactor_phi
                    * (factor_phi_aj_ante * delta_ante.z + factor_phi_aj_crnt * delta_crnt.z
                        + factor_phi_aj_post * delta_post.z);
        f_phi_ak.z = prefactor_phi
                    * (factor_phi_ak_ante * delta_ante.z + factor_phi_ak_crnt * delta_crnt.z
                        + factor_phi_ak_post * delta_post.z);
        f_phi_al.z = prefactor_phi
                    * (factor_phi_al_ante * delta_ante.z + factor_phi_al_crnt * delta_crnt.z
                        + factor_phi_al_post * delta_post.z);

        /* PART 3 - COMPUTES THE FORCE COMPONENTS DUE TO THE DERIVATIVES OF BENDING ANGLE THETA_ANTE */

        /*      Computation of ratios */
        ratio_theta_ante_ante = c_cros_ante / c_self_ante;
        ratio_theta_ante_crnt = c_cros_ante / c_self_crnt;

        /*      Computation of the prefactor */
        prefactor_theta_ante =
                -k_phi * norm_theta_ante
                * (a0 + a1 * cosine_phi + a2 * (cosine_phi_sq)
                + a3 * (cosine_phi * cosine_phi_sq) + a4 * (cosine_phi_sq * cosine_phi_sq))
                * (-3.0) * cosine_theta_ante * sine_theta_ante * sine_theta_post_sq * sine_theta_post;

        /*      Computation of forces due to the derivatives of bending angle theta_ante */
        f_theta_ante_ai.x =
                prefactor_theta_ante * (ratio_theta_ante_ante * delta_ante.x - delta_crnt.x);
        f_theta_ante_aj.x = prefactor_theta_ante
                            * ((ratio_theta_ante_crnt + 1.0) * delta_crnt.x
                                - (ratio_theta_ante_ante + 1.0) * delta_ante.x);
        f_theta_ante_ak.x =
                prefactor_theta_ante * (delta_ante.x - ratio_theta_ante_crnt * delta_crnt.x);

        f_theta_ante_ai.y =
                prefactor_theta_ante * (ratio_theta_ante_ante * delta_ante.y - delta_crnt.y);
        f_theta_ante_aj.y = prefactor_theta_ante
                            * ((ratio_theta_ante_crnt + 1.0) * delta_crnt.y
                                - (ratio_theta_ante_ante + 1.0) * delta_ante.y);
        f_theta_ante_ak.y =
                prefactor_theta_ante * (delta_ante.y - ratio_theta_ante_crnt * delta_crnt.y);

        f_theta_ante_ai.z =
                prefactor_theta_ante * (ratio_theta_ante_ante * delta_ante.z - delta_crnt.z);
        f_theta_ante_aj.z = prefactor_theta_ante
                            * ((ratio_theta_ante_crnt + 1.0) * delta_crnt.z
                                - (ratio_theta_ante_ante + 1.0) * delta_ante.z);
        f_theta_ante_ak.z =
                prefactor_theta_ante * (delta_ante.z - ratio_theta_ante_crnt * delta_crnt.z);

        /* PART 4 - COMPUTES THE FORCE COMPONENTS DUE TO THE DERIVATIVES OF THE BENDING ANGLE THETA_POST */

        /*      Computation of ratios */
        ratio_theta_post_crnt = c_cros_post / c_self_crnt;
        ratio_theta_post_post = c_cros_post / c_self_post;

        /*     Computation of the prefactor */
        prefactor_theta_post =
                -k_phi * norm_theta_post
                * (a0 + a1 * cosine_phi + a2 * (cosine_phi_sq)
                + a3 * (cosine_phi * cosine_phi_sq) + a4 * (cosine_phi_sq * cosine_phi_sq))
                * sine_theta_ante_sq * sine_theta_ante * (-3.0) * cosine_theta_post * sine_theta_post;

        /*      Computation of forces due to the derivatives of bending angle Theta_Post */
        f_theta_post_aj.x =
                prefactor_theta_post * (ratio_theta_post_crnt * delta_crnt.x - delta_post.x);
        f_theta_post_ak.x = prefactor_theta_post
                            * ((ratio_theta_post_post + 1.0) * delta_post.x
                                - (ratio_theta_post_crnt + 1.0) * delta_crnt.x);
        f_theta_post_al.x =
                prefactor_theta_post * (delta_crnt.x - ratio_theta_post_post * delta_post.x);
        
        f_theta_post_aj.y =
                prefactor_theta_post * (ratio_theta_post_crnt * delta_crnt.y - delta_post.y);
        f_theta_post_ak.y = prefactor_theta_post
                            * ((ratio_theta_post_post + 1.0) * delta_post.y
                                - (ratio_theta_post_crnt + 1.0) * delta_crnt.y);
        f_theta_post_al.y =
                prefactor_theta_post * (delta_crnt.y - ratio_theta_post_post * delta_post.y);

        f_theta_post_aj.z =
                prefactor_theta_post * (ratio_theta_post_crnt * delta_crnt.z - delta_post.z);
        f_theta_post_ak.z = prefactor_theta_post
                            * ((ratio_theta_post_post + 1.0) * delta_post.z
                                - (ratio_theta_post_crnt + 1.0) * delta_crnt.z);
        f_theta_post_al.z =
                prefactor_theta_post * (delta_crnt.z - ratio_theta_post_post * delta_post.z);

        /* PART 5 - APPLIES THE FORCES AND VIRIAL TENSOR FROM THIS DIHEDRAL */

        /* Contribution to energy - for formula see file "dihedral.py" */
        energy = k_phi
            * (a0 + a1 * cosine_phi + a2 * cosine_phi_sq
                + a3 * (cosine_phi * cosine_phi_sq) + a4 * (cosine_phi_sq * cosine_phi_sq))
            * sine_theta_ante_sq * sine_theta_ante * sine_theta_post_sq * sine_theta_post;
        // Compute 1/4 of energy to assign to each of 4 atoms in the dihedral
        e_dihedral = 0.25 * energy;

        // Apply force and energy to each of the 4 atoms
        h_force.data[i1].x = h_force.data[i1].x + f_phi_ai.x + f_theta_ante_ai.x;
        h_force.data[i1].y = h_force.data[i1].y + f_phi_ai.y + f_theta_ante_ai.y;
        h_force.data[i1].z = h_force.data[i1].z + f_phi_ai.z + f_theta_ante_ai.z;
        h_force.data[i1].w = h_force.data[i1].w + e_dihedral;
        h_force.data[i2].x = h_force.data[i2].x + f_phi_aj.x + f_theta_ante_aj.x + f_theta_post_aj.x;
        h_force.data[i2].y = h_force.data[i2].y + f_phi_aj.y + f_theta_ante_aj.y + f_theta_post_aj.y;
        h_force.data[i2].z = h_force.data[i2].z + f_phi_aj.z + f_theta_ante_aj.z + f_theta_post_aj.z;
        h_force.data[i2].w = h_force.data[i2].w + e_dihedral;
        h_force.data[i3].x = h_force.data[i3].x + f_phi_ak.x + f_theta_ante_ak.x + f_theta_post_ak.x;
        h_force.data[i3].y = h_force.data[i3].y + f_phi_ak.y + f_theta_ante_ak.y + f_theta_post_ak.y;
        h_force.data[i3].z = h_force.data[i3].z + f_phi_ak.z + f_theta_ante_ak.z + f_theta_post_ak.z;
        h_force.data[i3].w = h_force.data[i3].w + e_dihedral;
        h_force.data[i4].x = h_force.data[i4].x + f_phi_al.x + f_theta_post_al.x;
        h_force.data[i4].y = h_force.data[i4].y + f_phi_al.y + f_theta_post_al.y;
        h_force.data[i4].z = h_force.data[i4].z + f_phi_al.z + f_theta_post_al.z;
        h_force.data[i4].w = h_force.data[i4].w + e_dihedral;
        
        // Contributions to the virial from the dihedral, theta_ante, and theta_post
        theta_ante_virial[0] = (1. / 3.) * (delta_ante.x * f_theta_ante_ai.x + delta_crnt.x * f_theta_ante_ak.x);
        theta_ante_virial[1] = (1. / 3.) * (delta_ante.y * f_theta_ante_ai.x + delta_crnt.y * f_theta_ante_ak.x);
        theta_ante_virial[2] = (1. / 3.) * (delta_ante.z * f_theta_ante_ai.x + delta_crnt.z * f_theta_ante_ak.x);
        theta_ante_virial[3] = (1. / 3.) * (delta_ante.y * f_theta_ante_ai.y + delta_crnt.y * f_theta_ante_ak.y);
        theta_ante_virial[4] = (1. / 3.) * (delta_ante.z * f_theta_ante_ai.y + delta_crnt.z * f_theta_ante_ak.y);
        theta_ante_virial[5] = (1. / 3.) * (delta_ante.z * f_theta_ante_ai.z + delta_crnt.z * f_theta_ante_ak.z);

        theta_post_virial[0] = (1. / 3.) * (delta_crnt.x * f_theta_post_aj.x + delta_post.x * f_theta_post_al.x);
        theta_post_virial[1] = (1. / 3.) * (delta_crnt.y * f_theta_post_aj.x + delta_post.y * f_theta_post_al.x);
        theta_post_virial[2] = (1. / 3.) * (delta_crnt.z * f_theta_post_aj.x + delta_post.z * f_theta_post_al.x);
        theta_post_virial[3] = (1. / 3.) * (delta_crnt.y * f_theta_post_aj.y + delta_post.y * f_theta_post_al.y);
        theta_post_virial[4] = (1. / 3.) * (delta_crnt.z * f_theta_post_aj.y + delta_post.z * f_theta_post_al.y);
        theta_post_virial[5] = (1. / 3.) * (delta_crnt.z * f_theta_post_aj.z + delta_post.z * f_theta_post_al.z);
    
        dihedral_virial[0] = 0.25 * (delta_ante.x * f_phi_ai.x + delta_crnt.x * f_phi_ak.x + (delta_post.x+ delta_crnt.x) * f_phi_al.x);
        dihedral_virial[1] = 0.25 * (delta_ante.y * f_phi_ai.x + delta_crnt.y * f_phi_ak.x + (delta_post.y+ delta_crnt.y) * f_phi_al.x);
        dihedral_virial[2] = 0.25 * (delta_ante.z * f_phi_ai.x + delta_crnt.z * f_phi_ak.x + (delta_post.z+ delta_crnt.z) * f_phi_al.x);
        dihedral_virial[3] = 0.25 * (delta_ante.y * f_phi_ai.y + delta_crnt.y * f_phi_ak.y + (delta_post.y+ delta_crnt.y) * f_phi_al.y);
        dihedral_virial[4] = 0.25 * (delta_ante.z * f_phi_ai.y + delta_crnt.z * f_phi_ak.y + (delta_post.z+ delta_crnt.z) * f_phi_al.y);
        dihedral_virial[5] = 0.25 * (delta_ante.z * f_phi_ai.z + delta_crnt.z * f_phi_ak.z + (delta_post.z+ delta_crnt.z) * f_phi_al.z);

        // apply virial to each of the 4 particles
        for (int k = 0; k < 6; k++)
            {
            h_virial.data[virial_pitch * k + i1] += theta_ante_virial[k] + dihedral_virial[k];
            h_virial.data[virial_pitch * k + i2] += theta_ante_virial[k] + theta_post_virial[k] + dihedral_virial[k];
            h_virial.data[virial_pitch * k + i3] += theta_ante_virial[k] + theta_post_virial[k] + dihedral_virial[k];
            h_virial.data[virial_pitch * k + i4] += theta_post_virial[k] + dihedral_virial[k];
            }
        }
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
