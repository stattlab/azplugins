// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

#include "DihedralBendingTorsionForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

/*! \file DihedralBendingTorsionForceGPU.cu
    \brief Defines GPU kernel code for calculating Bending-Torsion dihedral forces. Used by
   DihedralBendingTorsionForceComputeGPU.
*/

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {
//! Kernel for calculating combined bending-torsion dihedral forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param d_params Array of combined B-T parameters k_phi, a0,a1,a2,a3,a4
    \param box Box dimensions for periodic boundary condition handling
    \param tlist Dihedral data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
*/
__global__ void gpu_compute_bending_torsion_dihedral_forces_kernel(Scalar4* d_force,
                                                        Scalar* d_virial,
                                                        const size_t virial_pitch,
                                                        const unsigned int N,
                                                        const Scalar4* d_pos,
                                                        const dihedral_bending_torsion_params* d_params,
                                                        BoxDim box,
                                                        const group_storage<4>* tlist,
                                                        const unsigned int* dihedral_ABCD,
                                                        const unsigned int pitch,
                                                        const unsigned int* n_dihedrals_list)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_dihedrals = n_dihedrals_list[idx];

    // read in the position of our b-particle from the a-b-c-d set. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx]; // we can be either a, b, or c in the a-b-c-d quartet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 pos_a, pos_b, pos_c,
        pos_d; // allocate space for the a,b, and c atoms in the a-b-c-d quartet

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial_idx[6];
    for (unsigned int i = 0; i < 6; i++)
        virial_idx[i] = Scalar(0.0);

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
    Scalar dihedral_phi_virial[6];
    Scalar theta_ante_virial[6];
    Scalar theta_post_virial[6];

    // loop over all dihedrals
    for (int dihedral_idx = 0; dihedral_idx < n_dihedrals; dihedral_idx++)
        {
        group_storage<4> cur_dihedral = tlist[pitch * dihedral_idx + idx];
        unsigned int cur_ABCD = dihedral_ABCD[pitch * dihedral_idx + idx];

        int cur_dihedral_x_idx = cur_dihedral.idx[0];
        int cur_dihedral_y_idx = cur_dihedral.idx[1];
        int cur_dihedral_z_idx = cur_dihedral.idx[2];
        int cur_dihedral_type = cur_dihedral.idx[3];
        int cur_dihedral_abcd = cur_ABCD;

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_dihedral_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = d_pos[cur_dihedral_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 z_postype = d_pos[cur_dihedral_z_idx];
        Scalar3 z_pos = make_scalar3(z_postype.x, z_postype.y, z_postype.z);

        if (cur_dihedral_abcd == 0)
            {
            pos_a = idx_pos;
            pos_b = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 1)
            {
            pos_b = idx_pos;
            pos_a = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 2)
            {
            pos_c = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_d = z_pos;
            }
        if (cur_dihedral_abcd == 3)
            {
            pos_d = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_c = z_pos;
            }

        // get the dihedral parameters according to the type
        dihedral_bending_torsion_params params = d_params[cur_dihedral_type];
        k_phi = params.k_phi;
        a0 = params.a0;
        a1 = params.a1;
        a2 = params.a2;
        a3 = params.a3;
        a4 = params.a4;

        // bond vectors
        Scalar3 delta_ante = pos_b - pos_a;
        Scalar3 delta_crnt = pos_c - pos_b;
        Scalar3 delta_post = pos_d - pos_c;

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
        Here we avoid small values to prevent round-off errors. */
        if (d_ante < FLOAT_EPS)
        {
            d_ante = FLOAT_EPS;
        }
        if (d_post < FLOAT_EPS)
        {
            d_post = FLOAT_EPS;
        }

        /* Computations of cosines and configuration geometry*/
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

        /* PART 3 - COMPUTES THE FORCE COMPONENTS DUE TO THE DERIVATIVES OF BENDING ANGLE THETHA_ANTHE */

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

        // Apply force and energy depending on the relative position of the particle
        force_idx.w += e_dihedral;
        if (cur_dihedral_abcd == 0)
            {
            force_idx.x += f_phi_ai.x + f_theta_ante_ai.x;
            force_idx.y += f_phi_ai.y + f_theta_ante_ai.y;
            force_idx.z += f_phi_ai.z + f_theta_ante_ai.z;
            }
        if (cur_dihedral_abcd == 1)
            {
            force_idx.x += f_phi_aj.x + f_theta_ante_aj.x + f_theta_post_aj.x;
            force_idx.y += f_phi_aj.y + f_theta_ante_aj.y + f_theta_post_aj.y;
            force_idx.z += f_phi_aj.z + f_theta_ante_aj.z + f_theta_post_aj.z;
            }
        if (cur_dihedral_abcd == 2)
            {
            force_idx.x += f_phi_ak.x + f_theta_ante_ak.x + f_theta_post_ak.x;
            force_idx.y += f_phi_ak.y + f_theta_ante_ak.y + f_theta_post_ak.y;
            force_idx.z += f_phi_ak.z + f_theta_ante_ak.z + f_theta_post_ak.z;
            }
        if (cur_dihedral_abcd == 3)
            {
            force_idx.x += f_phi_al.x + f_theta_post_al.x;
            force_idx.y += f_phi_al.y + f_theta_post_al.y;
            force_idx.z += f_phi_al.z + f_theta_post_al.z;
            }

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
    
        dihedral_phi_virial[0] = 0.25 * (delta_ante.x * f_phi_ai.x + delta_crnt.x * f_phi_ak.x + (delta_post.x+ delta_crnt.x) * f_phi_al.x);
        dihedral_phi_virial[1] = 0.25 * (delta_ante.y * f_phi_ai.x + delta_crnt.y * f_phi_ak.x + (delta_post.y+ delta_crnt.y) * f_phi_al.x);
        dihedral_phi_virial[2] = 0.25 * (delta_ante.z * f_phi_ai.x + delta_crnt.z * f_phi_ak.x + (delta_post.z+ delta_crnt.z) * f_phi_al.x);
        dihedral_phi_virial[3] = 0.25 * (delta_ante.y * f_phi_ai.y + delta_crnt.y * f_phi_ak.y + (delta_post.y+ delta_crnt.y) * f_phi_al.y);
        dihedral_phi_virial[4] = 0.25 * (delta_ante.z * f_phi_ai.y + delta_crnt.z * f_phi_ak.y + (delta_post.z+ delta_crnt.z) * f_phi_al.y);
        dihedral_phi_virial[5] = 0.25 * (delta_ante.z * f_phi_ai.z + delta_crnt.z * f_phi_ak.z + (delta_post.z+ delta_crnt.z) * f_phi_al.z);
        
        if (cur_dihedral_abcd == 0)
            {
            for (int k = 0; k < 6; k++)
                virial_idx[k] = dihedral_phi_virial[k]+theta_ante_virial[k];
            }
        if (cur_dihedral_abcd == 1)
            {
            for (int k = 0; k < 6; k++)
                virial_idx[k] = dihedral_phi_virial[k]
                                + theta_ante_virial[k]
                                + theta_post_virial[k];
            }
        if (cur_dihedral_abcd == 2)
            {
            for (int k = 0; k < 6; k++)
                virial_idx[k] = dihedral_phi_virial[k]
                                + theta_ante_virial[k]
                                + theta_post_virial[k];
            }
        if (cur_dihedral_abcd == 3)
            {
            for (int k = 0; k < 6; k++)
                virial_idx[k] = dihedral_phi_virial[k]
                                + theta_post_virial[k];
            }
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    for (int k = 0; k < 6; k++)
        d_virial[k * virial_pitch + idx] = virial_idx[k];
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the GPU
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist Dihedral data to use in calculating the forces
    \param dihedral_ABCD List of relative atom positions in the dihedrals
    \param pitch Pitch of 2D dihedral list
    \param n_dihedrals_list List of numbers of dihedrals per atom
    \param d_params Array of combined B-T parameters k1/2, k2/2, k3/2, and k4/2
    \param n_dihedral_types Number of dihedral types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()

    \a d_params should include one Scalar4 element per dihedral type. The x component contains K the
   spring constant and the y component contains sign, and the z component the multiplicity.
*/
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
                                            const int warp_size)
    {
    assert(d_params);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_bending_torsion_dihedral_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    if (max_block_size % warp_size)
        // handle non-sensical return values from hipFuncGetAttributes
        max_block_size = (max_block_size / warp_size - 1) * warp_size;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_bending_torsion_dihedral_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_params,
                       box,
                       tlist,
                       dihedral_ABCD,
                       pitch,
                       n_dihedrals_list);

    return hipSuccess;
    }

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd
