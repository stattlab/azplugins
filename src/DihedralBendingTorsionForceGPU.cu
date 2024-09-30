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
        Scalar k_phi = params.k_phi;
        Scalar a0 = params.a0;
        Scalar a1 = params.a1;
        Scalar a2 = params.a2;
        Scalar a3 = params.a3;
        Scalar a4 = params.a4;
        /**
        dihedral_type = m_dihedral_data->getTypeByIndex(n);
        const dihedral_bending_torsion_params& param = h_params.data[dihedral_type];
        k_phi = param.k_phi;
        a0 = param.a0;
        a1 = param.a1;
        a2 = param.a2;
        a3 = param.a3;
        a4 = param.a4;
         */

        // the three bonds
        Scalar3 vb1 = pos_b - pos_a;
        Scalar3 vb2 = pos_c - pos_b;
        Scalar3 vb3 = pos_d - pos_c;

        // apply periodic boundary conditions
        vb1 = box.minImage(vb1);
        vb2 = box.minImage(vb2);
        vb3 = box.minImage(vb3);

        // simplifiers
        Scalar C11 = vb1.x * vb1.x + vb1.y * vb1.y + vb1.z * vb1.z;
        Scalar C12 = vb1.x * vb2.x + vb1.y * vb2.y + vb1.z * vb2.z;
        Scalar C22 = vb2.x * vb2.x + vb2.y * vb2.y + vb2.z * vb2.z;
        Scalar C23 = vb2.x * vb3.x + vb2.y * vb3.y + vb2.z * vb3.z;
        Scalar C33 = vb3.x * vb3.x + vb3.y * vb3.y + vb3.z * vb3.z;
        Scalar C13 = vb1.x * vb3.x + vb1.y * vb3.y + vb1.z * vb3.z;
        Scalar D12 = C11*C22-C12*C12;
        Scalar D23 = C22*C33-C23*C23;
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
        Scalar dV_dtheta012 = prefactor*3*cos_theta_012/sin_theta_012*torsion_e;
        Scalar dV_dtheta123 = prefactor*3*cos_theta_123/sin_theta_123*torsion_e;
        Scalar dV_dcosphi0123 = prefactor*(torsion_e-a0)/cos_phi_0123;

        Scalar dtheta012_dcostheta012 = -1/sin_theta_012;
        Scalar dtheta123_dcostheta123 = -1/sin_theta_123;

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

        // upper triangular version of virial tensor
        Scalar dihedral_virial[6];
        dihedral_virial[0]
            = Scalar(1. / 4.) * (-vb1.x * F0.x + vb2.x * F2.x + (vb3.x + vb2.x) * F3.x);
        dihedral_virial[1]
            = Scalar(1. / 4.) * (-vb1.y * F0.x + vb2.y * F2.x + (vb3.y + vb2.y) * F3.x);
        dihedral_virial[2]
            = Scalar(1. / 4.) * (-vb1.z * F0.x + vb2.z * F2.x + (vb3.z + vb2.z) * F3.x);
        dihedral_virial[3]
            = Scalar(1. / 4.) * (-vb1.y * F0.y + vb2.y * F2.y + (vb3.y + vb2.y) * F3.y);
        dihedral_virial[4]
            = Scalar(1. / 4.) * (-vb1.z * F0.y + vb2.z * F2.y + (vb3.z + vb2.z) * F3.y);
        dihedral_virial[5]
            = Scalar(1. / 4.) * (-vb1.z * F0.z + vb2.z * F2.z + (vb3.z + vb2.z) * F3.z);
        /**
        Snippet from hoomd's OPLS DihedralForceGPU.cu:
        Scalar3 vb1 = pos_a - pos_b;
        Scalar3 vb2 = pos_c - pos_b;
        Scalar3 vb3 = pos_d - pos_c;

        // Compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // upper triangular version of virial tensor
        Scalar dihedral_virial[6];
        dihedral_virial[0] = 0.25 * (vb1.x * f1.x + vb2.x * f3.x + (vb3.x + vb2.x) * f4.x);
        dihedral_virial[1] = 0.25 * (vb1.y * f1.x + vb2.y * f3.x + (vb3.y + vb2.y) * f4.x);
        dihedral_virial[2] = 0.25 * (vb1.z * f1.x + vb2.z * f3.x + (vb3.z + vb2.z) * f4.x);
        dihedral_virial[3] = 0.25 * (vb1.y * f1.y + vb2.y * f3.y + (vb3.y + vb2.y) * f4.y);
        dihedral_virial[4] = 0.25 * (vb1.z * f1.y + vb2.z * f3.y + (vb3.z + vb2.z) * f4.y);
        dihedral_virial[5] = 0.25 * (vb1.z * f1.z + vb2.z * f3.z + (vb3.z + vb2.z) * f4.z);
        */
        

        if (cur_dihedral_abcd == 0)
            {
            force_idx.x += F0.x;
            force_idx.y += F0.y;
            force_idx.z += F0.z;
            }
        if (cur_dihedral_abcd == 1)
            {
            force_idx.x += F1.x;
            force_idx.y += F1.y;
            force_idx.z += F1.z;
            }
        if (cur_dihedral_abcd == 2)
            {
            force_idx.x += F2.x;
            force_idx.y += F2.y;
            force_idx.z += F2.z;
            }
        if (cur_dihedral_abcd == 3)
            {
            force_idx.x += F3.x;
            force_idx.y += F3.y;
            force_idx.z += F3.z;
            }

        // Compute 1/4 of energy to assign to each of 4 atoms in the dihedral
        e_dihedral = 0.25 * prefactor * torsion_e;
        force_idx.w += e_dihedral;

        for (int k = 0; k < 6; k++)
            virial_idx[k] += dihedral_virial[k];
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
