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

        // the three bonds

        Scalar3 vb1 = pos_a - pos_b;
        Scalar3 vb2 = pos_c - pos_b;
        Scalar3 vb3 = pos_d - pos_c;

        // apply periodic boundary conditions
        vb1 = box.minImage(vb1);
        vb2 = box.minImage(vb2);
        vb3 = box.minImage(vb3);

        Scalar3 vb2m = -vb2;
        vb2m = box.minImage(vb2m);

        // c,s calculation

        Scalar ax, ay, az, bx, by, bz;
        ax = vb1.y * vb2m.z - vb1.z * vb2m.y;
        ay = vb1.z * vb2m.x - vb1.x * vb2m.z;
        az = vb1.x * vb2m.y - vb1.y * vb2m.x;
        bx = vb3.y * vb2m.z - vb3.z * vb2m.y;
        by = vb3.z * vb2m.x - vb3.x * vb2m.z;
        bz = vb3.x * vb2m.y - vb3.y * vb2m.x;

        Scalar rasq = ax * ax + ay * ay + az * az;
        Scalar rbsq = bx * bx + by * by + bz * bz;
        Scalar rgsq = vb2m.x * vb2m.x + vb2m.y * vb2m.y + vb2m.z * vb2m.z;
        Scalar rg = fast::sqrt(rgsq);

        Scalar rginv, ra2inv, rb2inv;
        rginv = ra2inv = rb2inv = 0.0;
        if (rg > 0)
            rginv = 1.0 / rg;
        if (rasq > 0)
            ra2inv = 1.0 / rasq;
        if (rbsq > 0)
            rb2inv = 1.0 / rbsq;
        Scalar rabinv = fast::sqrt(ra2inv * rb2inv);

        Scalar c = (ax * bx + ay * by + az * bz) * rabinv;
        Scalar s = rg * rabinv * (ax * vb3.x + ay * vb3.y + az * vb3.z);

        if (c > 1.0)
            c = 1.0;
        if (c < -1.0)
            c = -1.0;

        // get values for k1/2 through k4/2 (MEM TRANSFER: 16 bytes)
        // ----- The 1/2 factor is already stored in the parameters --------
        /**
         * 
        dihedral_bending_torsion_params params = __ldg(d_params + cur_dihedral_type);
        Scalar k_phi = h_params.data[dihedral_type].k_phi;
        Scalar a0 = h_params.data[dihedral_type].a0;
        Scalar a1 = h_params.data[dihedral_type].a1;
        Scalar a2 = h_params.data[dihedral_type].a2;
        Scalar a3 = h_params.data[dihedral_type].a3;
        Scalar a4 = h_params.data[dihedral_type].a4;
         */
        dihedral_bending_torsion_params params = d_params[cur_dihedral_type];
        Scalar k_phi = params.k_phi;
        Scalar a0 = params.a0;
        Scalar a1 = params.a1;
        Scalar a2 = params.a2;
        Scalar a3 = params.a3;
        Scalar a4 = params.a4;

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
        */
        }

    // // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    // d_force[idx] = force_idx;
    // for (int k = 0; k < 6; k++)
    //     d_virial[k * virial_pitch + idx] = virial_idx[k];
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
