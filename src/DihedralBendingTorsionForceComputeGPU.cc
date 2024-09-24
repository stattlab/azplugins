// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*! \file DihedralBendingTorsionForceComputeGPU.cc
    \brief Defines DihedralBendingTorsionForceComputeGPU
*/

#include "DihedralBendingTorsionForceComputeGPU.h"

using namespace std;

namespace hoomd
    {
namespace azplugins
    {
/*! \param sysdef System to compute bond forces on
 */
DihedralBendingTorsionForceComputeGPU::DihedralBendingTorsionForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : DihedralBendingTorsionForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating an DihedralBendingTorsionForceComputeGPU with no GPU in execution configuration"
            << endl;
        throw std::runtime_error("Error initializing DihedralBendingTorsionForceComputeGPU");
        }

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "bending_torsion_dihedral"));
    m_autotuners.push_back(m_tuner);
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_bending_torsion_dihedral_forces to do the dirty work.
*/
void DihedralBendingTorsionForceComputeGPU::computeForces(uint64_t timestep)
    {
    ArrayHandle<DihedralData::members_t> d_gpu_dihedral_list(m_dihedral_data->getGPUTable(),
                                                             access_location::device,
                                                             access_mode::read);
    ArrayHandle<unsigned int> d_n_dihedrals(m_dihedral_data->getNGroupsArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_dihedrals_ABCD(m_dihedral_data->getGPUPosTable(),
                                               access_location::device,
                                               access_mode::read);

    // the dihedral table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<dihedral_bending_torsion_params> d_params(m_params, access_location::device, access_mode::read);

    // run the kernel in parallel on all GPUs
    m_tuner->begin();
    gpu::gpu_compute_bending_torsion_dihedral_forces(d_force.data,
                                             d_virial.data,
                                             m_virial.getPitch(),
                                             m_pdata->getN(),
                                             d_pos.data,
                                             box,
                                             d_gpu_dihedral_list.data,
                                             d_dihedrals_ABCD.data,
                                             m_dihedral_data->getGPUTableIndexer().getW(),
                                             d_n_dihedrals.data,
                                             d_params.data,
                                             m_dihedral_data->getNTypes(),
                                             m_tuner->getParam()[0],
                                             m_exec_conf->dev_prop.warpSize);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_DihedralBendingTorsionForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<DihedralBendingTorsionForceComputeGPU,
                     DihedralBendingTorsionForceCompute,
                     std::shared_ptr<DihedralBendingTorsionForceComputeGPU>>(m, "DihedralBendingTorsionForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd
