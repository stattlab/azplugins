// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_TWOSTEPCONSTANTVOLUMESLLODGPU_H_
#define AZPLUGINS_TWOSTEPCONSTANTVOLUMESLLODGPU_H_

#include "hoomd/md/Thermostat.h"
#include "hoomd/md/TwoStepConstantVolume.h"
#include <hoomd/Autotuner.h>


namespace hoomd
{
namespace azplugins
{
/// Implement TwoStepConstantVolume on the GPU.
class PYBIND11_EXPORT TwoStepConstantVolumeSLLODGPU : public TwoStepConstantVolumeSLLOD
    {
    public:
    TwoStepConstantVolumeSLLODGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<ThermostatMTTKSLLOD> thermostat,
                             Scalar shear_rate);

    virtual ~TwoStepConstantVolumeSLLODGPU() { }

    virtual void integrateStepOne(uint64_t timestep);

    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    /// Autotuner for block size (step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_one;

    /// Autotuner for block size (step two kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_two;

    /// Autotuner_angular for block size (angular step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_one;

    /// Autotuner_angular for block size (angular step two kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_two;
    };
} // end namespace azplugins
} // namespace hoomd

#endif // AZPLUGINS_TWOSTEPCONSTANTVOLUMESLLODGPU_H_
