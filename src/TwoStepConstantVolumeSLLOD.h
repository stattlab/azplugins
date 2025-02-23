// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef AZPLUGINS_TWOSTEPCONSTANTVOLUMESLLOD_H_
#define AZPLUGINS_TWOSTEPCONSTANTVOLUMESLLOD_H_


#include "hoomd/md/ComputeThermo.h"
#include "hoomd/md/IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"
#include "ThermostatMTTKSLLOD.h"


namespace hoomd
{
namespace azplugins
{

/** Perform constant volume simulation.

    Implement the the Velocity-Verlet integration scheme with an optional velocity rescaling
    Thermostat.
*/
class PYBIND11_EXPORT TwoStepConstantVolumeSLLOD : public md::IntegrationMethodTwoStep
    {
    public:
    /** Construct the constant volume integration method.

        @param sysdef System to work on.
        @param group Subset of particles to integrate.
        @param thermostat Thermostat to use. Set to null for constant energy simulations.
    */
    TwoStepConstantVolumeSLLOD(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group,
                          std::shared_ptr<MTTKThermostatSLLOD> thermostat);

    virtual ~TwoStepConstantVolumeSLLOD();

    /** Performs the first half-step of the integration.

        @param timestep Current simulation timestep.

        @post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per
        the Velocity-Verlet method.
    */
    virtual void integrateStepOne(uint64_t timestep);

    /** Performs the second half-step of the integration.

        @param timestep Current simulation timestep.

        @post Particle velocities are moved forward to timestep+1.
    */
    virtual void integrateStepTwo(uint64_t timestep);

    /** Set the thermostat.

        @param thermostat Thermostat to use for velocity rescaling (may be null).
    */
    void setThermostat(std::shared_ptr<MTTKThermostatSLLOD> thermostat)
        {
        m_thermostat = thermostat;
        }

    /** Set the distance limit applied to particles.

        @param limit Largest distance particles should move in a given timestep (may be null).

        When non-null, simulations do not conserve energy or momentum, but some users find the limit
        allows them to relax high energy initial conditions.
    */
    void setLimit(std::shared_ptr<Variant>& limit)
        {
        m_limit = limit;
        }

    /// Get the current limit.
    [[nodiscard]] auto getLimit() const
        {
        return m_limit;
        }

    /// Get needed pdata flags.
    virtual PDataFlags getRequestedPDataFlags()
        {
        PDataFlags flags;

        // Compute rotational kinetic energy in simulations with anisotropic degrees of freedom.
        if (m_aniso)
            {
            flags[pdata_flag::rotational_kinetic_energy] = 1;
            }
        return flags;
        }

    protected:
    /// Pack the limit values for use in the GPU kernel.
    auto getKernelLimitValues(uint64_t timestep)
        {
        const auto use_limit = static_cast<bool>(m_limit);
        Scalar maximum_displacement = use_limit ? m_limit->operator()(timestep) : 0.0;
        return std::make_pair(use_limit, maximum_displacement);
        }

    /// The thermostat to apply (may be null).
    std::shared_ptr<MTTKThermostatSLLOD> m_thermostat;

    /// The distance limit to apply (may be null).
    std::shared_ptr<Variant> m_limit;
    };


} // end namespace azplugins
} // end namespace hoomd

#endif // AZPLUGINS_TWOSTEPCONSTANTVOLUMESLLOD_H_
