// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ThermostatMTTKSLLOD.h"
#include "ComputeThermoSLLOD.h"

namespace hoomd
{
namespace azplugins
{
namespace detail
{

void export_Thermostat2(pybind11::module& m)
{

pybind11::class_<Thermostat2, std::shared_ptr<Thermostat2>>(m, "Thermostat2")
    .def(pybind11::init<std::shared_ptr<Variant>,
                        std::shared_ptr<ParticleGroup>,
                        std::shared_ptr<ComputeThermoSLLOD>,
                        std::shared_ptr<SystemDefinition>>())
    .def_property("kT", &Thermostat2::getT, &Thermostat2::setT);
}

void export_MTTKThermostatSLLOD(pybind11::module& m)
    {

    pybind11::class_<MTTKThermostatSLLOD, Thermostat2, std::shared_ptr<MTTKThermostatSLLOD>>(m,
                                                                                  "MTTKThermostatSLLOD")
        .def(pybind11::init<std::shared_ptr<Variant>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermoSLLOD>,
                            std::shared_ptr<SystemDefinition>,
                            Scalar>())
        .def_property("translational_dof",
                      &MTTKThermostatSLLOD::getTranslationalDOF,
                      &MTTKThermostatSLLOD::setTranslationalDOF)
        .def_property("rotational_dof",
                      &MTTKThermostatSLLOD::getRotationalDOF,
                      &MTTKThermostatSLLOD::setRotationalDOF)
        .def_property("tau", &MTTKThermostatSLLOD::getTau, &MTTKThermostatSLLOD::setTau)
        .def("getThermostatEnergy", &MTTKThermostatSLLOD::getThermostatEnergy)
        .def("thermalizeThermostat", &MTTKThermostatSLLOD::thermalizeThermostat);
    }


} // end amespace detail
} // end amespace azplugins
} // end amespace hoomd