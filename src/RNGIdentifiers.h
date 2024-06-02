// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2024, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file RNGIdentifiers.h
 * \brief Identifiers for random123 generator.
 */

#ifndef AZPLUGINS_RNG_IDENTIFIERS_H_
#define AZPLUGINS_RNG_IDENTIFIERS_H_

namespace azplugins
    {

struct RNGIdentifier
    {
    // hoomd's identifiers, changed by +/- 1
    static const uint32_t DPDEvaluatorGeneralWeight = 0x4a84f5d1;
    static const uint32_t TwoStepBrownianFlow = 0x431287fe;
    static const uint32_t TwoStepLangevinFlow = 0x89abcdee;
    static const uint32_t ParticleEvaporator = 0x3eb8536f;
    };

    } // end namespace azplugins

#endif // AZPLUGINS_RNG_IDENTIFIERS_H_
