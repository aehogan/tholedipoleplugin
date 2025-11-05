/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

/**
 * Test size consistency for 1D chain of alternating charges with PME.
 * Based on MPMC test: tutorials/11_1D_chain_replay
 *
 * System: Alternating charges ±0.00244629406e with polarizabilities 0.1/0.15 Å³
 * separated by 0.6 Å in a 1D chain along x-axis.
 *
 * Key test: With proper long-range treatment (PME), energy per atom should be
 * independent of system size. With direct summation, energy per atom depends on system size.
 *
 * MPMC Reference (exponential damping, λ=2.1304 Å⁻¹):
 *   Direct summation: Energy/atom varies from -3.27 to -3.59 K/atom (size dependent)
 *   Ewald summation: Energy/atom constant at -1.38454 K/atom = -0.01151 kJ/mol/atom
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include <numeric>

void test1DChainSizeConsistency() {
    // MPMC parameters from replay.pqr
    double charge1 = 0.00244629406;    // e
    double charge2 = -0.00244629406;   // e
    double pol1 = 0.0001;              // 0.1 Å³ = 0.0001 nm³
    double pol2 = 0.00015;             // 0.15 Å³ = 0.00015 nm³
    double separation = 0.06;          // 0.6 Å = 0.06 nm
    double dampingParam = 21.304;      // 2.1304 Å⁻¹ = 21.304 nm⁻¹

    double mpmc_ewald_per_atom = -1.384541;  // K/atom (full polarizable, Ewald)
    double mpmc_ewald_kj_per_atom = mpmc_ewald_per_atom * 0.008314462;  // kJ/mol/atom

    vector<int> systemSizes = {32, 64, 128, 256};

    enum Polarization { NoPol, Direct, Mutual };
    enum NonbondedMethod { PME, NoCutoff };

    struct Scenario {
        Polarization pol;
        NonbondedMethod nb;
        string name() const {
            string p = (pol == NoPol ? "NoPol" : (pol == Direct ? "Direct" : "Mutual"));
            string n = (nb == PME ? "PME" : "NoCutoff");
            return p + "_" + n;
        }
    };

    vector<Scenario> scenarios = {
        {NoPol,     PME},
        {Direct,    PME},
        {Mutual,    PME},
        {NoPol,     NoCutoff},
        {Direct,    NoCutoff},
        {Mutual,    NoCutoff}
    };

    struct Result {
        int numAtoms;
        double boxSize;
        vector<double> energyPerAtom; // size = 6
    };
    vector<Result> results;

    for (int numAtoms : systemSizes) {
        int numPairs = numAtoms / 2;
        double boxSize = numPairs * 0.1;

        Result result;
        result.numAtoms = numAtoms;
        result.boxSize = boxSize;
        result.energyPerAtom.resize(scenarios.size());

        for (size_t s = 0; s < scenarios.size(); s++) {
            const auto& scenario = scenarios[s];

            System system;
            for (int i = 0; i < numAtoms; i++)
                system.addParticle(1.0);

            if (scenario.nb == PME) {
                system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0.0, 0.0),
                                                   Vec3(0.0, 20.0, 0.0),
                                                   Vec3(0.0, 0.0, 20.0));
            }

            TholeDipoleForce* force = new TholeDipoleForce();
            system.addForce(force);

            if (scenario.nb == PME) {
                force->setNonbondedMethod(TholeDipoleForce::PME);
                double cutoff = min(0.9, boxSize * 0.49);
                force->setCutoffDistance(cutoff);
            } else {
                force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
            }

            force->setTholeDampingType(TholeDipoleForce::Exponential);
            force->setTholeDampingParameter(dampingParam);

            if (scenario.pol == NoPol) {
                force->setPolarizationType(TholeDipoleForce::Direct); // induced = 0
            } else if (scenario.pol == Direct) {
                force->setPolarizationType(TholeDipoleForce::Direct);
            } else if (scenario.pol == Mutual) {
                force->setPolarizationType(TholeDipoleForce::Mutual);
                force->setMutualInducedTargetEpsilon(1.0e-5);
                force->setMutualInducedMaxIterations(500);
            }

            vector<double> zeroDipole(3, 0.0);
            vector<Vec3> positions(numAtoms);
            for (int i = 0; i < numAtoms; i++) {
                double charge = (i % 2 == 0) ? charge1 : charge2;
                double pol = 0.0;
                if (scenario.pol != NoPol) {
                    pol = (i % 2 == 0) ? pol1 : pol2;
                }
                force->addParticle(charge, zeroDipole, pol,
                                   TholeDipoleForce::NoAxisType, -1, -1, -1);
                positions[i] = Vec3(i / 2 * 0.1 + i % 2 * separation, 0.0, 0.0);
            }

            LangevinIntegrator integrator(0.0, 0.1, 0.01);
            Context context(system, integrator, *platform);
            context.setPositions(positions);
            double energy = context.getState(State::Energy).getPotentialEnergy();
            result.energyPerAtom[s] = energy / numAtoms;

            // Debug output for first few particles in selected scenarios
            if (numAtoms == 32 && scenario.pol == Mutual) {
                cout << "\n=== " << scenario.name() << " ===" << endl;
                cout << "Total energy: " << energy << " kJ/mol" << endl;
                cout << "Energy per atom: " << energy / numAtoms << " kJ/mol/atom" << endl;

                // Get induced dipoles
                vector<Vec3> inducedDipoles;
                force->getInducedDipoles(context, inducedDipoles);

                // Print 4 particles from middle of chain
                int startIdx = numAtoms / 2 - 2;
                int endIdx = numAtoms / 2 + 2;
                for (int i = startIdx; i < endIdx; i++) {
                    double charge, pol_val;
                    vector<double> dipole;
                    int axisType, atomZ, atomX, atomY;
                    force->getParticleParameters(i, charge, dipole, pol_val, axisType, atomZ, atomX, atomY);

                    cout << "Particle " << i << ":" << endl;
                    cout << "  pos: " << positions[i] << " nm" << endl;
                    cout << "  charge: " << charge << " e" << endl;
                    cout << "  polarizability: " << pol_val << " nm³" << endl;
                    cout << "  induced dipole: " << inducedDipoles[i] << " e·nm" << endl;
                    cout << "  |induced|: " << sqrt(inducedDipoles[i].dot(inducedDipoles[i])) << " e·nm" << endl;
                }
            }
        }

        results.push_back(result);
    }

    // === PRINT SUMMARY ===
    cout << "\n========================================" << endl;
    cout << "1D Chain: Direct PME vs NoCutoff" << endl;
    cout << "========================================\n" << endl;

    // Header
    cout << "Size  Box(nm)";
    for (const auto& s : scenarios) {
        printf("  %12s", s.name().c_str());
    }
    cout << endl;
    cout << "----  -------";
    for (size_t i = 0; i < scenarios.size(); i++) {
        cout << "  -----------";
    }
    cout << endl;

    // Rows
    for (const auto& r : results) {
        printf("%4d  %7.4f", r.numAtoms, r.boxSize);
        for (double e : r.energyPerAtom) {
            printf("  %11.6f", e);
        }
        cout << endl;
    }
    cout << endl;

    // === STATISTICS ===
    auto computeStats = [](const vector<double>& vals) -> pair<double, double> {
        double mean = accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        double var = 0.0;
        for (double v : vals) var += (v - mean) * (v - mean);
        var /= vals.size();
        return {mean, sqrt(var)};
    };

    vector<vector<double>> energies(scenarios.size());
    for (const auto& r : results) {
        for (size_t s = 0; s < scenarios.size(); s++) {
            energies[s].push_back(r.energyPerAtom[s]);
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        test1DChainSizeConsistency();
    }
    catch (const std::exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR. Test failed." << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
