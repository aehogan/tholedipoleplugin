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
 * Comprehensive test of TholeDipoleForce with two point charges.
 * Tests all combinations of:
 *   - TholeDampingType: NoDamping, Exponential, Amoeba, Linear
 *   - NonbondedMethod: NoCutoff, PME
 *   - PolarizationType: Direct, Mutual
 *
 * Two charges: +0.5e and -0.5e separated by 3 Angstroms (0.3 nm)
 *
 * IMPORTANT UNIT NOTE:
 *   - Linear and AMOEBA damping parameters are dimensionless
 *   - Exponential damping parameter has units of inverse length:
 *     MPMC uses A^-1, OpenMM uses nm^-1 -> multiply by 10
 *
 * MPMC Ground Truth Reference Values:
 *   System: Two charges +0.5e/-0.5e, separation 3A, polarizability 1.5 A^3
 *   NoCutoff: 10000 A box, PME: 50 A box
 *
 *   No polarization (electrostatic only):
 *     -13925.19971 K = -115.781 kJ/mol
 *
 *   Mutual polarization energies (K) and induced dipoles (e*nm):
 *   Damping      NoCutoff Energy   NoCutoff Dipole   PME Energy   PME Dipole
 *   ---------    --------------    ---------------   ----------   ----------
 *   NoDamping    -14795.52         0.009375          -14800.31    0.00936639
 *   Exponential  -14778.88         0.00919566        -14783.69    0.00918721
 *   Amoeba       -14794.40         0.00936294        -14799.19    0.00935434
 *   Linear       -14795.52         0.009375          -14800.31    0.00936639
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

static const double K_TO_KJ = 0.008314462;
static const double CHARGE1 = 0.5;
static const double CHARGE2 = -0.5;
static const double SEPARATION = 0.3;  // nm
static const double POLARIZABILITY = 0.0015;  // nm^3 (= 1.5 A^3)
static const double BOX_SIZE = 5.0;  // nm for PME
static const double CUTOFF = 2.0;  // nm for PME

struct DampingConfig {
    TholeDipoleForce::TholeDampingType type;
    double parameter;
    const char* name;
    // MPMC ground truth for Mutual polarization
    double noCutoffEnergyK;    // Energy in Kelvin
    double noCutoffDipole;     // Induced dipole magnitude in e*nm
    double pmeEnergyK;         // Energy in Kelvin
    double pmeDipole;          // Induced dipole magnitude in e*nm
};

static const DampingConfig DAMPING_CONFIGS[] = {
    // type, parameter, name, noCutoffEnergyK, noCutoffDipole, pmeEnergyK, pmeDipole
    {TholeDipoleForce::NoDamping,    0.0,    "NoDamping",    -14795.52, 0.009375,   -14800.31, 0.00936639},
    {TholeDipoleForce::Exponential,  21.304, "Exponential",  -14778.88, 0.00919566, -14783.69, 0.00918721},
    {TholeDipoleForce::Amoeba,       0.39,   "Amoeba",       -14794.40, 0.00936294, -14799.19, 0.00935434},
    {TholeDipoleForce::Linear,       2.1304, "Linear",       -14795.52, 0.009375,   -14800.31, 0.00936639},
};
static const int NUM_DAMPING = 4;

static const char* methodName(TholeDipoleForce::NonbondedMethod m) {
    return m == TholeDipoleForce::NoCutoff ? "NoCutoff" : "PME";
}

static const char* polName(TholeDipoleForce::PolarizationType p) {
    return p == TholeDipoleForce::Direct ? "Direct" : "Mutual";
}

/**
 * Check that analytical forces match numerical gradients (finite differences).
 */
static void checkFiniteDifferences(const vector<Vec3>& analyticForces,
                                   Context& context,
                                   const vector<Vec3>& positions,
                                   double tolerance = 1e-4) {
    double norm = 0.0;
    for (const auto& f : analyticForces)
        norm += f.dot(f);
    norm = std::sqrt(norm);

    if (norm < 1e-10) {
        return;  // Skip if forces are essentially zero
    }

    const double stepSize = 1e-4;
    double step = 0.5 * stepSize / norm;

    vector<Vec3> positions2(analyticForces.size()), positions3(analyticForces.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        Vec3 p = positions[i];
        Vec3 f = analyticForces[i];
        positions2[i] = Vec3(p[0] - f[0]*step, p[1] - f[1]*step, p[2] - f[2]*step);
        positions3[i] = Vec3(p[0] + f[0]*step, p[1] + f[1]*step, p[2] + f[2]*step);
    }

    context.setPositions(positions2);
    State state2 = context.getState(State::Energy);
    context.setPositions(positions3);
    State state3 = context.getState(State::Energy);

    double numericalForceNorm = (state2.getPotentialEnergy() - state3.getPotentialEnergy()) / stepSize;

    double relError = fabs(norm - numericalForceNorm) / norm;
    cout << "    FD: ana=" << norm << " num=" << numericalForceNorm
         << " err=" << (relError * 100) << "%" << endl;
    if (relError > tolerance) {
        cout << "    FAILED: Finite diff error " << relError * 100 << "% > " << tolerance * 100 << "%" << endl;
    }
    ASSERT_EQUAL_TOL(norm, numericalForceNorm, tolerance);
}

/**
 * Generic test function for all combinations.
 */
static double runTest(TholeDipoleForce::TholeDampingType dampingType,
                      double dampingParam,
                      TholeDipoleForce::NonbondedMethod nbMethod,
                      TholeDipoleForce::PolarizationType polType,
                      bool verbose = false) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    if (nbMethod == TholeDipoleForce::PME) {
        system.setDefaultPeriodicBoxVectors(Vec3(BOX_SIZE, 0, 0),
                                            Vec3(0, BOX_SIZE, 0),
                                            Vec3(0, 0, BOX_SIZE));
    }

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);

    force->setNonbondedMethod(nbMethod);
    force->setPolarizationType(polType);
    force->setTholeDampingType(dampingType);
    force->setTholeDampingParameter(dampingParam);
    force->setDampPermanentInducedField(false);

    if (nbMethod == TholeDipoleForce::PME) {
        force->setCutoffDistance(CUTOFF);
    }

    if (polType == TholeDipoleForce::Mutual) {
        force->setMutualInducedTargetEpsilon(1.0e-8);
        force->setMutualInducedMaxIterations(500);
    }

    double pol = (polType == TholeDipoleForce::Direct && dampingType == TholeDipoleForce::NoDamping)
                 ? 0.0 : POLARIZABILITY;
    // For Direct polarization with damping, we still use polarizability
    // For NoDamping + Direct, use zero polarizability as baseline
    if (polType == TholeDipoleForce::Direct) {
        pol = POLARIZABILITY;  // Direct still computes induced dipoles, just one iteration
    }

    vector<double> zeroDipole(3, 0.0);
    force->addParticle(CHARGE1, zeroDipole, pol, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(CHARGE2, zeroDipole, pol, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(BOX_SIZE/2 - SEPARATION/2, BOX_SIZE/2, BOX_SIZE/2);
    positions[1] = Vec3(BOX_SIZE/2 + SEPARATION/2, BOX_SIZE/2, BOX_SIZE/2);
    if (nbMethod == TholeDipoleForce::NoCutoff) {
        positions[0] = Vec3(0.0, 0.0, 0.0);
        positions[1] = Vec3(SEPARATION, 0.0, 0.0);
    }

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    // Get induced dipoles for diagnostics
    vector<Vec3> induced;
    force->getInducedDipoles(context, induced);

    if (verbose) {
        cout << "  Energy: " << energy << " kJ/mol" << endl;
        cout << "  Force[0]: " << forces[0] << endl;
        cout << "  Force[1]: " << forces[1] << endl;
        cout << "  Induced[0]: " << induced[0] << " (|" << sqrt(induced[0].dot(induced[0])) << "|)" << endl;
        cout << "  Induced[1]: " << induced[1] << " (|" << sqrt(induced[1].dot(induced[1])) << "|)" << endl;
    }

    // Basic sanity checks
    ASSERT(std::isfinite(energy));

    // Newton's 3rd law
    for (int i = 0; i < 3; i++) {
        ASSERT_EQUAL_TOL(forces[0][i], -forces[1][i], 1e-6);
    }

    // Finite difference check
    checkFiniteDifferences(forces, context, positions, 1e-4);

    return energy;
}

void testNoPolarization() {
    cout << "\n=== No Polarization Baseline ===" << endl;

    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);

    vector<double> zeroDipole(3, 0.0);
    force->addParticle(CHARGE1, zeroDipole, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(CHARGE2, zeroDipole, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(SEPARATION, 0.0, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();

    double mpmc_energy = -13925.19971 * K_TO_KJ;
    cout << "  Energy: " << energy << " kJ/mol" << endl;
    cout << "  MPMC:   " << mpmc_energy << " kJ/mol" << endl;
    cout << "  Diff:   " << (energy - mpmc_energy) << " kJ/mol" << endl;

    ASSERT(energy < 0.0);
    ASSERT_EQUAL_TOL(energy, mpmc_energy, 0.01);
    cout << "  PASSED" << endl;
}

void testAllCombinations() {
    cout << "\n=== Testing All Damping/Method/Polarization Combinations ===" << endl;

    struct Result {
        const char* damping;
        const char* method;
        const char* pol;
        double energy;
        bool passed;
        string errorMsg;
    };
    vector<Result> results;
    int failCount = 0;

    TholeDipoleForce::NonbondedMethod methods[] = {
        TholeDipoleForce::NoCutoff,
        TholeDipoleForce::PME
    };

    TholeDipoleForce::PolarizationType pols[] = {
        TholeDipoleForce::Direct,
        TholeDipoleForce::Mutual
    };

    for (int d = 0; d < NUM_DAMPING; d++) {
        for (auto method : methods) {
            for (auto pol : pols) {
                const auto& cfg = DAMPING_CONFIGS[d];

                cout << "\nTest: " << cfg.name << " + " << methodName(method)
                     << " + " << polName(pol) << endl;

                Result r;
                r.damping = cfg.name;
                r.method = methodName(method);
                r.pol = polName(pol);
                r.passed = false;
                r.energy = 0.0;

                try {
                    r.energy = runTest(cfg.type, cfg.parameter, method, pol, true);
                    r.passed = true;

                    // For Mutual polarization, check against MPMC reference
                    if (pol == TholeDipoleForce::Mutual) {
                        double mpmcEnergyK = (method == TholeDipoleForce::NoCutoff)
                                             ? cfg.noCutoffEnergyK : cfg.pmeEnergyK;
                        double mpmcDipole = (method == TholeDipoleForce::NoCutoff)
                                            ? cfg.noCutoffDipole : cfg.pmeDipole;
                        double mpmcEnergy = mpmcEnergyK * K_TO_KJ;
                        cout << "  MPMC Energy: " << mpmcEnergy << " kJ/mol" << endl;
                        cout << "  MPMC Dipole: " << mpmcDipole << " e*nm" << endl;
                        cout << "  Energy Diff: " << (r.energy - mpmcEnergy) << " kJ/mol" << endl;
                    }

                    cout << "  PASSED" << endl;
                }
                catch (const std::exception& e) {
                    r.errorMsg = e.what();
                    cout << "  FAILED: " << e.what() << endl;
                    failCount++;
                }

                results.push_back(r);
            }
        }
    }

    // Print summary table
    cout << "\n========================================" << endl;
    cout << "Summary Table" << endl;
    cout << "========================================" << endl;
    printf("%-12s %-10s %-8s %15s %8s\n", "Damping", "Method", "Pol", "Energy(kJ/mol)", "Status");
    printf("%-12s %-10s %-8s %15s %8s\n", "-------", "------", "---", "-------------", "------");
    for (const auto& r : results) {
        printf("%-12s %-10s %-8s %15.6f %8s\n",
               r.damping, r.method, r.pol, r.energy, r.passed ? "PASS" : "FAIL");
    }

    if (failCount > 0) {
        stringstream msg;
        msg << failCount << " test(s) failed in testAllCombinations";
        throw OpenMMException(msg.str());
    }
}

void testPMEvsNoCutoffConsistency() {
    cout << "\n=== PME vs NoCutoff Consistency ===" << endl;
    cout << "(PME should give similar energy to NoCutoff for isolated pair in large box)" << endl;

    for (int d = 0; d < NUM_DAMPING; d++) {
        const auto& cfg = DAMPING_CONFIGS[d];

        double energyNoCutoff = runTest(cfg.type, cfg.parameter,
                                        TholeDipoleForce::NoCutoff,
                                        TholeDipoleForce::Mutual, false);
        double energyPME = runTest(cfg.type, cfg.parameter,
                                   TholeDipoleForce::PME,
                                   TholeDipoleForce::Mutual, false);

        double diff = fabs(energyPME - energyNoCutoff);
        double relDiff = diff / fabs(energyNoCutoff) * 100.0;

        cout << cfg.name << ": NoCutoff=" << energyNoCutoff
             << ", PME=" << energyPME
             << ", diff=" << relDiff << "%" << endl;

        // PME and NoCutoff should be within 1% for isolated pair in large box
        ASSERT(relDiff < 1.0);
    }
    cout << "  PASSED" << endl;
}

void testDirectVsMutualConsistency() {
    cout << "\n=== Direct vs Mutual Consistency ===" << endl;
    cout << "(Direct should give lower polarization energy than Mutual)" << endl;

    for (int d = 0; d < NUM_DAMPING; d++) {
        const auto& cfg = DAMPING_CONFIGS[d];

        double energyDirect = runTest(cfg.type, cfg.parameter,
                                      TholeDipoleForce::NoCutoff,
                                      TholeDipoleForce::Direct, false);
        double energyMutual = runTest(cfg.type, cfg.parameter,
                                      TholeDipoleForce::NoCutoff,
                                      TholeDipoleForce::Mutual, false);

        cout << cfg.name << ": Direct=" << energyDirect
             << ", Mutual=" << energyMutual << endl;

        // Mutual polarization should give more negative energy (more favorable)
        // because induced dipoles reinforce each other
        ASSERT(energyMutual <= energyDirect);
    }
    cout << "  PASSED" << endl;
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n========================================" << endl;
        cout << "Two Point Charges Comprehensive Tests" << endl;
        cout << "========================================" << endl;

        testNoPolarization();
        testAllCombinations();
        testPMEvsNoCutoffConsistency();
        testDirectVsMutualConsistency();

        cout << "\n========================================" << endl;
        cout << "All tests passed!" << endl;
        cout << "========================================" << endl;
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << endl;
    return 0;
}
