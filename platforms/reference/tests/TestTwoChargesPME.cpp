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
 * Test PME with two polarizable point charges.
 * Compares PME energy/forces with NoCutoff reference.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testTwoChargesPME() {
    double charge1 = 0.5;
    double charge2 = -0.5;
    double polarizability = 0.0015;  // 1.5 Å³ = 0.0015 nm³
    double separation = 0.3;         // 3 Å = 0.3 nm
    double boxSize = 2.0;            // nm

    // Permanent dipoles - z-only dipoles with ZOnly axis type
    // Particle 0: dipole along +z (0.01 e·nm), references particle 1
    // Particle 1: dipole along +z (0.008 e·nm), references particle 0
    vector<double> dipole1 = {0.0, 0.0, 0.01};
    vector<double> dipole2 = {0.0, 0.0, 0.008};

    vector<Vec3> positions(2);
    positions[0] = Vec3(0.5, 1.0, 1.0);
    positions[1] = Vec3(0.5 + separation, 1.0, 1.0);

    // NoCutoff reference
    System systemNoCutoff;
    systemNoCutoff.addParticle(1.0);
    systemNoCutoff.addParticle(1.0);

    TholeDipoleForce* forceNoCutoff = new TholeDipoleForce();
    systemNoCutoff.addForce(forceNoCutoff);
    forceNoCutoff->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    forceNoCutoff->setPolarizationType(TholeDipoleForce::Direct);
    forceNoCutoff->setTholeDampingType(TholeDipoleForce::Amoeba);
    forceNoCutoff->setTholeDampingParameter(0.39);

    // Use ZOnly axis type - each particle references the other for Z-axis
    forceNoCutoff->addParticle(charge1, dipole1, polarizability,
                               TholeDipoleForce::ZOnly, 1, -1, -1);
    forceNoCutoff->addParticle(charge2, dipole2, polarizability,
                               TholeDipoleForce::ZOnly, 0, -1, -1);

    LangevinIntegrator integratorNoCutoff(0.0, 0.1, 0.01);
    Context contextNoCutoff(systemNoCutoff, integratorNoCutoff, *platform);
    contextNoCutoff.setPositions(positions);

    State stateNoCutoff = contextNoCutoff.getState(State::Forces | State::Energy);
    double energyNoCutoff = stateNoCutoff.getPotentialEnergy();
    const vector<Vec3>& forcesNoCutoff = stateNoCutoff.getForces();

    vector<Vec3> inducedNoCutoff;
    forceNoCutoff->getInducedDipoles(contextNoCutoff, inducedNoCutoff);

    vector<Vec3> labDipolesNoCutoff;
    forceNoCutoff->getLabFramePermanentDipoles(contextNoCutoff, labDipolesNoCutoff);

    cout << "=== NoCutoff Reference ===" << endl;
    cout << "Energy: " << energyNoCutoff << " kJ/mol" << endl;
    cout << "Force[0]: " << forcesNoCutoff[0] << " kJ/mol/nm" << endl;
    cout << "Force[1]: " << forcesNoCutoff[1] << " kJ/mol/nm" << endl;
    cout << "Induced[0]: " << inducedNoCutoff[0] << " e·nm" << endl;
    cout << "Induced[1]: " << inducedNoCutoff[1] << " e·nm" << endl;
    cout << "Lab Perm Dipole[0]: " << labDipolesNoCutoff[0] << " e·nm" << endl;
    cout << "Lab Perm Dipole[1]: " << labDipolesNoCutoff[1] << " e·nm" << endl;
    cout << endl;

    // PME
    System systemPME;
    systemPME.addParticle(1.0);
    systemPME.addParticle(1.0);
    systemPME.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0.0, 0.0),
                                           Vec3(0.0, boxSize, 0.0),
                                           Vec3(0.0, 0.0, boxSize));

    TholeDipoleForce* forcePME = new TholeDipoleForce();
    systemPME.addForce(forcePME);
    forcePME->setNonbondedMethod(TholeDipoleForce::PME);
    forcePME->setCutoffDistance(0.9);
    forcePME->setPolarizationType(TholeDipoleForce::Direct);
    forcePME->setTholeDampingType(TholeDipoleForce::Amoeba);
    forcePME->setTholeDampingParameter(0.39);

    forcePME->addParticle(charge1, dipole1, polarizability,
                          TholeDipoleForce::ZOnly, 1, -1, -1);
    forcePME->addParticle(charge2, dipole2, polarizability,
                          TholeDipoleForce::ZOnly, 0, -1, -1);

    LangevinIntegrator integratorPME(0.0, 0.1, 0.01);
    Context contextPME(systemPME, integratorPME, *platform);
    contextPME.setPositions(positions);

    State statePME = contextPME.getState(State::Forces | State::Energy);
    double energyPME = statePME.getPotentialEnergy();
    const vector<Vec3>& forcesPME = statePME.getForces();

    vector<Vec3> inducedPME;
    forcePME->getInducedDipoles(contextPME, inducedPME);

    cout << "=== PME ===" << endl;
    cout << "Energy: " << energyPME << " kJ/mol" << endl;
    cout << "Force[0]: " << forcesPME[0] << " kJ/mol/nm" << endl;
    cout << "Force[1]: " << forcesPME[1] << " kJ/mol/nm" << endl;
    cout << "Induced[0]: " << inducedPME[0] << " e·nm" << endl;
    cout << "Induced[1]: " << inducedPME[1] << " e·nm" << endl;
    cout << endl;

    // Comparison
    cout << "=== Comparison ===" << endl;
    double energyDiff = energyPME - energyNoCutoff;
    cout << "Energy difference: " << energyDiff << " kJ/mol" << endl;

    double forceDiff0 = sqrt((forcesPME[0] - forcesNoCutoff[0]).dot(forcesPME[0] - forcesNoCutoff[0]));
    double forceDiff1 = sqrt((forcesPME[1] - forcesNoCutoff[1]).dot(forcesPME[1] - forcesNoCutoff[1]));
    cout << "Force[0] difference: " << forceDiff0 << " kJ/mol/nm" << endl;
    cout << "Force[1] difference: " << forceDiff1 << " kJ/mol/nm" << endl;

    double indDiff0 = sqrt((inducedPME[0] - inducedNoCutoff[0]).dot(inducedPME[0] - inducedNoCutoff[0]));
    double indDiff1 = sqrt((inducedPME[1] - inducedNoCutoff[1]).dot(inducedPME[1] - inducedNoCutoff[1]));
    cout << "Induced[0] difference: " << indDiff0 << " e·nm" << endl;
    cout << "Induced[1] difference: " << indDiff1 << " e·nm" << endl;

    // Newton's 3rd law check for both
    cout << endl << "=== Newton's 3rd Law Check ===" << endl;
    Vec3 sumNoCutoff = forcesNoCutoff[0] + forcesNoCutoff[1];
    Vec3 sumPME = forcesPME[0] + forcesPME[1];
    cout << "NoCutoff sum of forces: " << sumNoCutoff << endl;
    cout << "PME sum of forces: " << sumPME << endl;

    // Finite difference checks
    double delta = 1e-5;

    cout << endl << "=== Finite Difference Check (NoCutoff) ===" << endl;
    cout << "Particle  Dim  Analytical    Numerical     Difference" << endl;
    cout << "----------------------------------------------------" << endl;
    for (int p = 0; p < 2; p++) {
        for (int d = 0; d < 3; d++) {
            vector<Vec3> posPlus = positions;
            vector<Vec3> posMinus = positions;
            posPlus[p][d] += delta;
            posMinus[p][d] -= delta;

            contextNoCutoff.setPositions(posPlus);
            double ePlus = contextNoCutoff.getState(State::Energy).getPotentialEnergy();
            contextNoCutoff.setPositions(posMinus);
            double eMinus = contextNoCutoff.getState(State::Energy).getPotentialEnergy();

            double numericalForce = -(ePlus - eMinus) / (2 * delta);
            double analyticalForce = forcesNoCutoff[p][d];
            double diff = analyticalForce - numericalForce;

            char dimChar = "xyz"[d];
            printf("   %d      %c    %10.4f    %10.4f    %10.4f\n",
                   p, dimChar, analyticalForce, numericalForce, diff);
        }
    }
    contextNoCutoff.setPositions(positions);

    // Finite difference check for PME - all components
    cout << endl << "=== Finite Difference Check (PME) ===" << endl;

    // Check all 6 force components (2 particles x 3 dimensions)
    cout << "Particle  Dim  Analytical    Numerical     Difference" << endl;
    cout << "----------------------------------------------------" << endl;

    double maxDiff = 0.0;
    for (int p = 0; p < 2; p++) {
        for (int d = 0; d < 3; d++) {
            vector<Vec3> posPlus = positions;
            vector<Vec3> posMinus = positions;
            posPlus[p][d] += delta;
            posMinus[p][d] -= delta;

            contextPME.setPositions(posPlus);
            double ePlus = contextPME.getState(State::Energy).getPotentialEnergy();
            contextPME.setPositions(posMinus);
            double eMinus = contextPME.getState(State::Energy).getPotentialEnergy();

            double numericalForce = -(ePlus - eMinus) / (2 * delta);
            double analyticalForce = forcesPME[p][d];
            double diff = analyticalForce - numericalForce;

            char dimChar = "xyz"[d];
            printf("   %d      %c    %10.4f    %10.4f    %10.4f\n",
                   p, dimChar, analyticalForce, numericalForce, diff);

            // Print energy breakdown for x-direction on particle 0
            if (p == 0 && d == 0) {
                printf("       E(center)=%14.8f  E(+)=%14.8f  E(-)=%14.8f\n",
                       energyPME, ePlus, eMinus);
                printf("       dE/dx numeric = (E(+)-E(-))/2delta = %.8f\n",
                       (ePlus - eMinus) / (2 * delta));
                printf("       Delta_E = E(+)-E(-) = %.10f kJ/mol\n", ePlus - eMinus);
                printf("       Expected from force: -F*2*delta = %.10f kJ/mol\n", -analyticalForce * 2 * delta);

                // Check induced dipoles at +/- delta
                vector<Vec3> indPlus, indMinus;
                contextPME.setPositions(posPlus);
                forcePME->getInducedDipoles(contextPME, indPlus);
                contextPME.setPositions(posMinus);
                forcePME->getInducedDipoles(contextPME, indMinus);
                printf("       Induced[0] at +delta: [%.8f, %.8f, %.8f]\n",
                       indPlus[0][0], indPlus[0][1], indPlus[0][2]);
                printf("       Induced[0] at -delta: [%.8f, %.8f, %.8f]\n",
                       indMinus[0][0], indMinus[0][1], indMinus[0][2]);
                printf("       Induced[0] change: [%.8e, %.8e, %.8e]\n",
                       indPlus[0][0]-indMinus[0][0], indPlus[0][1]-indMinus[0][1], indPlus[0][2]-indMinus[0][2]);
                printf("       Induced[1] at +delta: [%.8f, %.8f, %.8f]\n",
                       indPlus[1][0], indPlus[1][1], indPlus[1][2]);
                printf("       Induced[1] at -delta: [%.8f, %.8f, %.8f]\n",
                       indMinus[1][0], indMinus[1][1], indMinus[1][2]);
                printf("       Induced[1] change: [%.8e, %.8e, %.8e]\n",
                       indPlus[1][0]-indMinus[1][0], indPlus[1][1]-indMinus[1][1], indPlus[1][2]-indMinus[1][2]);
            }

            if (fabs(diff) > maxDiff) maxDiff = fabs(diff);
        }
    }
    cout << "Max difference: " << maxDiff << " kJ/mol/nm" << endl;

    // AMOEBA comparison - load from custom build
    Platform::loadPluginsFromDirectory("/home/aehogan2/PycharmProjects/tholedipoleplugin/amoeba_reference/build/plugins");

    System systemAmoeba;
    systemAmoeba.addParticle(1.0);
    systemAmoeba.addParticle(1.0);
    systemAmoeba.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0.0, 0.0),
                                              Vec3(0.0, boxSize, 0.0),
                                              Vec3(0.0, 0.0, boxSize));

    AmoebaMultipoleForce* forceAmoeba = new AmoebaMultipoleForce();
    systemAmoeba.addForce(forceAmoeba);
    forceAmoeba->setNonbondedMethod(AmoebaMultipoleForce::PME);
    forceAmoeba->setCutoffDistance(0.9);
    forceAmoeba->setPolarizationType(AmoebaMultipoleForce::Direct);

    vector<double> zeroQuadrupole(9, 0.0);
    double thole = 0.39;
    double dampingFactor1 = pow(polarizability, 1.0/6.0);
    double dampingFactor2 = pow(polarizability, 1.0/6.0);

    forceAmoeba->addMultipole(charge1, dipole1, zeroQuadrupole,
                              AmoebaMultipoleForce::ZOnly, 1, -1, -1,
                              thole, dampingFactor1, polarizability);
    forceAmoeba->addMultipole(charge2, dipole2, zeroQuadrupole,
                              AmoebaMultipoleForce::ZOnly, 0, -1, -1,
                              thole, dampingFactor2, polarizability);

    LangevinIntegrator integratorAmoeba(0.0, 0.1, 0.01);
    Context contextAmoeba(systemAmoeba, integratorAmoeba, Platform::getPlatformByName("Reference"));
    contextAmoeba.setPositions(positions);

    State stateAmoeba = contextAmoeba.getState(State::Forces | State::Energy);
    double energyAmoeba = stateAmoeba.getPotentialEnergy();
    const vector<Vec3>& forcesAmoeba = stateAmoeba.getForces();

    vector<Vec3> inducedAmoeba;
    forceAmoeba->getInducedDipoles(contextAmoeba, inducedAmoeba);

    cout << endl << "=== AMOEBA PME ===" << endl;
    cout << "Energy: " << energyAmoeba << " kJ/mol" << endl;
    cout << "Force[0]: " << forcesAmoeba[0] << " kJ/mol/nm" << endl;
    cout << "Force[1]: " << forcesAmoeba[1] << " kJ/mol/nm" << endl;
    cout << "Induced[0]: " << inducedAmoeba[0] << " e·nm" << endl;
    cout << "Induced[1]: " << inducedAmoeba[1] << " e·nm" << endl;

    cout << endl << "=== TholeDipole vs AMOEBA ===" << endl;
    cout << "Energy diff: " << (energyPME - energyAmoeba) << " kJ/mol" << endl;
    double forceDiffAmoeba0 = sqrt((forcesPME[0] - forcesAmoeba[0]).dot(forcesPME[0] - forcesAmoeba[0]));
    double forceDiffAmoeba1 = sqrt((forcesPME[1] - forcesAmoeba[1]).dot(forcesPME[1] - forcesAmoeba[1]));
    cout << "Force[0] diff: " << forceDiffAmoeba0 << " kJ/mol/nm" << endl;
    cout << "Force[1] diff: " << forceDiffAmoeba1 << " kJ/mol/nm" << endl;

    // Finite difference check for AMOEBA
    cout << endl << "=== Finite Difference Check (AMOEBA) ===" << endl;
    cout << "Particle  Dim  Analytical    Numerical     Difference" << endl;
    cout << "----------------------------------------------------" << endl;
    for (int p = 0; p < 2; p++) {
        for (int d = 0; d < 3; d++) {
            vector<Vec3> posPlus = positions;
            vector<Vec3> posMinus = positions;
            posPlus[p][d] += delta;
            posMinus[p][d] -= delta;

            contextAmoeba.setPositions(posPlus);
            double ePlus = contextAmoeba.getState(State::Energy).getPotentialEnergy();
            contextAmoeba.setPositions(posMinus);
            double eMinus = contextAmoeba.getState(State::Energy).getPotentialEnergy();

            double numericalForce = -(ePlus - eMinus) / (2 * delta);
            double analyticalForce = forcesAmoeba[p][d];
            double diff = analyticalForce - numericalForce;

            char dimChar = "xyz"[d];
            printf("   %d      %c    %10.4f    %10.4f    %10.4f\n",
                   p, dimChar, analyticalForce, numericalForce, diff);

            if (p == 0 && d == 0) {
                printf("       AMOEBA E(center)=%14.8f  E(+)=%14.8f  E(-)=%14.8f\n",
                       energyAmoeba, ePlus, eMinus);
                printf("       AMOEBA Delta_E = %.10f kJ/mol\n", ePlus - eMinus);
            }
        }
    }

    // Force component comparison
    cout << endl << "=== Component-by-Component Force Comparison ===" << endl;
    cout << "TholeDipole Force[0][x]: " << forcesPME[0][0] << " kJ/mol/nm" << endl;
    cout << "AMOEBA Force[0][x]:      " << forcesAmoeba[0][0] << " kJ/mol/nm" << endl;
    cout << "Difference:              " << (forcesPME[0][0] - forcesAmoeba[0][0]) << " kJ/mol/nm" << endl;

    // Assertions - Newton's 3rd law check
    // Small violations (~0.02 kJ/mol/nm) can occur due to PME grid interpolation asymmetry
    ASSERT_EQUAL_TOL(sumPME[0], 0.0, 0.05);
    ASSERT_EQUAL_TOL(sumPME[1], 0.0, 0.05);
    ASSERT_EQUAL_TOL(sumPME[2], 0.0, 0.05);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n========================================" << endl;
        cout << "Two Charges PME Test" << endl;
        cout << "========================================\n" << endl;

        testTwoChargesPME();

        cout << "\n========================================" << endl;
        cout << "Test passed!" << endl;
        cout << "========================================" << endl;
    }
    catch (const std::exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR. Test failed." << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
