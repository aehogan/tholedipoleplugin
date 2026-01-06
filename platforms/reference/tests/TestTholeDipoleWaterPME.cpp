/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

/**
 * Tests TholeDipoleForce with water molecules using PME for different polarization types.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

static void testWaterPMEWithPolarization(TholeDipoleForce::PolarizationType polType) {
    std::string polName = (polType == TholeDipoleForce::Direct) ? "Direct" : "Mutual";
    std::string testName = "testTholeDipoleWaterPME" + polName;
    cout << "\n=== Testing Water PME " << polName << " Polarization ===" << endl;

    int numberOfParticles = 12;
    int inputPmeGridDimension = 20;
    double cutoff = 0.70;

    std::vector<Vec3> forces;
    double energy;
    setupAndGetForcesEnergyTholeDipoleWater(TholeDipoleForce::PME, polType,
                                            cutoff, inputPmeGridDimension, forces, energy);

    cout << "TholeDipole energy: " << energy << " kJ/mol" << endl;

    ASSERT(std::isfinite(energy));
    for (int i = 0; i < numberOfParticles; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }

    System tholeDipoleSystem;
    double boxDimension = 1.8643;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);
    tholeDipoleSystem.setDefaultPeriodicBoxVectors(a, b, c);

    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeDipoleForce->setPolarizationType(polType);
    tholeDipoleForce->setCutoffDistance(cutoff);
    tholeDipoleForce->setMutualInducedTargetEpsilon(1.0e-6);
    tholeDipoleForce->setMutualInducedMaxIterations(500);
    tholeDipoleForce->setPMEParameters(5.4459052e+00, inputPmeGridDimension, inputPmeGridDimension, inputPmeGridDimension);
    tholeDipoleForce->setEwaldErrorTolerance(1.0e-4);

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        tholeDipoleSystem.addParticle(1.5995000e+01);
        tholeDipoleSystem.addParticle(1.0080000e+00);
        tholeDipoleSystem.addParticle(1.0080000e+00);
    }

    std::vector<double> oxygenMolecularDipole = {0.0, 0.0, 7.5561214e-3};
    std::vector<double> hydrogenMolecularDipole = {-2.0420949e-3, 0.0, -3.0787530e-3};

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        tholeDipoleForce->addParticle(-5.1966000e-1, oxygenMolecularDipole, 8.3700000e-4, 1, jj+1, jj+2, -1);
        tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, jj, jj+2, -1);
        tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, jj, jj+1, -1);
    }

    std::vector<int> covalentMap;
    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        covalentMap.clear();
        covalentMap.push_back(jj+1);
        covalentMap.push_back(jj+2);
        tholeDipoleForce->setCovalentMap(jj, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj);
        tholeDipoleForce->setCovalentMap(jj+1, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);
        tholeDipoleForce->setCovalentMap(jj+2, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+2);
        tholeDipoleForce->setCovalentMap(jj+1, static_cast<TholeDipoleForce::CovalentType>(1), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+1);
        tholeDipoleForce->setCovalentMap(jj+2, static_cast<TholeDipoleForce::CovalentType>(1), covalentMap);
    }

    tholeDipoleSystem.addForce(tholeDipoleForce);

    std::vector<Vec3> positions(numberOfParticles);
    positions[0]  = Vec3(-8.7387270e-01,  5.3220410e-01,  7.4214000e-03);
    positions[1]  = Vec3(-9.6050090e-01,  5.1173410e-01, -2.2202700e-02);
    positions[2]  = Vec3(-8.5985900e-01,  4.9658230e-01,  1.0283390e-01);
    positions[3]  = Vec3( 9.1767100e-02, -7.8956650e-01,  4.3804200e-01);
    positions[4]  = Vec3( 1.2333420e-01, -7.0267430e-01,  4.2611550e-01);
    positions[5]  = Vec3( 1.7267090e-01, -8.2320810e-01,  4.8124750e-01);
    positions[6]  = Vec3( 8.6290110e-01,  6.2153500e-02,  4.1280850e-01);
    positions[7]  = Vec3( 8.6385200e-01,  1.2684730e-01,  3.3887060e-01);
    positions[8]  = Vec3( 9.5063550e-01,  5.3173300e-02,  4.4799160e-01);
    positions[9]  = Vec3( 5.0844930e-01,  2.8684740e-01, -6.9293750e-01);
    positions[10] = Vec3( 6.0459330e-01,  3.0620510e-01, -7.0100130e-01);
    positions[11] = Vec3( 5.0590640e-01,  1.8880920e-01, -6.8813470e-01);

    System amoebaSystem;
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);
    for (int i = 0; i < numberOfParticles; i++)
        amoebaSystem.addParticle(tholeDipoleSystem.getParticleMass(i));

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::PME);
    amoebaForce->setPolarizationType(polType == TholeDipoleForce::Direct ?
                                      AmoebaMultipoleForce::Direct : AmoebaMultipoleForce::Mutual);
    amoebaSystem.addForce(amoebaForce);

    // For Mutual polarization, use looser tolerances because TholeDipole (single induced dipole)
    // and AMOEBA (d/p induced dipole split) have fundamentally different convergence dynamics.
    // Direct polarization should match with machine precision.
    double energyTol = (polType == TholeDipoleForce::Mutual) ? 1e-3 : 1e-4;
    double forceTol = (polType == TholeDipoleForce::Mutual) ? 1e-2 : 1e-3;
    compareForces(testName, tholeDipoleSystem, amoebaSystem, positions, energyTol, forceTol);
}

static void testWaterPMENoPolarization() {
    std::string testName = "testTholeDipoleWaterPMENoPolarization";
    cout << "\n=== Testing Water PME No Polarization ===" << endl;

    int numberOfParticles = 12;
    int inputPmeGridDimension = 20;
    double cutoff = 0.70;

    System tholeDipoleSystem;
    double boxDimension = 1.8643;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);
    tholeDipoleSystem.setDefaultPeriodicBoxVectors(a, b, c);

    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeDipoleForce->setPolarizationType(TholeDipoleForce::Direct);
    tholeDipoleForce->setCutoffDistance(cutoff);
    tholeDipoleForce->setMutualInducedTargetEpsilon(1.0e-6);
    tholeDipoleForce->setMutualInducedMaxIterations(500);
    tholeDipoleForce->setPMEParameters(5.4459052e+00, inputPmeGridDimension, inputPmeGridDimension, inputPmeGridDimension);
    tholeDipoleForce->setEwaldErrorTolerance(1.0e-4);

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        tholeDipoleSystem.addParticle(1.5995000e+01);
        tholeDipoleSystem.addParticle(1.0080000e+00);
        tholeDipoleSystem.addParticle(1.0080000e+00);
    }

    std::vector<double> oxygenMolecularDipole = {0.0, 0.0, 7.5561214e-3};
    std::vector<double> hydrogenMolecularDipole = {-2.0420949e-3, 0.0, -3.0787530e-3};

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        tholeDipoleForce->addParticle(-5.1966000e-1, oxygenMolecularDipole, 0.0, TholeDipoleForce::Bisector, jj+1, jj+2, -1);
        tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZThenX, jj, jj+2, -1);
        tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZThenX, jj, jj+1, -1);
    }

    std::vector<int> covalentMap;
    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        covalentMap.clear();
        covalentMap.push_back(jj+1);
        covalentMap.push_back(jj+2);
        tholeDipoleForce->setCovalentMap(jj, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj);
        tholeDipoleForce->setCovalentMap(jj+1, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);
        tholeDipoleForce->setCovalentMap(jj+2, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+2);
        tholeDipoleForce->setCovalentMap(jj+1, static_cast<TholeDipoleForce::CovalentType>(1), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+1);
        tholeDipoleForce->setCovalentMap(jj+2, static_cast<TholeDipoleForce::CovalentType>(1), covalentMap);
    }

    tholeDipoleSystem.addForce(tholeDipoleForce);

    std::vector<Vec3> positions(numberOfParticles);
    positions[0]  = Vec3(-8.7387270e-01,  5.3220410e-01,  7.4214000e-03);
    positions[1]  = Vec3(-9.6050090e-01,  5.1173410e-01, -2.2202700e-02);
    positions[2]  = Vec3(-8.5985900e-01,  4.9658230e-01,  1.0283390e-01);
    positions[3]  = Vec3( 9.1767100e-02, -7.8956650e-01,  4.3804200e-01);
    positions[4]  = Vec3( 1.2333420e-01, -7.0267430e-01,  4.2611550e-01);
    positions[5]  = Vec3( 1.7267090e-01, -8.2320810e-01,  4.8124750e-01);
    positions[6]  = Vec3( 8.6290110e-01,  6.2153500e-02,  4.1280850e-01);
    positions[7]  = Vec3( 8.6385200e-01,  1.2684730e-01,  3.3887060e-01);
    positions[8]  = Vec3( 9.5063550e-01,  5.3173300e-02,  4.4799160e-01);
    positions[9]  = Vec3( 5.0844930e-01,  2.8684740e-01, -6.9293750e-01);
    positions[10] = Vec3( 6.0459330e-01,  3.0620510e-01, -7.0100130e-01);
    positions[11] = Vec3( 5.0590640e-01,  1.8880920e-01, -6.8813470e-01);

    LangevinIntegrator tholeInteg(0.0, 0.1, 0.01);
    Context tholeContext(tholeDipoleSystem, tholeInteg, *platform);
    tholeContext.setPositions(positions);
    State tholeState = tholeContext.getState(State::Forces | State::Energy);
    double tholeEnergy = tholeState.getPotentialEnergy();

    cout << "TholeDipole energy: " << tholeEnergy << " kJ/mol" << endl;

    ASSERT(std::isfinite(tholeEnergy));
    for (int i = 0; i < numberOfParticles; i++) {
        ASSERT(std::isfinite(tholeState.getForces()[i][0]));
        ASSERT(std::isfinite(tholeState.getForces()[i][1]));
        ASSERT(std::isfinite(tholeState.getForces()[i][2]));
    }

    System amoebaSystem;
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);
    for (int i = 0; i < numberOfParticles; i++)
        amoebaSystem.addParticle(tholeDipoleSystem.getParticleMass(i));

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::PME);
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
    amoebaSystem.addForce(amoebaForce);

    compareForces(testName, tholeDipoleSystem, amoebaSystem, positions, 1e-4, 1e-3);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testWaterPMEWithPolarization(TholeDipoleForce::Direct);
        testWaterPMEWithPolarization(TholeDipoleForce::Mutual);
        testWaterPMENoPolarization();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
