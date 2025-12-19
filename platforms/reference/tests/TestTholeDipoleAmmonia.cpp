/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

/**
 * Tests TholeDipoleForce with ammonia molecules for different polarization types.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

static void testAmmoniaWithPolarization(TholeDipoleForce::PolarizationType polType) {
    std::string polName = (polType == TholeDipoleForce::Direct) ? "Direct" : "Mutual";
    std::string testName = "testTholeDipoleAmmonia" + polName;
    cout << "\n=== Testing Ammonia " << polName << " Polarization ===" << endl;

    int numberOfParticles = 8;
    int inputPmeGridDimension = 0;
    double cutoff = 9000000.0;

    System tholeDipoleSystem;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(tholeDipoleSystem, tholeDipoleForce, TholeDipoleForce::NoCutoff, polType,
                            cutoff, inputPmeGridDimension);

    LangevinIntegrator tempIntegrator(0.0, 0.1, 0.01);
    Context tempContext(tholeDipoleSystem, tempIntegrator, *platform);
    std::vector<Vec3> forces;
    double energy;
    getForcesEnergyTholeDipoleAmmonia(tempContext, forces, energy);
    std::vector<Vec3> positions = tempContext.getState(State::Positions).getPositions();

    cout << "TholeDipole energy: " << energy << " kJ/mol" << endl;

    ASSERT(std::isfinite(energy));
    for (int i = 0; i < numberOfParticles; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }

    System amoebaSystem;
    for (int i = 0; i < numberOfParticles; i++)
        amoebaSystem.addParticle(tholeDipoleSystem.getParticleMass(i));

    Vec3 a, b, c;
    tholeDipoleSystem.getDefaultPeriodicBoxVectors(a, b, c);
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
    amoebaForce->setPolarizationType(polType == TholeDipoleForce::Direct ?
                                      AmoebaMultipoleForce::Direct : AmoebaMultipoleForce::Mutual);
    amoebaSystem.addForce(amoebaForce);

    compareForces(testName, tholeDipoleSystem, amoebaSystem, positions, 1e-4, 1e-3);
}

static void testAmmoniaNoPolarization() {
    std::string testName = "testTholeDipoleAmmoniaNoPolarization";
    cout << "\n=== Testing Ammonia No Polarization (PME) ===" << endl;

    int numberOfParticles = 8;
    double boxDimension = 2.0;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);

    System tholeDipoleSystem;
    tholeDipoleSystem.setDefaultPeriodicBoxVectors(a, b, c);

    int inputPmeGridDimension = 32;
    double cutoff = 0.9;

    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeDipoleForce->setPolarizationType(TholeDipoleForce::Direct);
    tholeDipoleForce->setCutoffDistance(cutoff);
    tholeDipoleForce->setPMEParameters(5.0, inputPmeGridDimension, inputPmeGridDimension, inputPmeGridDimension);
    tholeDipoleForce->setEwaldErrorTolerance(1.0e-4);

    std::vector<double> nitrogenMolecularDipole = {0.0, 0.0, 3.4e-3};
    std::vector<double> hydrogenMolecularDipole = {0.0, 0.0, -4.7e-3};

    tholeDipoleSystem.addParticle(1.4007000e+01);
    tholeDipoleForce->addParticle(-5.7960000e-1, nitrogenMolecularDipole, 0.0, TholeDipoleForce::ThreeFold, 1, 2, 3);

    tholeDipoleSystem.addParticle(1.0080000e+00);
    tholeDipoleSystem.addParticle(1.0080000e+00);
    tholeDipoleSystem.addParticle(1.0080000e+00);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZOnly, 0, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZOnly, 0, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZOnly, 0, -1, -1);

    tholeDipoleSystem.addParticle(1.4007000e+01);
    tholeDipoleForce->addParticle(-5.796e-1, nitrogenMolecularDipole, 0.0, TholeDipoleForce::ThreeFold, 5, 6, 7);

    tholeDipoleSystem.addParticle(1.0080000e+00);
    tholeDipoleSystem.addParticle(1.0080000e+00);
    tholeDipoleSystem.addParticle(1.0080000e+00);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZOnly, 4, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZOnly, 4, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZOnly, 4, -1, -1);

    std::vector<int> covalentMap;
    covalentMap = {1, 2, 3};
    tholeDipoleForce->setCovalentMap(0, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {0};
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {2, 3};
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent13, covalentMap);
    covalentMap = {0};
    tholeDipoleForce->setCovalentMap(2, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {1, 3};
    tholeDipoleForce->setCovalentMap(2, TholeDipoleForce::Covalent13, covalentMap);
    covalentMap = {0};
    tholeDipoleForce->setCovalentMap(3, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {1, 2};
    tholeDipoleForce->setCovalentMap(3, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap = {5, 6, 7};
    tholeDipoleForce->setCovalentMap(4, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {4};
    tholeDipoleForce->setCovalentMap(5, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {6, 7};
    tholeDipoleForce->setCovalentMap(5, TholeDipoleForce::Covalent13, covalentMap);
    covalentMap = {4};
    tholeDipoleForce->setCovalentMap(6, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {5, 7};
    tholeDipoleForce->setCovalentMap(6, TholeDipoleForce::Covalent13, covalentMap);
    covalentMap = {4};
    tholeDipoleForce->setCovalentMap(7, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {5, 6};
    tholeDipoleForce->setCovalentMap(7, TholeDipoleForce::Covalent13, covalentMap);

    tholeDipoleSystem.addForce(tholeDipoleForce);

    std::vector<Vec3> positions(numberOfParticles);
    positions[0] = Vec3(  1.5927280e-01,  1.7000000e-06,   1.6491000e-03);
    positions[1] = Vec3(  2.0805540e-01, -8.1258800e-02,   3.7282500e-02);
    positions[2] = Vec3(  2.0843610e-01,  8.0953200e-02,   3.7462200e-02);
    positions[3] = Vec3(  1.7280780e-01,  2.0730000e-04,  -9.8741700e-02);
    positions[4] = Vec3( -1.6743680e-01,  1.5900000e-05,  -6.6149000e-03);
    positions[5] = Vec3( -2.0428260e-01,  8.1071500e-02,   4.1343900e-02);
    positions[6] = Vec3( -6.7308300e-02,  1.2800000e-05,   1.0623300e-02);
    positions[7] = Vec3( -2.0426290e-01, -8.1231400e-02,   4.1033500e-02);

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
        testAmmoniaWithPolarization(TholeDipoleForce::Direct);
        testAmmoniaWithPolarization(TholeDipoleForce::Mutual);
        testAmmoniaNoPolarization();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
