/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "CudaTests.h"
#include "CudaTestCommon.h"

static void testWaterPMEDirect() {
    cout << "  WaterPME Direct..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    setupWaterPME(system, force, TholeDipoleForce::Direct, 0.70, 20);
    vector<Vec3> positions = getWaterPMEPositions();
    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

static void testWaterPMEMutual() {
    cout << "  WaterPME Mutual..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    setupWaterPME(system, force, TholeDipoleForce::Mutual, 0.70, 20);
    vector<Vec3> positions = getWaterPMEPositions();
    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

static void testWaterPMENoPol() {
    cout << "  WaterPME NoPol..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();

    int numberOfParticles = 12;
    double boxDimension = 1.8643;
    system.setDefaultPeriodicBoxVectors(Vec3(boxDimension, 0, 0),
                                        Vec3(0, boxDimension, 0),
                                        Vec3(0, 0, boxDimension));

    force->setNonbondedMethod(TholeDipoleForce::PME);
    force->setPolarizationType(TholeDipoleForce::Direct);
    force->setCutoffDistance(0.70);
    force->setPMEParameters(5.4459052e+00, 20, 20, 20);

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        system.addParticle(1.5995000e+01);
        system.addParticle(1.0080000e+00);
        system.addParticle(1.0080000e+00);
    }

    vector<double> oxygenMolecularDipole = {0.0, 0.0, 7.5561214e-3};
    vector<double> hydrogenMolecularDipole = {-2.0420949e-3, 0.0, -3.0787530e-3};

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        force->addParticle(-5.1966000e-1, oxygenMolecularDipole, 0.0, TholeDipoleForce::Bisector, jj+1, jj+2, -1);
        force->addParticle(2.5983000e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZThenX, jj, jj+2, -1);
        force->addParticle(2.5983000e-1, hydrogenMolecularDipole, 0.0, TholeDipoleForce::ZThenX, jj, jj+1, -1);
    }

    vector<int> covalentMap;
    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        covalentMap.clear();
        covalentMap.push_back(jj+1);
        covalentMap.push_back(jj+2);
        force->setCovalentMap(jj, TholeDipoleForce::Covalent12, covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj);
        force->setCovalentMap(jj+1, TholeDipoleForce::Covalent12, covalentMap);
        force->setCovalentMap(jj+2, TholeDipoleForce::Covalent12, covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+2);
        force->setCovalentMap(jj+1, TholeDipoleForce::Covalent13, covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+1);
        force->setCovalentMap(jj+2, TholeDipoleForce::Covalent13, covalentMap);
    }

    system.addForce(force);

    vector<Vec3> positions = getWaterPMEPositions();
    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n=== Water PME Tests ===" << endl;
        testWaterPMEDirect();
        testWaterPMEMutual();
        testWaterPMENoPol();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
