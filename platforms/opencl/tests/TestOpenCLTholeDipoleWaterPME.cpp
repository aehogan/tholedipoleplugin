/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "OpenCLTests.h"
#include "OpenCLTestCommon.h"

static void testWaterPMEDirect() {
    cout << "  WaterPME Direct..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    setupWaterPME(system, force, TholeDipoleForce::Direct, 0.7, 24);
    vector<Vec3> positions = getWaterPMEPositions();
    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

static void testWaterPMEMutual() {
    cout << "  WaterPME Mutual..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    setupWaterPME(system, force, TholeDipoleForce::Mutual, 0.7, 24);
    vector<Vec3> positions = getWaterPMEPositions();
    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

static void testWaterPMENoPol() {
    cout << "  WaterPME NoPol..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    setupWaterPME(system, force, TholeDipoleForce::Direct, 0.7, 24);

    for (int i = 0; i < force->getNumParticles(); i++) {
        double charge, polarity;
        int axisType, atomZ, atomX, atomY;
        vector<double> dipole;
        force->getParticleParameters(i, charge, dipole, polarity, axisType, atomZ, atomX, atomY);
        force->setParticleParameters(i, charge, dipole, 0.0, axisType, atomZ, atomX, atomY);
    }

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
