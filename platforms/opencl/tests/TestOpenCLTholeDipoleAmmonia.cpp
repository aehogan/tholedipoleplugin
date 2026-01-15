/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "OpenCLTests.h"
#include "OpenCLTestCommon.h"

static void testAmmoniaDirect() {
    cout << "  AmmoniaDirect..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    setupTholeDipoleAmmonia(system, force, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, 9000000.0, 0);
    vector<Vec3> positions = getAmmoniaPositions();
    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

static void testAmmoniaMutual() {
    cout << "  AmmoniaMutual..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    setupTholeDipoleAmmonia(system, force, TholeDipoleForce::NoCutoff, TholeDipoleForce::Mutual, 9000000.0, 0);
    vector<Vec3> positions = getAmmoniaPositions();
    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

static void testAmmoniaNoPol() {
    cout << "  AmmoniaNoPol..." << endl;
    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    setupTholeDipoleAmmonia(system, force, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, 9000000.0, 0);

    for (int i = 0; i < force->getNumParticles(); i++) {
        double charge, polarity;
        int axisType, atomZ, atomX, atomY;
        vector<double> dipole;
        force->getParticleParameters(i, charge, dipole, polarity, axisType, atomZ, atomX, atomY);
        force->setParticleParameters(i, charge, dipole, 0.0, axisType, atomZ, atomX, atomY);
    }

    vector<Vec3> positions = getAmmoniaPositions();
    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n=== Ammonia Tests ===" << endl;
        testAmmoniaDirect();
        testAmmoniaMutual();
        testAmmoniaNoPol();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
