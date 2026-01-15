/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "OpenCLTests.h"
#include "OpenCLTestCommon.h"

void testPMECutoffConvergence() {
    const int numAtoms = 256;

    double charge1 = 0.00244629406;
    double charge2 = -0.00244629406;
    double pol1 = 0.0001;
    double pol2 = 0.00015;
    double separation = 0.06;
    double dampingParam = 21.304;

    int numPairs = numAtoms / 2;
    double boxSize = numPairs * 0.1;

    vector<double> cutoffs = {1.0, 2.0, 3.0, 5.0};

    cout << "\n=== PME Cutoff Convergence (" << numAtoms << " atoms) ===" << endl;

    for (double cutoff : cutoffs) {
        System system;
        for (int i = 0; i < numAtoms; i++)
            system.addParticle(1.0);

        system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0.0, 0.0),
                                           Vec3(0.0, boxSize, 0.0),
                                           Vec3(0.0, 0.0, boxSize));

        TholeDipoleForce* force = new TholeDipoleForce();
        system.addForce(force);

        force->setNonbondedMethod(TholeDipoleForce::PME);
        force->setCutoffDistance(cutoff);
        force->setTholeDampingType(TholeDipoleForce::Exponential);
        force->setTholeDampingParameter(dampingParam);
        force->setPolarizationType(TholeDipoleForce::Mutual);
        force->setMutualInducedMaxIterations(500);
        force->setMutualInducedTargetEpsilon(1.0e-9);

        vector<double> zeroDipole(3, 0.0);
        vector<Vec3> positions(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double charge = (i % 2 == 0) ? charge1 : charge2;
            double pol = (i % 2 == 0) ? pol1 : pol2;
            force->addParticle(charge, zeroDipole, pol,
                               TholeDipoleForce::NoAxisType, -1, -1, -1);
            positions[i] = Vec3(i / 2 * 0.1 + i % 2 * separation, 0.0, 0.0);
        }

        cout << "  Testing cutoff = " << cutoff << " nm..." << endl;
        assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testPMECutoffConvergence();
    }
    catch (const std::exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR. Test failed." << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
