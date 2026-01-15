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

#include "OpenCLTests.h"
#include "OpenCLTestCommon.h"

void test1DChainSizeConsistency() {
    double charge1 = 0.00244629406;
    double charge2 = -0.00244629406;
    double pol1 = 0.0001;
    double pol2 = 0.00015;
    double separation = 0.06;
    double dampingParam = 21.304;

    vector<int> systemSizes = {32, 64, 128};

    cout << "\n=== 1D Chain Size Consistency ===" << endl;

    for (int numAtoms : systemSizes) {
        int numPairs = numAtoms / 2;
        double boxSize = numPairs * 0.1;

        System system;
        for (int i = 0; i < numAtoms; i++)
            system.addParticle(1.0);

        system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0.0, 0.0),
                                           Vec3(0.0, 20.0, 0.0),
                                           Vec3(0.0, 0.0, 20.0));

        TholeDipoleForce* force = new TholeDipoleForce();
        system.addForce(force);

        force->setNonbondedMethod(TholeDipoleForce::PME);
        double cutoff = min(0.9, boxSize * 0.49);
        force->setCutoffDistance(cutoff);
        force->setTholeDampingType(TholeDipoleForce::Exponential);
        force->setTholeDampingParameter(dampingParam);
        force->setPolarizationType(TholeDipoleForce::Mutual);
        force->setMutualInducedTargetEpsilon(1.0e-9);
        force->setMutualInducedMaxIterations(500);

        vector<double> zeroDipole(3, 0.0);
        vector<Vec3> positions(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double charge = (i % 2 == 0) ? charge1 : charge2;
            double pol = (i % 2 == 0) ? pol1 : pol2;
            force->addParticle(charge, zeroDipole, pol,
                               TholeDipoleForce::NoAxisType, -1, -1, -1);
            positions[i] = Vec3(i / 2 * 0.1 + i % 2 * separation, 0.0, 0.0);
        }

        double energyTol = 5e-5;
        double forceTol = 5e-4;

        cout << "  Testing " << numAtoms << " atoms (box=" << boxSize << " nm)..." << endl;
        assertForcesAndEnergiesMatch(system, positions, energyTol, forceTol);
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
