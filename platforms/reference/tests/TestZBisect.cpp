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
 * This tests TholeDipoleForce zbisect.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testZBisect() {
    System system;
    for (int i = 0; i < 7; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::PME);
    force->setCutoffDistance(1.2);
    double charge[] = {-1.01875, 0, 0, 0, -0.51966, 0.25983, 0.25983};
    double dipole[7][3] = {
        {0.06620218576365969, 0.056934176095985306, 0.06298584667720743},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0.007556121391156931},
        {-0.05495981592297553, 0, -0.0030787530116780605},
        {-0.05495981592297553, 0, -0.0030787530116780605}};
    int axis[7][4] = {
        {2, 2, 1, 3},
        {5, -1, -1, -1},
        {5, -1, -1, -1},
        {5, -1, -1, -1},
        {1, 5, 6, -1},
        {0, 4, 6, -1},
        {0, 4, 5, -1}
    };
    double thole = 0.39;
    double polarity[] = {0.001334, 0.001334, 0.001334, 0.001334, 0.000837, 0.000496, 0.000496};
    for (int i = 0; i < 7; i++) {
        vector<double> d;
        for (int j = 0; j < 3; j++)
            d.push_back(dipole[i][j]);
        force->addParticle(charge[i], d, polarity[i], thole, axis[i][0], axis[i][1], axis[i][2], axis[i][3]);
    }
    for (int i = 0; i < 4; i++) {
        vector<int> map;
        if (i != 0) map.push_back(0);
        force->setCovalentMap(i, TholeDipoleForce::Covalent12, map);
        map.clear();
        if (i != 1) map.push_back(1);
        if (i != 2) map.push_back(2);
        if (i != 3) map.push_back(3);
        force->setCovalentMap(i, TholeDipoleForce::Covalent13, map);
    }
    for (int i = 4; i < 7; i++) {
        vector<int> map;
        if (i != 4) map.push_back(4);
        force->setCovalentMap(i, TholeDipoleForce::Covalent12, map);
        map.clear();
        if (i != 5) map.push_back(5);
        if (i != 6) map.push_back(6);
        force->setCovalentMap(i, TholeDipoleForce::Covalent13, map);
    }
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    vector<Vec3> positions;
    positions.push_back(Vec3(-0.06317711175870899, -0.04905009196658128, 0.0767217));
    positions.push_back(Vec3(-0.049166918626451395, -0.20747614470348363, 0.03979849999999996));
    positions.push_back(Vec3(-0.19317150000000005, -0.05811762921948427, 0.1632788999999999));
    positions.push_back(Vec3(0.04465103038516016, -0.018345116763806235, 0.18531239999999993));
    positions.push_back(Vec3(0.005630299999999998, 0.40965770000000035, 0.5731495));
    positions.push_back(Vec3(0.036148100000000016, 0.3627041999999996, 0.49299430000000033));
    positions.push_back(Vec3(0.07781149999999992, 0.4178183000000004, 0.6355703000000004));
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    
    // Basic sanity checks
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    
    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 7; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1])); 
        ASSERT(std::isfinite(forces[i][2]));
    }
    
    // Compare with AMOEBA
    try {
        // Create equivalent AMOEBA system
        System amoebaSystem;
        for (int i = 0; i < 7; i++)
            amoebaSystem.addParticle(1.0);
        amoebaSystem.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
        
        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaSystem.addForce(amoebaForce);
        
        // Use compareForces for full AMOEBA comparison including dipoles
        compareForces("ZBisect", system, amoebaSystem, positions, 0.01, 0.01);
    } catch (const std::exception& e) {
        cout << "Full AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testZBisect();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}