#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testZThenX() {
    System tholeSystem;
    for (int i = 0; i < 3; i++)
        tholeSystem.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    tholeSystem.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    double charge[] = {0.5, -0.25, -0.25};
    double dipole[3][3] = {{0.0, 0.0, 0.1}, {0.1, 0.0, 0.0}, {0.0, 0.1, 0.0}};
    double polarizability[] = {0.001, 0.001, 0.001};

    vector<double> d0, d1, d2;
    for (int j = 0; j < 3; j++) { d0.push_back(dipole[0][j]); d1.push_back(dipole[1][j]); d2.push_back(dipole[2][j]); }

    force->addParticle(charge[0], d0, polarizability[0], TholeDipoleForce::ZThenX, 1, 2, -1);
    force->addParticle(charge[1], d1, polarizability[1], TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge[2], d2, polarizability[2], TholeDipoleForce::NoAxisType, -1, -1, -1);

    System amoebaSystem;
    for (int i = 0; i < 3; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
    amoebaSystem.addForce(amoebaForce);

    vector<Vec3> positions(3);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.0, 0.0, 0.3);
    positions[2] = Vec3(0.3, 0.0, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    ASSERT(state.getPotentialEnergy() < 0.0);

    Vec3 forceSum = state.getForces()[0] + state.getForces()[1] + state.getForces()[2];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);

    try {
        compareForces("ZThenX", tholeSystem, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testBisector() {
    System tholeSystem;
    for (int i = 0; i < 4; i++)
        tholeSystem.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    tholeSystem.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    double charge[] = {0.5, -0.25, -0.25, -0.1};
    double dipole[4][3] = {{0.0, 0.05, 0.08}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.1}};
    double polarizability[] = {0.001, 0.001, 0.001, 0.001};

    vector<double> d0, d1, d2, d3;
    for (int j = 0; j < 3; j++) { d0.push_back(dipole[0][j]); d1.push_back(dipole[1][j]); d2.push_back(dipole[2][j]); d3.push_back(dipole[3][j]); }

    force->addParticle(charge[0], d0, polarizability[0], TholeDipoleForce::Bisector, 3, 1, 2);
    force->addParticle(charge[1], d1, polarizability[1], TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge[2], d2, polarizability[2], TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge[3], d3, polarizability[3], TholeDipoleForce::NoAxisType, -1, -1, -1);

    System amoebaSystem;
    for (int i = 0; i < 4; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
    amoebaSystem.addForce(amoebaForce);

    vector<Vec3> positions(4);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(-0.15, 0.15, 0.0);
    positions[2] = Vec3(0.15, 0.15, 0.0);
    positions[3] = Vec3(0.0, 0.0, 0.3);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    Vec3 forceSum(0, 0, 0);
    for (int i = 0; i < 4; i++) forceSum += state.getForces()[i];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);

    try {
        compareForces("Bisector", tholeSystem, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testZBisect() {
    System system;
    for (int i = 0; i < 7; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);

    double charge[] = {-1.01875, 0, 0, 0, -0.51966, 0.25983, 0.25983};
    double dipole[7][3] = {
        {0.06620218576365969, 0.056934176095985306, 0.06298584667720743},
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
        {0, 0, 0.007556121391156931},
        {-0.05495981592297553, 0, -0.0030787530116780605},
        {-0.05495981592297553, 0, -0.0030787530116780605}
    };
    int axis[7][4] = {{2, 2, 1, 3}, {5, -1, -1, -1}, {5, -1, -1, -1}, {5, -1, -1, -1}, {1, 5, 6, -1}, {0, 4, 6, -1}, {0, 4, 5, -1}};
    double polarity[] = {0.001334, 0.001334, 0.001334, 0.001334, 0.000837, 0.000496, 0.000496};

    for (int i = 0; i < 7; i++) {
        vector<double> d;
        for (int j = 0; j < 3; j++) d.push_back(dipole[i][j]);
        force->addParticle(charge[i], d, polarity[i], axis[i][0], axis[i][1], axis[i][2], axis[i][3]);
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
    const vector<Vec3>& forces = state.getForces();

    Vec3 forceSum(0, 0, 0);
    for (int i = 0; i < 7; i++) forceSum += forces[i];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);

    try {
        System amoebaSystem;
        for (int i = 0; i < 7; i++) amoebaSystem.addParticle(1.0);
        amoebaSystem.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaSystem.addForce(amoebaForce);
        compareForces("ZBisect", system, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testThreeFold() {
    System tholeSystem;
    for (int i = 0; i < 4; i++)
        tholeSystem.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    tholeSystem.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    double charge[] = {-0.5, 0.2, 0.2, 0.1};
    double dipole[4][3] = {{0.0, 0.0, 0.1}, {0.05, 0.0, 0.0}, {0.0, 0.05, 0.0}, {-0.05, -0.05, 0.0}};
    double polarizability[] = {0.001, 0.001, 0.001, 0.001};

    vector<double> d0, d1, d2, d3;
    for (int j = 0; j < 3; j++) { d0.push_back(dipole[0][j]); d1.push_back(dipole[1][j]); d2.push_back(dipole[2][j]); d3.push_back(dipole[3][j]); }

    force->addParticle(charge[0], d0, polarizability[0], TholeDipoleForce::ThreeFold, 1, 2, 3);
    force->addParticle(charge[1], d1, polarizability[1], TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge[2], d2, polarizability[2], TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge[3], d3, polarizability[3], TholeDipoleForce::NoAxisType, -1, -1, -1);

    System amoebaSystem;
    for (int i = 0; i < 4; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
    amoebaSystem.addForce(amoebaForce);

    vector<Vec3> positions(4);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.2, 0.0, 0.0);
    positions[2] = Vec3(-0.1, 0.173, 0.0);
    positions[3] = Vec3(-0.1, -0.173, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    Vec3 forceSum(0, 0, 0);
    for (int i = 0; i < 4; i++) forceSum += state.getForces()[i];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);

    try {
        compareForces("ThreeFold", tholeSystem, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testNoAxisType() {
    System tholeSystem;
    for (int i = 0; i < 3; i++)
        tholeSystem.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    tholeSystem.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    double charge[] = {0.5, -0.25, -0.25};
    double polarizability[] = {0.001, 0.001, 0.001};
    vector<double> zeroDipole(3, 0.0);

    force->addParticle(charge[0], zeroDipole, polarizability[0], TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge[1], zeroDipole, polarizability[1], TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge[2], zeroDipole, polarizability[2], TholeDipoleForce::NoAxisType, -1, -1, -1);

    System amoebaSystem;
    for (int i = 0; i < 3; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
    amoebaSystem.addForce(amoebaForce);

    vector<Vec3> positions(3);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.3, 0.0, 0.0);
    positions[2] = Vec3(0.0, 0.3, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    ASSERT(state.getPotentialEnergy() < 0.0);

    Vec3 forceSum = state.getForces()[0] + state.getForces()[1] + state.getForces()[2];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);

    try {
        compareForces("NoAxisType", tholeSystem, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);

        cout << "=== ZThenX ===" << endl;
        testZThenX();

        cout << "\n=== Bisector ===" << endl;
        testBisector();

        cout << "\n=== ZBisect ===" << endl;
        testZBisect();

        cout << "\n=== ThreeFold ===" << endl;
        testThreeFold();

        cout << "\n=== NoAxisType ===" << endl;
        testNoAxisType();

        cout << "\nAll axis type tests passed!" << endl;
    }
    catch (const std::exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR.  Test failed." << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
