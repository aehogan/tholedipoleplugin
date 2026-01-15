/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "OpenCLTests.h"
#include "OpenCLTestCommon.h"

void compareVec3Arrays(const vector<Vec3>& v1, const vector<Vec3>& v2, double tol, const string& testName) {
    ASSERT_EQUAL(v1.size(), v2.size());
    double maxDiff = 0.0;
    for (size_t i = 0; i < v1.size(); i++) {
        Vec3 diff = v1[i] - v2[i];
        maxDiff = max(maxDiff, sqrt(diff.dot(diff)));
    }
    ASSERT_EQUAL_TOL(0.0, maxDiff, tol);
}

void testInducedDipoleAccessor() {
    cout << "  InducedDipoleAccessor..." << endl;
    System openclSystem, refSystem;
    TholeDipoleForce* openclForce = new TholeDipoleForce();
    TholeDipoleForce* refForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(openclSystem, openclForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, 9000000.0, 0);
    setupTholeDipoleAmmonia(refSystem, refForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, 9000000.0, 0);

    vector<Vec3> positions = getAmmoniaPositions();

    VerletIntegrator integOpenCL(1.0);
    VerletIntegrator integRef(1.0);
    Context openclContext(openclSystem, integOpenCL, *openclPlatform);
    Context refContext(refSystem, integRef, *referencePlatform);
    openclContext.setPositions(positions);
    refContext.setPositions(positions);

    openclContext.getState(State::Energy);
    refContext.getState(State::Energy);

    vector<Vec3> openclDipoles, refDipoles;
    openclForce->getInducedDipoles(openclContext, openclDipoles);
    refForce->getInducedDipoles(refContext, refDipoles);

    compareVec3Arrays(openclDipoles, refDipoles, 1e-5, "InducedDipoles");
}

void testLabFramePermanentDipoleAccessor() {
    cout << "  LabFramePermanentDipoleAccessor..." << endl;
    System openclSystem, refSystem;
    TholeDipoleForce* openclForce = new TholeDipoleForce();
    TholeDipoleForce* refForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(openclSystem, openclForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, 9000000.0, 0);
    setupTholeDipoleAmmonia(refSystem, refForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, 9000000.0, 0);

    vector<Vec3> positions = getAmmoniaPositions();

    VerletIntegrator integOpenCL(1.0);
    VerletIntegrator integRef(1.0);
    Context openclContext(openclSystem, integOpenCL, *openclPlatform);
    Context refContext(refSystem, integRef, *referencePlatform);
    openclContext.setPositions(positions);
    refContext.setPositions(positions);

    openclContext.getState(State::Energy);
    refContext.getState(State::Energy);

    vector<Vec3> openclDipoles, refDipoles;
    openclForce->getLabFramePermanentDipoles(openclContext, openclDipoles);
    refForce->getLabFramePermanentDipoles(refContext, refDipoles);

    compareVec3Arrays(openclDipoles, refDipoles, 1e-5, "LabFramePermanentDipoles");
}

void testTotalDipoleAccessor() {
    cout << "  TotalDipoleAccessor..." << endl;
    System openclSystem, refSystem;
    TholeDipoleForce* openclForce = new TholeDipoleForce();
    TholeDipoleForce* refForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(openclSystem, openclForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, 9000000.0, 0);
    setupTholeDipoleAmmonia(refSystem, refForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, 9000000.0, 0);

    vector<Vec3> positions = getAmmoniaPositions();

    VerletIntegrator integOpenCL(1.0);
    VerletIntegrator integRef(1.0);
    Context openclContext(openclSystem, integOpenCL, *openclPlatform);
    Context refContext(refSystem, integRef, *referencePlatform);
    openclContext.setPositions(positions);
    refContext.setPositions(positions);

    openclContext.getState(State::Energy);
    refContext.getState(State::Energy);

    vector<Vec3> openclDipoles, refDipoles;
    openclForce->getTotalDipoles(openclContext, openclDipoles);
    refForce->getTotalDipoles(refContext, refDipoles);

    compareVec3Arrays(openclDipoles, refDipoles, 1e-5, "TotalDipoles");
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n=== Particle Accessor Tests ===" << endl;
        testInducedDipoleAccessor();
        testLabFramePermanentDipoleAccessor();
        testTotalDipoleAccessor();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
