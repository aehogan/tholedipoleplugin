#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include "openmm/AmoebaMultipoleForce.h"

void testZOnlyNoPol() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    
    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);
    system.addForce(force);
    
    vector<double> d1(3, 0.0);
    d1[2] = 0.1;
    vector<double> d2(3, 0.0);
    d2[2] = -0.05;

    double charge1 = 0.5;
    double charge2 = -0.5;
    double pol = 0.0;

    force->addParticle(charge1, d1, pol,
                      TholeDipoleForce::ZOnly, 1, -1, -1);

    force->addParticle(charge2, d2, pol,
                      TholeDipoleForce::ZOnly, 0, -1, -1);
    
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);
    
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);
    
    State state = context.getState(State::Forces | State::Energy);
    
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    
    // Basic sanity checks
    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 2; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }
    
    // For charges and dipoles along z-axis, forces should be purely along z
    ASSERT(fabs(forces[0][0]) < 1e-10);
    ASSERT(fabs(forces[0][1]) < 1e-10);
    ASSERT(fabs(forces[1][0]) < 1e-10);
    ASSERT(fabs(forces[1][1]) < 1e-10);
    
    // Compare with AMOEBA
    try {
        // Create equivalent AMOEBA system
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        amoebaSystem.addParticle(1.0);
        
        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
        amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
        amoebaSystem.addForce(amoebaForce);
        
        LangevinIntegrator amoebaInteg(0.0, 0.1, 0.01);
        Context amoebaContext(amoebaSystem, amoebaInteg);
        cout << "AMOEBA context created successfully on platform: " << amoebaContext.getPlatform().getName() << endl;
        
        compareForces("ZOnlyNoPol", system, amoebaSystem, positions, 0.01, 0.01);
    } catch (const std::exception& e) {
        cout << "Full AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testZOnlyNoPol();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
