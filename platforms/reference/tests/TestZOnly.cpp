#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include "openmm/AmoebaMultipoleForce.h"

void testZOnly() {
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
    double thole = 0.39;
    double pol = 0.001;
    
    force->addParticle(charge1, d1, pol, thole, 
                      TholeDipoleForce::ZOnly, 1, -1, -1);
     
    force->addParticle(charge2, d2, pol, thole,
                      TholeDipoleForce::ZOnly, 0, -1, -1);
    
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);  // Along z-axis
    
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);
    
    State state = context.getState(State::Forces | State::Energy);
    
    // Print forces and energy
    printf("ZOnly Test:\n");
    printf("Energy: %.8f\n", state.getPotentialEnergy());
    for (int i = 0; i < 2; i++) {
        Vec3 f = state.getForces()[i];
        printf("Force[%d]: (%.6e, %.6e, %.6e)\n", i, f[0], f[1], f[2]);
    }
    
    // For a dipole along Z interacting with a charge along Z,
    // forces should be purely along Z (no x,y components)
    ASSERT(fabs(state.getForces()[0][0]) < 1e-10);
    ASSERT(fabs(state.getForces()[0][1]) < 1e-10);
    ASSERT(fabs(state.getForces()[1][0]) < 1e-10);
    ASSERT(fabs(state.getForces()[1][1]) < 1e-10);
    
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
        // Let OpenMM choose the best platform for AMOEBA (don't force Reference platform)
        Context amoebaContext(amoebaSystem, amoebaInteg);
        cout << "AMOEBA context created successfully on platform: " << amoebaContext.getPlatform().getName() << endl;
        
        // Use compareForces for full AMOEBA comparison including dipoles
        compareForces("ZOnly", system, amoebaSystem, positions, 1.0, 0.01);
    } catch (const std::exception& e) {
        cout << "Full AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testZOnly();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
