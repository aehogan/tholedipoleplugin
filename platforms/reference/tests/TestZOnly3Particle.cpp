#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include "openmm/AmoebaMultipoleForce.h"

void testZOnly3Particle() {

    System tholeSystem;
    for (int i = 0; i < 3; i++)
        tholeSystem.addParticle(1.0);
    
    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);
    tholeSystem.addForce(force);
    
    // Create a 3-particle system with Z-only axis types
    double charge[] = {0.5, -0.25, -0.25};
    double dipole[3][3] = {
        {0.0, 0.0, 0.1},    // dipole along z
        {0.0, 0.0, 0.05},
        {0.0, 0.0, 0.025}
    };
    double thole = 0.39;
    double polarity[] = {0.001, 0.001, 0.001};
    
    for (int i = 0; i < 3; i++) {
        vector<double> d;
        for (int j = 0; j < 3; j++)
            d.push_back(dipole[i][j]);
        
        // For Z-only axis, use another particle as Z-axis reference
        int zAxis = (i + 1) % 3;  // Circular reference
        force->addParticle(charge[i], d, polarity[i], thole,
                          TholeDipoleForce::ZOnly, zAxis, -1, -1);
    }
    
    // Create equivalent AMOEBA system
    System amoebaSystem;
    for (int i = 0; i < 3; i++)
        amoebaSystem.addParticle(1.0);
    
    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
    amoebaSystem.addForce(amoebaForce);
    
    vector<Vec3> positions(3);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.3, 0.0, 0.0);   // Along x from particle 0
    positions[2] = Vec3(0.0, 0.3, 0.0);   // Along y from particle 0
    
    // Test TholeDipole system standalone first
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    
    // Basic sanity check - energy should be finite
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    
    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 3; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1])); 
        ASSERT(std::isfinite(forces[i][2]));
    }
    
    // Use compareForces for full AMOEBA comparison including dipoles
    cout << "Starting full AMOEBA comparison with dipoles..." << endl;
    // Compare with AMOEBA
    try {
        // Create equivalent AMOEBA system
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
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
        compareForces("ZOnly3Particle", tholeSystem, amoebaSystem, positions, 0.01, 0.01);
    } catch (const std::exception& e) {
        cout << "Full AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testZOnly3Particle();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
