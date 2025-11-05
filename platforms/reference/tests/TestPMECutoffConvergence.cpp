#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testPMECutoffConvergence() {
    const int numAtoms = 1024;

    double charge1 = 0.00244629406;    // e
    double charge2 = -0.00244629406;   // e
    double pol1 = 0.0001;              // 0.1 Å³ = 0.0001 nm³
    double pol2 = 0.00015;             // 0.15 Å³ = 0.00015 nm³
    double separation = 0.06;          // 0.6 Å = 0.06 nm
    double dampingParam = 21.304;      // 2.1304 Å⁻¹ = 21.304 nm⁻¹

    int numPairs = numAtoms / 2;
    double boxSize = numPairs * 0.1;

    vector<double> cutoffs = {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0};

    struct Result {
        double cutoff;
        double totalEnergy;
        double energyPerAtom;
    };
    vector<Result> results;

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

        vector<double> zeroDipole(3, 0.0);
        vector<Vec3> positions(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double charge = (i % 2 == 0) ? charge1 : charge2;
            double pol = (i % 2 == 0) ? pol1 : pol2;
            force->addParticle(charge, zeroDipole, pol,
                               TholeDipoleForce::NoAxisType, -1, -1, -1);
            positions[i] = Vec3(i / 2 * 0.1 + i % 2 * separation, 0.0, 0.0);
        }

        LangevinIntegrator integrator(0.0, 0.1, 0.01);
        Context context(system, integrator, *platform);
        context.setPositions(positions);
        double energy = context.getState(State::Energy).getPotentialEnergy();

        Result result;
        result.cutoff = cutoff;
        result.totalEnergy = energy;
        result.energyPerAtom = energy / numAtoms;
        results.push_back(result);

        cout << "Cutoff = " << cutoff << " nm: Energy = " << energy
             << " kJ/mol (" << result.energyPerAtom << " kJ/mol/atom)" << endl;
    }

    cout << "\n========================================" << endl;
    cout << "PME Cutoff Convergence (1024 atoms)" << endl;
    cout << "Box size: " << boxSize << " nm" << endl;
    cout << "========================================\n" << endl;

    cout << "Cutoff (nm)  Total Energy (kJ/mol)  Energy/Atom (kJ/mol)  Change (%)" << endl;
    cout << "-----------  --------------------  --------------------  ----------" << endl;

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        double changePercent = 0.0;
        if (i > 0) {
            double prevEnergy = results[i-1].totalEnergy;
            changePercent = 100.0 * (r.totalEnergy - prevEnergy) / fabs(prevEnergy);
        }

        printf("%11.2f  %20.8f  %20.10f  %10.4f\n",
               r.cutoff, r.totalEnergy, r.energyPerAtom, changePercent);
    }

    double finalEnergy = results.back().totalEnergy;
    double relativeChange = fabs((results.back().totalEnergy - results[results.size()-2].totalEnergy)
                                / results.back().totalEnergy);

    cout << "\nFinal energy (5.0 nm cutoff): " << finalEnergy << " kJ/mol" << endl;
    cout << "Relative change (4.0 -> 5.0 nm): " << (relativeChange * 100.0) << " %" << endl;

    cout << "\n✅ PME cutoff convergence test complete!" << endl;
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
