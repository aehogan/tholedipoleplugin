#ifndef CUDA_TEST_COMMON_H_
#define CUDA_TEST_COMMON_H_

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/CustomBondForce.h"
#include "TholeDipoleForce.h"
#include "openmm/System.h"
#include "openmm/LangevinIntegrator.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/Platform.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace OpenMM;
using namespace TholeDipolePlugin;
using namespace std;

extern Platform* cudaPlatform;
extern Platform* referencePlatform;

const double TOL = 1e-4;

// Clone a TholeDipoleForce
TholeDipoleForce* cloneTholeDipoleForce(const TholeDipoleForce* origForce);

// Compare CUDA vs Reference forces and energies
void assertForcesAndEnergiesMatch(System& system, vector<Vec3>& positions,
                                   double energyTol, double forceTol);

// Setup for 2 ammonia molecules
void setupTholeDipoleAmmonia(System& system, TholeDipoleForce* tholeDipoleForce,
                            TholeDipoleForce::NonbondedMethod nonbondedMethod,
                            TholeDipoleForce::PolarizationType polarizationType,
                            double cutoff, int inputPmeGridDimension);

// Get ammonia positions
vector<Vec3> getAmmoniaPositions();

// Setup water PME system
void setupWaterPME(System& system, TholeDipoleForce* force,
                   TholeDipoleForce::PolarizationType polType,
                   double cutoff, int pmeGrid);

// Get water PME positions
vector<Vec3> getWaterPMEPositions();

#endif // CUDA_TEST_COMMON_H_
