#ifndef THOLE_DIPOLE_TEST_COMMON_H_
#define THOLE_DIPOLE_TEST_COMMON_H_

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/CustomBondForce.h"
#include "TholeDipoleForce.h"
#include "openmm/System.h"
#include "openmm/LangevinIntegrator.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/AmoebaMultipoleForce.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

using namespace OpenMM;
using namespace TholeDipolePlugin;
using namespace std;

extern ReferencePlatform* platform;

#define ASSERT_EQUAL_TOL_MOD(expected, found, tol, testname) {double _scale_ = std::abs(expected) > 1.0 ? std::abs(expected) : 1.0; if (!(std::abs((expected)-(found))/_scale_ <= (tol))) {std::stringstream details; details << testname << " Expected "<<(expected)<<", found "<<(found); throwException(__FILE__, __LINE__, details.str());}};

#define ASSERT_EQUAL_VEC_MOD(expected, found, tol, testname) {double _norm_ = std::sqrt(expected.dot(expected)); double _scale_ = _norm_ > 1.0 ? _norm_ : 1.0; if ((std::abs((expected[0])-(found[0]))/_scale_ > (tol)) || (std::abs((expected[1])-(found[1]))/_scale_ > (tol)) || (std::abs((expected[2])-(found[2]))/_scale_ > (tol))) {std::stringstream details; details << testname << " Expected "<<(expected)<<", found "<<(found); throwException(__FILE__, __LINE__, details.str());}};

using namespace OpenMM;
using namespace TholeDipolePlugin;
using namespace std;

const double TOL = 1e-4;

// setup for 2 ammonia molecules
void setupTholeDipoleAmmonia(System& system, TholeDipoleForce* tholeDipoleForce, TholeDipoleForce::NonbondedMethod nonbondedMethod,
                            TholeDipoleForce::PolarizationType polarizationType,
                            double cutoff, int inputPmeGridDimension);

void getForcesEnergyTholeDipoleAmmonia(Context& context, std::vector<Vec3>& forces, double& energy);

// compare forces and energies 
void compareForcesEnergy(std::string& testName, double expectedEnergy, double energy,
                        const std::vector<Vec3>& expectedForces,
                        const std::vector<Vec3>& forces, double tolerance);

// setup for box of 4 water molecules -- used to test PME
void setupAndGetForcesEnergyTholeDipoleWater(TholeDipoleForce::NonbondedMethod nonbondedMethod,
                                           TholeDipoleForce::PolarizationType polarizationType,
                                           double cutoff, int inputPmeGridDimension, std::vector<Vec3>& forces,
                                           double& energy);

// AMOEBA comparison functions
AmoebaMultipoleForce* createEquivalentAmoebaForce(TholeDipoleForce* tholeDipoleForce);

void compareForces(const string& testName,
                   System& tholeDipoleSystem,
                   System& amoebaSystem,
                   const vector<Vec3>& positions,
                   double energyTolerance = 1e-4,
                   double forceTolerance = 1e-4);

#endif // THOLE_DIPOLE_TEST_COMMON_H_
