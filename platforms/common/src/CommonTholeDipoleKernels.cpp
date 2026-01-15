/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
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

#ifdef WIN32
  #define _USE_MATH_DEFINES
#endif
#include "CommonTholeDipoleKernels.h"
#include "CommonTholeDipoleKernelSources.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"

#include <algorithm>
#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace TholeDipolePlugin;
using namespace OpenMM;
using namespace std;

constexpr int CommonCalcTholeDipoleForceKernel::PmeOrder;
constexpr int CommonCalcTholeDipoleForceKernel::MaxPrevDIISDipoles;

static const double ONE_4PI_EPS0 = 138.935456;

static void setPeriodicBoxArgs(ComputeContext& cc, ComputeKernel kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
}

class CommonCalcTholeDipoleForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const TholeDipoleForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, polarity1, polarity2;
        int axis1, axis2, atomZ1, atomZ2, atomX1, atomX2, atomY1, atomY2;
        vector<double> dipole1, dipole2;
        force.getParticleParameters(particle1, charge1, dipole1, polarity1, axis1, atomZ1, atomX1, atomY1);
        force.getParticleParameters(particle2, charge2, dipole2, polarity2, axis2, atomZ2, atomX2, atomY2);
        if (charge1 != charge2 || polarity1 != polarity2 || axis1 != axis2)
            return false;
        for (int i = 0; i < (int) dipole1.size(); ++i)
            if (dipole1[i] != dipole2[i])
                return false;
        return true;
    }
    int getNumParticleGroups() {
        return 4*force.getNumParticles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle = index/4;
        int type = index - 4*particle;
        force.getCovalentMap(particle, TholeDipoleForce::CovalentType(type), particles);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return ((group1%4) == (group2%4));
    }
private:
    const TholeDipoleForce& force;
};

CommonCalcTholeDipoleForceKernel::~CommonCalcTholeDipoleForceKernel() {
}

void CommonCalcTholeDipoleForceKernel::initialize(const System& system, const TholeDipoleForce& force) {
    ContextSelector selector(cc);

    numParticles = force.getNumParticles();
    if (numParticles == 0)
        return;

    usePME = (force.getNonbondedMethod() == TholeDipoleForce::PME);
    polarizationType = force.getPolarizationType();
    dampingType = force.getTholeDampingType();
    tholeDampingParameter = force.getTholeDampingParameter();
    cutoffDistance = force.getCutoffDistance();
    maxInducedIterations = force.getMutualInducedMaxIterations();
    inducedEpsilon = force.getMutualInducedTargetEpsilon();

    ArrayInterface& posq = cc.getPosq();
    vector<mm_double4> temp(posq.getSize());
    mm_float4* posqf = (mm_float4*) &temp[0];
    mm_double4* posqd = (mm_double4*) &temp[0];
    vector<mm_float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> localDipolesVec;
    vector<mm_int4> multipoleParticlesVec;

    // Arrays are stored in ORIGINAL index order. Kernels will use atomIndex
    // to map from GPU index to original index when accessing these arrays.
    double totalCharge = 0.0;
    for (int i = 0; i < numParticles; i++) {
        double charge, polarity;
        int axisType, atomZ, atomX, atomY;
        vector<double> dipole;
        force.getParticleParameters(i, charge, dipole, polarity, axisType, atomZ, atomX, atomY);
        totalCharge += charge;
        if (cc.getUseDoublePrecision())
            posqd[i] = mm_double4(0, 0, 0, charge);
        else
            posqf[i] = mm_float4(0, 0, 0, (float) charge);

        // Store axis atom indices as original indices - kernel will map them
        double damp = (polarity > 0 ? pow(polarity, 1.0/6.0) : 0.0);
        dampingAndTholeVec.push_back(mm_float2((float) damp, (float) tholeDampingParameter));
        polarizabilityVec.push_back((float) polarity);
        multipoleParticlesVec.push_back(mm_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back((float) dipole[j]);
    }

    int paddedNumAtoms = cc.getPaddedNumAtoms();
    for (int i = numParticles; i < paddedNumAtoms; i++) {
        dampingAndTholeVec.push_back(mm_float2(0, 0));
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(mm_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
    }

    dampingAndThole.initialize<mm_float2>(cc, paddedNumAtoms, "dampingAndThole");
    polarizability.initialize<float>(cc, paddedNumAtoms, "polarizability");
    multipoleParticles.initialize<mm_int4>(cc, paddedNumAtoms, "multipoleParticles");
    localDipoles.initialize<float>(cc, 3*paddedNumAtoms, "localDipoles");
    lastPositions.initialize(cc, cc.getPosq().getSize(), cc.getPosq().getElementSize(), "lastPositions");
    dampingAndThole.upload(dampingAndTholeVec);
    polarizability.upload(polarizabilityVec);
    multipoleParticles.upload(multipoleParticlesVec);
    localDipoles.upload(localDipolesVec);
    posq.upload(&temp[0]);

    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    labDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "labDipoles");
    field.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "field");
    torque.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "torque");
    inducedDipole.initialize(cc, 3*paddedNumAtoms, elementSize, "inducedDipole");

    // Inverse atom index maps original indices to GPU indices
    // Initially identity, will be updated when atoms are reordered
    inverseAtomIndex.initialize<int>(cc, paddedNumAtoms, "inverseAtomIndex");
    vector<int> invIdxVec(paddedNumAtoms);
    for (int i = 0; i < paddedNumAtoms; i++)
        invIdxVec[i] = i;  // Identity mapping initially
    inverseAtomIndex.upload(invIdxVec);

    cc.addAutoclearBuffer(field);
    cc.addAutoclearBuffer(torque);

    // Initialize covalentFlags with placeholder size (will be resized in initializeScaleFactors)
    covalentFlags.initialize<mm_int2>(cc, 1, "covalentFlags");

    // Initialize pair scale factors array
    // mScale values: Cov12=0, Cov13=0, Cov14=0.5, Cov15+=1.0
    // Build in original atom order - will be reordered in initializeScaleFactors()
    originalScaleFactors.resize(paddedNumAtoms * paddedNumAtoms, 1.0f);
    for (int i = 0; i < numParticles; i++) {
        originalScaleFactors[i * paddedNumAtoms + i] = 0.0f;  // Self-interaction excluded
        for (int type = 0; type < 4; type++) {
            vector<int> covalentAtoms;
            force.getCovalentMap(i, TholeDipoleForce::CovalentType(type), covalentAtoms);
            float mScale;
            switch(type) {
                case 0: mScale = 0.0f; break;  // Covalent12
                case 1: mScale = 0.0f; break;  // Covalent13
                case 2: mScale = 0.5f; break;  // Covalent14
                default: mScale = 1.0f; break; // Covalent15+
            }
            for (int atom : covalentAtoms) {
                originalScaleFactors[i * paddedNumAtoms + atom] = mScale;
                originalScaleFactors[atom * paddedNumAtoms + i] = mScale;
            }
        }
    }
    // Initialize with placeholder - will be properly set up in initializeScaleFactors()
    pairScaleFactors.initialize<float>(cc, paddedNumAtoms * paddedNumAtoms, "pairScaleFactors");

    // inducedField is needed for Mutual polarization and for PME (induced potential kernel writes to it)
    if (polarizationType == TholeDipoleForce::Mutual || usePME) {
        inducedField.initialize(cc, 3*paddedNumAtoms, sizeof(long long), "inducedField");
        cc.addAutoclearBuffer(inducedField);
    }

    if (polarizationType == TholeDipoleForce::Mutual) {
        inducedDipoleErrors.initialize(cc, cc.getNumThreadBlocks(), sizeof(mm_float2), "inducedDipoleErrors");
        prevDipoles.initialize(cc, 3*numParticles*MaxPrevDIISDipoles, elementSize, "prevDipoles");
        prevErrors.initialize(cc, 3*numParticles*MaxPrevDIISDipoles, elementSize, "prevErrors");
        diisMatrix.initialize<float>(cc, MaxPrevDIISDipoles*MaxPrevDIISDipoles, "diisMatrix");
        diisCoefficients.initialize<float>(cc, MaxPrevDIISDipoles+1, "diisCoefficients");
    }

    // Build exclusion lists and record covalent flag values
    // Include self-exclusions to ensure NonbondedUtilities has valid data
    vector<set<int>> exclusions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        exclusions[i].insert(i);  // Self-exclusion
        for (int type = 0; type < 4; type++) {
            vector<int> covalentAtoms;
            force.getCovalentMap(i, TholeDipoleForce::CovalentType(type), covalentAtoms);
            for (int atom : covalentAtoms) {
                exclusions[i].insert(atom);
                exclusions[atom].insert(i);
                covalentFlagValues.push_back(mm_int4(i, atom, type, 0));
            }
        }
    }

    // Create compute kernels
    map<string, string> defines;
    defines["NUM_ATOMS"] = cc.intToString(numParticles);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(paddedNumAtoms);
    defines["NUM_BLOCKS"] = cc.intToString(cc.getNumAtomBlocks());
    defines["ENERGY_SCALE_FACTOR"] = cc.doubleToString(ONE_4PI_EPS0);

    if (polarizationType == TholeDipoleForce::Direct)
        defines["DIRECT_POLARIZATION"] = "";
    else if (polarizationType == TholeDipoleForce::Mutual) {
        defines["MUTUAL_POLARIZATION"] = "";
        defines["MAX_PREV_DIIS_DIPOLES"] = cc.intToString(MaxPrevDIISDipoles);
    }

    if (dampingType == TholeDipoleForce::NoDamping)
        defines["NO_DAMPING"] = "";
    else if (dampingType == TholeDipoleForce::Exponential)
        defines["EXPONENTIAL_DAMPING"] = "";
    else if (dampingType == TholeDipoleForce::Linear)
        defines["LINEAR_DAMPING"] = "";
    else
        defines["AMOEBA_DAMPING"] = "";

    defines["THOLE_PARAMETER"] = cc.doubleToString(tholeDampingParameter);
    if (force.getDampPermanentInducedField())
        defines["DAMP_PERM_IND_FIELD"] = "";
    defines["TILE_SIZE"] = cc.intToString(ComputeContext::TileSize);

    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    int numExclusionTiles = nb.getExclusionTiles().getSize();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(numExclusionTiles);
    int numContexts = cc.getNumContexts();
    int startExclusionIndex = cc.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (cc.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = cc.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = cc.intToString(endExclusionIndex);

    if (usePME) {
        int nx, ny, nz;
        force.getPMEParameters(pmeAlpha, nx, ny, nz);
        if (nx == 0 || pmeAlpha == 0) {
            NonbondedForce nb_force;
            nb_force.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb_force.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb_force, pmeAlpha, gridSizeX, gridSizeY, gridSizeZ, false);
            gridSizeX = cc.findLegalFFTDimension(gridSizeX);
            gridSizeY = cc.findLegalFFTDimension(gridSizeY);
            gridSizeZ = cc.findLegalFFTDimension(gridSizeZ);
        } else {
            gridSizeX = cc.findLegalFFTDimension(nx);
            gridSizeY = cc.findLegalFFTDimension(ny);
            gridSizeZ = cc.findLegalFFTDimension(nz);
        }
        defines["EWALD_ALPHA"] = cc.doubleToString(pmeAlpha);
        defines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
        defines["USE_EWALD"] = "";
        defines["USE_CUTOFF"] = "";
        defines["USE_PERIODIC"] = "";
        defines["CUTOFF_SQUARED"] = cc.doubleToString(cutoffDistance*cutoffDistance);
        defines["PME_ORDER"] = cc.intToString(PmeOrder);
        defines["GRID_SIZE_X"] = cc.intToString(gridSizeX);
        defines["GRID_SIZE_Y"] = cc.intToString(gridSizeY);
        defines["GRID_SIZE_Z"] = cc.intToString(gridSizeZ);

        // Allocate PME arrays
        int gridElements = gridSizeX * gridSizeY * gridSizeZ;
        pmeGrid1.initialize(cc, gridElements, 2*elementSize, "pmeGrid1");
        pmeGrid2.initialize(cc, gridElements, 2*elementSize, "pmeGrid2");
        if (useFixedPointChargeSpreading()) {
            pmeGridLong.initialize(cc, 2*gridElements, sizeof(long long), "pmeGridLong");
            cc.addAutoclearBuffer(pmeGridLong);
            defines["USE_FIXED_POINT_CHARGE_SPREADING"] = "";
        }
        pmeBsplineModuliX.initialize(cc, gridSizeX, elementSize, "pmeBsplineModuliX");
        pmeBsplineModuliY.initialize(cc, gridSizeY, elementSize, "pmeBsplineModuliY");
        pmeBsplineModuliZ.initialize(cc, gridSizeZ, elementSize, "pmeBsplineModuliZ");
        pmePhi.initialize(cc, 10*paddedNumAtoms, elementSize, "pmePhi");
        pmePhid.initialize(cc, 10*paddedNumAtoms, elementSize, "pmePhid");
        pmeCphi.initialize(cc, 10*paddedNumAtoms, elementSize, "pmeCphi");
        sphericalDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "sphericalDipoles");
        fracDipoles.initialize(cc, 3*paddedNumAtoms, elementSize, "fracDipoles");
        pmeEnergyBuffer.initialize<long long>(cc, 1, "pmeEnergyBuffer");

        initializeBSplineModuli();

        // Create PME kernels (using the same program compiled later)
    }

    int maxThreads = max(32, nb.getForceThreadBlockSize());
    int fixedThreadMemory = 10*elementSize + 2*sizeof(float) + sizeof(mm_int2);
    int fixedFieldThreads = min(maxThreads, cc.computeThreadBlockSize(fixedThreadMemory));
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(fixedFieldThreads);

    ComputeProgram program = cc.compileProgram(CommonTholeDipoleKernelSources::tholeDipoleForce, defines);
    computeMomentsKernel = program->createKernel("computeLabFrameMoments");
    computeMomentsKernel->addArg(cc.getPosq());
    computeMomentsKernel->addArg(multipoleParticles);
    computeMomentsKernel->addArg(localDipoles);
    computeMomentsKernel->addArg(labDipoles);
    computeMomentsKernel->addArg(cc.getAtomIndexArray());
    computeMomentsKernel->addArg(inverseAtomIndex);
    if (usePME) {
        computeMomentsKernel->addArg();  // periodicBoxSize
        computeMomentsKernel->addArg();  // invPeriodicBoxSize
        computeMomentsKernel->addArg();  // periodicBoxVecX
        computeMomentsKernel->addArg();  // periodicBoxVecY
        computeMomentsKernel->addArg();  // periodicBoxVecZ
    }

    recordInducedDipolesKernel = program->createKernel("recordInducedDipoles");
    recordInducedDipolesKernel->addArg(field);
    recordInducedDipolesKernel->addArg(inducedDipole);
    recordInducedDipolesKernel->addArg(polarizability);

    mapTorqueKernel = program->createKernel("mapTorqueToForce");
    mapTorqueKernel->addArg(cc.getLongForceBuffer());
    mapTorqueKernel->addArg(torque);
    mapTorqueKernel->addArg(cc.getPosq());
    mapTorqueKernel->addArg(multipoleParticles);
    mapTorqueKernel->addArg(cc.getAtomIndexArray());
    mapTorqueKernel->addArg(inverseAtomIndex);
    if (usePME) {
        mapTorqueKernel->addArg();  // periodicBoxSize
        mapTorqueKernel->addArg();  // invPeriodicBoxSize
        mapTorqueKernel->addArg();  // periodicBoxVecX
        mapTorqueKernel->addArg();  // periodicBoxVecY
        mapTorqueKernel->addArg();  // periodicBoxVecZ
    }

    computeFixedFieldKernel = program->createKernel("computeFixedField");
    computeFixedFieldKernel->addArg(field);
    computeFixedFieldKernel->addArg(cc.getPosq());
    computeFixedFieldKernel->addArg(covalentFlags);
    computeFixedFieldKernel->addArg(nb.getExclusionTiles());
    computeFixedFieldKernel->addArg();  // startTileIndex
    computeFixedFieldKernel->addArg();  // numTiles
    computeFixedFieldKernel->addArg(labDipoles);
    computeFixedFieldKernel->addArg(dampingAndThole);
    computeFixedFieldKernel->addArg(pairScaleFactors);
    if (usePME) {
        computeFixedFieldKernel->addArg();  // periodicBoxSize
        computeFixedFieldKernel->addArg();  // invPeriodicBoxSize
        computeFixedFieldKernel->addArg();  // periodicBoxVecX
        computeFixedFieldKernel->addArg();  // periodicBoxVecY
        computeFixedFieldKernel->addArg();  // periodicBoxVecZ
    }

    electrostaticsKernel = program->createKernel("computeElectrostatics");
    electrostaticsKernel->addArg(cc.getLongForceBuffer());
    electrostaticsKernel->addArg(torque);
    electrostaticsKernel->addArg(cc.getEnergyBuffer());
    electrostaticsKernel->addArg(cc.getPosq());
    electrostaticsKernel->addArg(covalentFlags);
    electrostaticsKernel->addArg(nb.getExclusionTiles());
    electrostaticsKernel->addArg();  // startTileIndex
    electrostaticsKernel->addArg();  // numTiles
    electrostaticsKernel->addArg(labDipoles);
    electrostaticsKernel->addArg(inducedDipole);
    electrostaticsKernel->addArg(dampingAndThole);
    electrostaticsKernel->addArg(pairScaleFactors);
    if (usePME) {
        electrostaticsKernel->addArg();  // periodicBoxSize
        electrostaticsKernel->addArg();  // invPeriodicBoxSize
        electrostaticsKernel->addArg();  // periodicBoxVecX
        electrostaticsKernel->addArg();  // periodicBoxVecY
        electrostaticsKernel->addArg();  // periodicBoxVecZ

        // Create PME-specific kernels
        pmeTransformMultipolesKernel = program->createKernel("pmeTransformMultipoles");
        pmeTransformMultipolesKernel->addArg(labDipoles);
        pmeTransformMultipolesKernel->addArg(fracDipoles);
        pmeTransformMultipolesKernel->addArg();  // recipBoxVecX
        pmeTransformMultipolesKernel->addArg();  // recipBoxVecY
        pmeTransformMultipolesKernel->addArg();  // recipBoxVecZ

        pmeSpreadFixedMultipolesKernel = program->createKernel("pmeSpreadFixedMultipoles");
        pmeSpreadFixedMultipolesKernel->addArg(cc.getPosq());
        pmeSpreadFixedMultipolesKernel->addArg(fracDipoles);
        if (useFixedPointChargeSpreading())
            pmeSpreadFixedMultipolesKernel->addArg(pmeGridLong);
        else
            pmeSpreadFixedMultipolesKernel->addArg(pmeGrid1);
        pmeSpreadFixedMultipolesKernel->addArg();  // periodicBoxVecX
        pmeSpreadFixedMultipolesKernel->addArg();  // periodicBoxVecY
        pmeSpreadFixedMultipolesKernel->addArg();  // periodicBoxVecZ
        pmeSpreadFixedMultipolesKernel->addArg();  // recipBoxVecX
        pmeSpreadFixedMultipolesKernel->addArg();  // recipBoxVecY
        pmeSpreadFixedMultipolesKernel->addArg();  // recipBoxVecZ

        pmeConvolutionKernel = program->createKernel("pmeReciprocalConvolution");
        pmeConvolutionKernel->addArg(pmeGrid2);
        pmeConvolutionKernel->addArg(pmeBsplineModuliX);
        pmeConvolutionKernel->addArg(pmeBsplineModuliY);
        pmeConvolutionKernel->addArg(pmeBsplineModuliZ);
        pmeConvolutionKernel->addArg();  // periodicBoxVecX
        pmeConvolutionKernel->addArg();  // periodicBoxVecY
        pmeConvolutionKernel->addArg();  // periodicBoxVecZ
        pmeConvolutionKernel->addArg();  // recipBoxVecX
        pmeConvolutionKernel->addArg();  // recipBoxVecY
        pmeConvolutionKernel->addArg();  // recipBoxVecZ
        pmeConvolutionKernel->addArg(pmeEnergyBuffer);

        pmeFixedPotentialKernel = program->createKernel("pmeComputeFixedPotentialFromGrid");
        pmeFixedPotentialKernel->addArg(pmeGrid1);
        pmeFixedPotentialKernel->addArg(pmePhi);
        pmeFixedPotentialKernel->addArg(field);
        pmeFixedPotentialKernel->addArg(cc.getPosq());
        pmeFixedPotentialKernel->addArg(labDipoles);
        pmeFixedPotentialKernel->addArg();  // periodicBoxVecX
        pmeFixedPotentialKernel->addArg();  // periodicBoxVecY
        pmeFixedPotentialKernel->addArg();  // periodicBoxVecZ
        pmeFixedPotentialKernel->addArg();  // recipBoxVecX
        pmeFixedPotentialKernel->addArg();  // recipBoxVecY
        pmeFixedPotentialKernel->addArg();  // recipBoxVecZ

        pmeRecipForceKernel = program->createKernel("pmeRecipForceAndTorque");
        pmeRecipForceKernel->addArg(pmePhi);
        pmeRecipForceKernel->addArg(pmePhid);
        pmeRecipForceKernel->addArg(cc.getPosq());
        pmeRecipForceKernel->addArg(labDipoles);
        pmeRecipForceKernel->addArg(fracDipoles);
        pmeRecipForceKernel->addArg(inducedDipole);
        pmeRecipForceKernel->addArg(cc.getLongForceBuffer());
        pmeRecipForceKernel->addArg(torque);
        pmeRecipForceKernel->addArg();  // recipBoxVecX
        pmeRecipForceKernel->addArg();  // recipBoxVecY
        pmeRecipForceKernel->addArg();  // recipBoxVecZ

        // Kernels for computing induced potential (needed for force calculation)
        pmeSpreadInducedDipolesKernel = program->createKernel("pmeSpreadInducedDipoles");
        pmeSpreadInducedDipolesKernel->addArg(cc.getPosq());
        pmeSpreadInducedDipolesKernel->addArg(inducedDipole);
        if (useFixedPointChargeSpreading())
            pmeSpreadInducedDipolesKernel->addArg(pmeGridLong);
        else
            pmeSpreadInducedDipolesKernel->addArg(pmeGrid1);
        pmeSpreadInducedDipolesKernel->addArg();  // periodicBoxVecX
        pmeSpreadInducedDipolesKernel->addArg();  // periodicBoxVecY
        pmeSpreadInducedDipolesKernel->addArg();  // periodicBoxVecZ
        pmeSpreadInducedDipolesKernel->addArg();  // recipBoxVecX
        pmeSpreadInducedDipolesKernel->addArg();  // recipBoxVecY
        pmeSpreadInducedDipolesKernel->addArg();  // recipBoxVecZ

        if (useFixedPointChargeSpreading()) {
            pmeFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
            pmeFinishSpreadChargeKernel->addArg(pmeGridLong);
            pmeFinishSpreadChargeKernel->addArg(pmeGrid1);
        }

        pmeInducedPotentialKernel = program->createKernel("pmeComputeInducedPotentialFromGrid");
        pmeInducedPotentialKernel->addArg(pmeGrid1);
        pmeInducedPotentialKernel->addArg(pmePhid);
        pmeInducedPotentialKernel->addArg(inducedField);
        pmeInducedPotentialKernel->addArg(cc.getPosq());
        pmeInducedPotentialKernel->addArg(inducedDipole);
        pmeInducedPotentialKernel->addArg();  // periodicBoxVecX
        pmeInducedPotentialKernel->addArg();  // periodicBoxVecY
        pmeInducedPotentialKernel->addArg();  // periodicBoxVecZ
        pmeInducedPotentialKernel->addArg();  // recipBoxVecX
        pmeInducedPotentialKernel->addArg();  // recipBoxVecY
        pmeInducedPotentialKernel->addArg();  // recipBoxVecZ
    }

    if (polarizationType == TholeDipoleForce::Mutual) {
        defines["MAX_PREV_DIIS_DIPOLES"] = cc.intToString(MaxPrevDIISDipoles);
        int inducedThreadMemory = 8*elementSize + 2*sizeof(float) + sizeof(mm_int2);
        int inducedFieldThreads = min(maxThreads, cc.computeThreadBlockSize(inducedThreadMemory));
        // Use the computed block size for DIIS kernels
        diisBlockSize = inducedFieldThreads;
        defines["THREAD_BLOCK_SIZE"] = cc.intToString(inducedFieldThreads);

        program = cc.compileProgram(CommonTholeDipoleKernelSources::tholeDipoleForce, defines);
        computeInducedFieldKernel = program->createKernel("computeInducedField");
        computeInducedFieldKernel->addArg(inducedField);
        computeInducedFieldKernel->addArg(cc.getPosq());
        computeInducedFieldKernel->addArg(nb.getExclusionTiles());
        computeInducedFieldKernel->addArg(inducedDipole);
        computeInducedFieldKernel->addArg();  // startTileIndex
        computeInducedFieldKernel->addArg();  // numTiles
        computeInducedFieldKernel->addArg(dampingAndThole);
        if (usePME) {
            computeInducedFieldKernel->addArg();  // periodicBoxSize
            computeInducedFieldKernel->addArg();  // invPeriodicBoxSize
            computeInducedFieldKernel->addArg();  // periodicBoxVecX
            computeInducedFieldKernel->addArg();  // periodicBoxVecY
            computeInducedFieldKernel->addArg();  // periodicBoxVecZ
        }

        updateInducedFieldKernel = program->createKernel("updateInducedFieldByDIIS");
        updateInducedFieldKernel->addArg(inducedDipole);
        updateInducedFieldKernel->addArg(prevDipoles);
        updateInducedFieldKernel->addArg(diisCoefficients);
        updateInducedFieldKernel->addArg();  // iteration

        recordDIISDipolesKernel = program->createKernel("recordInducedDipolesForDIIS");
        recordDIISDipolesKernel->addArg(field);
        recordDIISDipolesKernel->addArg(inducedField);
        recordDIISDipolesKernel->addArg(polarizability);
        recordDIISDipolesKernel->addArg(inducedDipole);
        recordDIISDipolesKernel->addArg(inducedDipoleErrors);
        recordDIISDipolesKernel->addArg(prevDipoles);
        recordDIISDipolesKernel->addArg(prevErrors);
        recordDIISDipolesKernel->addArg(diisMatrix);
        recordDIISDipolesKernel->addArg();  // iteration

        buildMatrixKernel = program->createKernel("computeDIISMatrix");
        buildMatrixKernel->addArg(prevErrors);
        buildMatrixKernel->addArg();  // iteration
        buildMatrixKernel->addArg(diisMatrix);

        solveMatrixKernel = program->createKernel("solveDIISMatrix");
        solveMatrixKernel->addArg();  // iteration
        solveMatrixKernel->addArg(diisMatrix);
        solveMatrixKernel->addArg(diisCoefficients);
    }

    computePotentialKernel = program->createKernel("computePotentialAtPoints");
    computePotentialKernel->addArg(cc.getPosq());
    computePotentialKernel->addArg(labDipoles);
    computePotentialKernel->addArg(inducedDipole);
    for (int i = 0; i < 8; i++)
        computePotentialKernel->addArg();

    vector<vector<int>> exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++)
        exclusionList[i].assign(exclusions[i].begin(), exclusions[i].end());
    nb.addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusionList, "", force.getForceGroup());
    nb.setUsePadding(false);
    cc.addForce(new ForceInfo(force));
    hasInitializedKernels = true;
    hasInitializedScaleFactors = false;
}

void CommonCalcTholeDipoleForceKernel::initializeScaleFactors() {
    hasInitializedScaleFactors = true;
    NonbondedUtilities& nb = cc.getNonbondedUtilities();

    vector<mm_int2> exclusionTiles;
    nb.getExclusionTiles().download(exclusionTiles);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTiles.size(); i++) {
        mm_int2 tile = exclusionTiles[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }

    covalentFlags.resize(nb.getExclusions().getSize());
    vector<mm_int2> covalentFlagsVec(nb.getExclusions().getSize(), mm_int2(0, 0));
    for (mm_int4 values : covalentFlagValues) {
        int atom1 = values.x;
        int atom2 = values.y;
        int type = values.z;
        int x = atom1/ComputeContext::TileSize;
        int offset1 = atom1 - x*ComputeContext::TileSize;
        int y = atom2/ComputeContext::TileSize;
        int offset2 = atom2 - y*ComputeContext::TileSize;
        // mScale: [0,0,0.5,1.0] for types 0,1,2,3 -> encode as: type<2 means scale=0, type=2 means scale=0.5, type=3 means scale=1.0
        // We'll encode the covalent type directly and decode in the kernel
        int flag = type;
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*ComputeContext::TileSize;
            covalentFlagsVec[index+offset1].x |= flag << (2*offset2);
            covalentFlagsVec[index+offset2].x |= flag << (2*offset1);
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*ComputeContext::TileSize;
            covalentFlagsVec[index+offset1].x |= flag << (2*offset2);
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*ComputeContext::TileSize;
            covalentFlagsVec[index+offset2].x |= flag << (2*offset1);
        }
    }
    covalentFlags.upload(covalentFlagsVec);

    // Reorder pairScaleFactors to match GPU atom ordering
    // cc.getAtomIndex()[gpuIndex] = originalIndex
    const vector<int>& atomIndex = cc.getAtomIndex();
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    vector<float> scaleFactorsVec(paddedNumAtoms * paddedNumAtoms, 1.0f);
    for (int i = 0; i < numParticles; i++) {
        int orig_i = atomIndex[i];
        for (int j = 0; j < numParticles; j++) {
            int orig_j = atomIndex[j];
            scaleFactorsVec[i * paddedNumAtoms + j] = originalScaleFactors[orig_i * paddedNumAtoms + orig_j];
        }
    }
    pairScaleFactors.upload(scaleFactorsVec);
}

void CommonCalcTholeDipoleForceKernel::initializeBSplineModuli() {
    int maxSize = max(max(gridSizeX, gridSizeY), gridSizeZ);
    vector<double> bsarray(maxSize + 1, 0.0);

    // Build B-spline coefficients
    double array[PmeOrder];
    double x = 0.0;
    array[0] = 1.0 - x;
    array[1] = x;
    for (int k = 2; k < PmeOrder; k++) {
        double denom = 1.0/k;
        array[k] = x*array[k-1]*denom;
        for (int i = 1; i < k; i++) {
            array[k-i] = ((x+i)*array[k-i-1] + ((k-i+1)-x)*array[k-i])*denom;
        }
        array[0] = (1.0-x)*array[0]*denom;
    }
    for (int i = 2; i <= PmeOrder+1; i++) {
        bsarray[i] = array[i-2];
    }

    // Compute moduli for each dimension
    int gridDims[3] = {gridSizeX, gridSizeY, gridSizeZ};
    ComputeArray* moduliArrays[3] = {&pmeBsplineModuliX, &pmeBsplineModuliY, &pmeBsplineModuliZ};

    for (int dim = 0; dim < 3; dim++) {
        int ndata = gridDims[dim];
        double factor = 2.0 * M_PI / ndata;
        vector<double> moduli(ndata);

        for (int i = 0; i < ndata; i++) {
            double sum1 = 0.0, sum2 = 0.0;
            for (int j = 1; j <= ndata; j++) {
                double arg = factor * i * (j-1);
                sum1 += bsarray[j] * cos(arg);
                sum2 += bsarray[j] * sin(arg);
            }
            moduli[i] = sum1*sum1 + sum2*sum2;
        }

        // Fix small values
        double eps = 1.0e-7;
        if (moduli[0] < eps)
            moduli[0] = 0.5 * moduli[1];
        for (int i = 1; i < ndata-1; i++) {
            if (moduli[i] < eps)
                moduli[i] = 0.5 * (moduli[i-1] + moduli[i+1]);
        }
        if (moduli[ndata-1] < eps)
            moduli[ndata-1] = 0.5 * moduli[ndata-2];

        // Compute and apply zeta coefficient
        int jcut = 50;
        for (int i = 1; i <= ndata; i++) {
            int k = i - 1;
            if (i > ndata/2)
                k = k - ndata;
            double zeta;
            if (k == 0)
                zeta = 1.0;
            else {
                double sum1 = 1.0;
                double sum2 = 1.0;
                double factor2 = M_PI*k/ndata;
                for (int j = 1; j <= jcut; j++) {
                    double arg = factor2/(factor2+M_PI*j);
                    sum1 = sum1 + pow(arg, PmeOrder);
                    sum2 = sum2 + pow(arg, 2*PmeOrder);
                }
                for (int j = 1; j <= jcut; j++) {
                    double arg = factor2/(factor2-M_PI*j);
                    sum1 += pow(arg, PmeOrder);
                    sum2 += pow(arg, 2*PmeOrder);
                }
                zeta = sum2/sum1;
            }
            moduli[i-1] = moduli[i-1]*(zeta*zeta);
        }

        // Upload to GPU
        if (cc.getUseDoublePrecision()) {
            moduliArrays[dim]->upload(moduli);
        } else {
            vector<float> moduliFloat(ndata);
            for (int i = 0; i < ndata; i++)
                moduliFloat[i] = (float) moduli[i];
            moduliArrays[dim]->upload(moduliFloat);
        }
    }
}

double CommonCalcTholeDipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);

    if (numParticles == 0)
        return 0.0;

    if (!hasInitializedScaleFactors)
        initializeScaleFactors();

    NonbondedUtilities& nb = cc.getNonbondedUtilities();

    // Update inverse atom index mapping (original -> GPU) since atoms may have been reordered
    vector<int> invIdx(cc.getPaddedNumAtoms());
    for (int gpuIdx = 0; gpuIdx < numParticles; gpuIdx++) {
        int origIdx = cc.getAtomIndex()[gpuIdx];
        invIdx[origIdx] = gpuIdx;
    }
    // Fill padding with identity mapping
    for (int i = numParticles; i < cc.getPaddedNumAtoms(); i++)
        invIdx[i] = i;
    inverseAtomIndex.upload(invIdx);

    // Compute lab frame moments
    if (usePME)
        setPeriodicBoxArgs(context, computeMomentsKernel, 6);
    computeMomentsKernel->execute(cc.getNumAtoms());

    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();

    computeFixedFieldKernel->setArg(4, startTileIndex);
    computeFixedFieldKernel->setArg(5, numTileIndices);
    electrostaticsKernel->setArg(6, startTileIndex);
    electrostaticsKernel->setArg(7, numTileIndices);

    // Clear field buffer before any fixed field computation
    cc.clearBuffer(field);

    // Set periodic box arguments for PME
    double reciprocalEnergy = 0.0;
    if (usePME) {
        computeReciprocalBoxVectors(context);
        setPeriodicBoxArgs(context, computeFixedFieldKernel, 9);
        setPeriodicBoxArgs(context, electrostaticsKernel, 12);

        // Compute reciprocal space fixed field (adds to field array)
        reciprocalEnergy = computePmeReciprocalField(context, true);
    }

    // Compute direct space fixed field
    computeFixedFieldKernel->execute(numForceThreadBlocks*128, 128);

    // Record induced dipoles (μ_ind = α * E_fixed)
    recordInducedDipolesKernel->execute(cc.getNumAtoms());

    // For mutual polarization, iterate to convergence
    if (polarizationType == TholeDipoleForce::Mutual) {
        computeInducedFieldKernel->setArg(4, startTileIndex);
        computeInducedFieldKernel->setArg(5, numTileIndices);
        if (usePME)
            setPeriodicBoxArgs(context, computeInducedFieldKernel, 7);

        for (int i = 0; i < maxInducedIterations; i++) {
            // Clear induced field buffer at start of iteration
            cc.clearBuffer(inducedField);

            // Compute reciprocal space induced field (adds to inducedField)
            if (usePME)
                computePmeReciprocalField(context, false);

            // Compute direct space induced field (adds to inducedField)
            computeInducedFieldDirect();

            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
    }

    // Compute electrostatic forces and energy
    electrostaticsKernel->execute(numForceThreadBlocks*128, 128);

    // Compute reciprocal space forces and torques from PME potential
    if (usePME) {
        // Compute induced potential (phid) by spreading induced dipoles on grid
        computePmeReciprocalField(context, false);

        if (cc.getUseDoublePrecision()) {
            pmeRecipForceKernel->setArg(8, recipBoxVecXD);
            pmeRecipForceKernel->setArg(9, recipBoxVecYD);
            pmeRecipForceKernel->setArg(10, recipBoxVecZD);
        } else {
            pmeRecipForceKernel->setArg(8, recipBoxVecXF);
            pmeRecipForceKernel->setArg(9, recipBoxVecYF);
            pmeRecipForceKernel->setArg(10, recipBoxVecZF);
        }
        pmeRecipForceKernel->execute(numParticles);
    }

    // Map torques to forces
    if (usePME)
        setPeriodicBoxArgs(context, mapTorqueKernel, 6);
    mapTorqueKernel->execute(cc.getNumAtoms());

    // Compute polarization energy and PME self-energy correction
    double polarizationEnergy = 0.0;
    if (usePME) {
        // Compute induced reciprocal energy: inducedDipole_frac · grad_phi_fixed_frac
        // This uses the already-computed fixed potential (pmePhi), no grid ops needed
        double inducedRecipEnergy = computePmeInducedRecipEnergy();

        // PME reciprocal energy + induced reciprocal energy + self-energy correction
        double selfEnergy = computePmeSelfEnergy();
        polarizationEnergy = reciprocalEnergy + inducedRecipEnergy + selfEnergy;

    } else {
        // NoCutoff: Compute polarization energy: -0.5 * Σ μ_ind · E_fixed
        vector<long long> fieldVec;
        field.download(fieldVec);

        double fieldScale = 1.0 / 0x100000000;
        double muDotE = 0.0;
        int paddedAtoms = cc.getPaddedNumAtoms();

        if (cc.getUseDoublePrecision()) {
            vector<double> indDipoleVec;
            inducedDipole.download(indDipoleVec);
            for (int i = 0; i < numParticles; i++) {
                muDotE += indDipoleVec[3*i] * fieldScale * fieldVec[i];
                muDotE += indDipoleVec[3*i+1] * fieldScale * fieldVec[i + paddedAtoms];
                muDotE += indDipoleVec[3*i+2] * fieldScale * fieldVec[i + 2*paddedAtoms];
            }
        } else {
            vector<float> indDipoleVec;
            inducedDipole.download(indDipoleVec);
            for (int i = 0; i < numParticles; i++) {
                muDotE += indDipoleVec[3*i] * fieldScale * fieldVec[i];
                muDotE += indDipoleVec[3*i+1] * fieldScale * fieldVec[i + paddedAtoms];
                muDotE += indDipoleVec[3*i+2] * fieldScale * fieldVec[i + 2*paddedAtoms];
            }
        }
        polarizationEnergy = -0.5 * ONE_4PI_EPS0 * muDotE;
    }

    // Record positions for tracking changes
    cc.getPosq().copyTo(lastPositions);
    multipolesAreValid = true;

    return polarizationEnergy;
}

void CommonCalcTholeDipoleForceKernel::computeInducedFieldDirect() {
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    computeInducedFieldKernel->execute(numForceThreadBlocks*128, 128);
}

void CommonCalcTholeDipoleForceKernel::setPeriodicBoxArgs(ContextImpl& context, ComputeKernel kernel, int index) {
    Vec3 boxVectors[3];
    context.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);

    if (cc.getUseDoublePrecision()) {
        mm_double4 boxSize = mm_double4(boxVectors[0][0], boxVectors[1][1], boxVectors[2][2], 0.0);
        mm_double4 invBoxSize = mm_double4(1.0/boxVectors[0][0], 1.0/boxVectors[1][1], 1.0/boxVectors[2][2], 0.0);
        mm_double4 boxVecX = mm_double4(boxVectors[0][0], boxVectors[0][1], boxVectors[0][2], 0.0);
        mm_double4 boxVecY = mm_double4(boxVectors[1][0], boxVectors[1][1], boxVectors[1][2], 0.0);
        mm_double4 boxVecZ = mm_double4(boxVectors[2][0], boxVectors[2][1], boxVectors[2][2], 0.0);
        kernel->setArg(index, boxSize);
        kernel->setArg(index+1, invBoxSize);
        kernel->setArg(index+2, boxVecX);
        kernel->setArg(index+3, boxVecY);
        kernel->setArg(index+4, boxVecZ);
    }
    else {
        mm_float4 boxSize = mm_float4((float)boxVectors[0][0], (float)boxVectors[1][1], (float)boxVectors[2][2], 0.0f);
        mm_float4 invBoxSize = mm_float4(1.0f/(float)boxVectors[0][0], 1.0f/(float)boxVectors[1][1], 1.0f/(float)boxVectors[2][2], 0.0f);
        mm_float4 boxVecX = mm_float4((float)boxVectors[0][0], (float)boxVectors[0][1], (float)boxVectors[0][2], 0.0f);
        mm_float4 boxVecY = mm_float4((float)boxVectors[1][0], (float)boxVectors[1][1], (float)boxVectors[1][2], 0.0f);
        mm_float4 boxVecZ = mm_float4((float)boxVectors[2][0], (float)boxVectors[2][1], (float)boxVectors[2][2], 0.0f);
        kernel->setArg(index, boxSize);
        kernel->setArg(index+1, invBoxSize);
        kernel->setArg(index+2, boxVecX);
        kernel->setArg(index+3, boxVecY);
        kernel->setArg(index+4, boxVecZ);
    }
}

void CommonCalcTholeDipoleForceKernel::computeReciprocalBoxVectors(ContextImpl& context) {
    Vec3 boxVectors[3];
    context.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);

    double determinant = boxVectors[0][0]*(boxVectors[1][1]*boxVectors[2][2])
                       - boxVectors[0][0]*boxVectors[1][2]*boxVectors[2][1];
    double scale = 1.0/determinant;
    double recipBoxVectors[3][3];
    recipBoxVectors[0][0] = boxVectors[1][1]*boxVectors[2][2]*scale;
    recipBoxVectors[0][1] = 0;
    recipBoxVectors[0][2] = 0;
    recipBoxVectors[1][0] = -boxVectors[1][0]*boxVectors[2][2]*scale;
    recipBoxVectors[1][1] = boxVectors[0][0]*boxVectors[2][2]*scale;
    recipBoxVectors[1][2] = 0;
    recipBoxVectors[2][0] = (boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale;
    recipBoxVectors[2][1] = -boxVectors[0][0]*boxVectors[2][1]*scale;
    recipBoxVectors[2][2] = boxVectors[0][0]*boxVectors[1][1]*scale;

    if (cc.getUseDoublePrecision()) {
        recipBoxVecXD = mm_double4(recipBoxVectors[0][0], recipBoxVectors[0][1], recipBoxVectors[0][2], 0);
        recipBoxVecYD = mm_double4(recipBoxVectors[1][0], recipBoxVectors[1][1], recipBoxVectors[1][2], 0);
        recipBoxVecZD = mm_double4(recipBoxVectors[2][0], recipBoxVectors[2][1], recipBoxVectors[2][2], 0);
        periodicBoxVecXD = mm_double4(boxVectors[0][0], boxVectors[0][1], boxVectors[0][2], 0);
        periodicBoxVecYD = mm_double4(boxVectors[1][0], boxVectors[1][1], boxVectors[1][2], 0);
        periodicBoxVecZD = mm_double4(boxVectors[2][0], boxVectors[2][1], boxVectors[2][2], 0);
        periodicBoxSizeD = mm_double4(boxVectors[0][0], boxVectors[1][1], boxVectors[2][2], 0);
    }
    else {
        recipBoxVecXF = mm_float4((float)recipBoxVectors[0][0], (float)recipBoxVectors[0][1], (float)recipBoxVectors[0][2], 0);
        recipBoxVecYF = mm_float4((float)recipBoxVectors[1][0], (float)recipBoxVectors[1][1], (float)recipBoxVectors[1][2], 0);
        recipBoxVecZF = mm_float4((float)recipBoxVectors[2][0], (float)recipBoxVectors[2][1], (float)recipBoxVectors[2][2], 0);
        periodicBoxVecXF = mm_float4((float)boxVectors[0][0], (float)boxVectors[0][1], (float)boxVectors[0][2], 0);
        periodicBoxVecYF = mm_float4((float)boxVectors[1][0], (float)boxVectors[1][1], (float)boxVectors[1][2], 0);
        periodicBoxVecZF = mm_float4((float)boxVectors[2][0], (float)boxVectors[2][1], (float)boxVectors[2][2], 0);
        periodicBoxSizeF = mm_float4((float)boxVectors[0][0], (float)boxVectors[1][1], (float)boxVectors[2][2], 0);
    }
}

double CommonCalcTholeDipoleForceKernel::computePmeReciprocalField(ContextImpl& context, bool isFixedField) {
    int gridElements = gridSizeX * gridSizeY * gridSizeZ;

    // Clear grid
    if (useFixedPointChargeSpreading())
        cc.clearBuffer(pmeGridLong);
    else
        cc.clearBuffer(pmeGrid1);

    if (isFixedField) {
        // Transform lab dipoles to fractional coordinates
        if (cc.getUseDoublePrecision()) {
            pmeTransformMultipolesKernel->setArg(2, recipBoxVecXD);
            pmeTransformMultipolesKernel->setArg(3, recipBoxVecYD);
            pmeTransformMultipolesKernel->setArg(4, recipBoxVecZD);
        } else {
            pmeTransformMultipolesKernel->setArg(2, recipBoxVecXF);
            pmeTransformMultipolesKernel->setArg(3, recipBoxVecYF);
            pmeTransformMultipolesKernel->setArg(4, recipBoxVecZF);
        }
        pmeTransformMultipolesKernel->execute(numParticles);

        // Spread fixed multipoles onto grid
        if (cc.getUseDoublePrecision()) {
            pmeSpreadFixedMultipolesKernel->setArg(3, periodicBoxVecXD);
            pmeSpreadFixedMultipolesKernel->setArg(4, periodicBoxVecYD);
            pmeSpreadFixedMultipolesKernel->setArg(5, periodicBoxVecZD);
            pmeSpreadFixedMultipolesKernel->setArg(6, recipBoxVecXD);
            pmeSpreadFixedMultipolesKernel->setArg(7, recipBoxVecYD);
            pmeSpreadFixedMultipolesKernel->setArg(8, recipBoxVecZD);
        } else {
            pmeSpreadFixedMultipolesKernel->setArg(3, periodicBoxVecXF);
            pmeSpreadFixedMultipolesKernel->setArg(4, periodicBoxVecYF);
            pmeSpreadFixedMultipolesKernel->setArg(5, periodicBoxVecZF);
            pmeSpreadFixedMultipolesKernel->setArg(6, recipBoxVecXF);
            pmeSpreadFixedMultipolesKernel->setArg(7, recipBoxVecYF);
            pmeSpreadFixedMultipolesKernel->setArg(8, recipBoxVecZF);
        }
        pmeSpreadFixedMultipolesKernel->execute(numParticles);
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
    } else {
        // Spread induced dipoles onto grid
        if (cc.getUseDoublePrecision()) {
            pmeSpreadInducedDipolesKernel->setArg(3, periodicBoxVecXD);
            pmeSpreadInducedDipolesKernel->setArg(4, periodicBoxVecYD);
            pmeSpreadInducedDipolesKernel->setArg(5, periodicBoxVecZD);
            pmeSpreadInducedDipolesKernel->setArg(6, recipBoxVecXD);
            pmeSpreadInducedDipolesKernel->setArg(7, recipBoxVecYD);
            pmeSpreadInducedDipolesKernel->setArg(8, recipBoxVecZD);
        } else {
            pmeSpreadInducedDipolesKernel->setArg(3, periodicBoxVecXF);
            pmeSpreadInducedDipolesKernel->setArg(4, periodicBoxVecYF);
            pmeSpreadInducedDipolesKernel->setArg(5, periodicBoxVecZF);
            pmeSpreadInducedDipolesKernel->setArg(6, recipBoxVecXF);
            pmeSpreadInducedDipolesKernel->setArg(7, recipBoxVecYF);
            pmeSpreadInducedDipolesKernel->setArg(8, recipBoxVecZF);
        }
        pmeSpreadInducedDipolesKernel->execute(numParticles);
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
    }

    // FFT forward: grid1 -> grid2
    computeFFT(true);

    // Clear energy buffer before convolution
    cc.clearBuffer(pmeEnergyBuffer);

    // Reciprocal convolution
    if (cc.getUseDoublePrecision()) {
        pmeConvolutionKernel->setArg(4, periodicBoxVecXD);
        pmeConvolutionKernel->setArg(5, periodicBoxVecYD);
        pmeConvolutionKernel->setArg(6, periodicBoxVecZD);
        pmeConvolutionKernel->setArg(7, recipBoxVecXD);
        pmeConvolutionKernel->setArg(8, recipBoxVecYD);
        pmeConvolutionKernel->setArg(9, recipBoxVecZD);
    } else {
        pmeConvolutionKernel->setArg(4, periodicBoxVecXF);
        pmeConvolutionKernel->setArg(5, periodicBoxVecYF);
        pmeConvolutionKernel->setArg(6, periodicBoxVecZF);
        pmeConvolutionKernel->setArg(7, recipBoxVecXF);
        pmeConvolutionKernel->setArg(8, recipBoxVecYF);
        pmeConvolutionKernel->setArg(9, recipBoxVecZF);
    }
    pmeConvolutionKernel->execute(gridElements);

    // FFT inverse: grid2 -> grid1
    computeFFT(false);

    // Compute potential from grid
    if (isFixedField) {
        if (cc.getUseDoublePrecision()) {
            pmeFixedPotentialKernel->setArg(5, periodicBoxVecXD);
            pmeFixedPotentialKernel->setArg(6, periodicBoxVecYD);
            pmeFixedPotentialKernel->setArg(7, periodicBoxVecZD);
            pmeFixedPotentialKernel->setArg(8, recipBoxVecXD);
            pmeFixedPotentialKernel->setArg(9, recipBoxVecYD);
            pmeFixedPotentialKernel->setArg(10, recipBoxVecZD);
        } else {
            pmeFixedPotentialKernel->setArg(5, periodicBoxVecXF);
            pmeFixedPotentialKernel->setArg(6, periodicBoxVecYF);
            pmeFixedPotentialKernel->setArg(7, periodicBoxVecZF);
            pmeFixedPotentialKernel->setArg(8, recipBoxVecXF);
            pmeFixedPotentialKernel->setArg(9, recipBoxVecYF);
            pmeFixedPotentialKernel->setArg(10, recipBoxVecZF);
        }
        pmeFixedPotentialKernel->execute(numParticles);
    } else {
        if (cc.getUseDoublePrecision()) {
            pmeInducedPotentialKernel->setArg(5, periodicBoxVecXD);
            pmeInducedPotentialKernel->setArg(6, periodicBoxVecYD);
            pmeInducedPotentialKernel->setArg(7, periodicBoxVecZD);
            pmeInducedPotentialKernel->setArg(8, recipBoxVecXD);
            pmeInducedPotentialKernel->setArg(9, recipBoxVecYD);
            pmeInducedPotentialKernel->setArg(10, recipBoxVecZD);
        } else {
            pmeInducedPotentialKernel->setArg(5, periodicBoxVecXF);
            pmeInducedPotentialKernel->setArg(6, periodicBoxVecYF);
            pmeInducedPotentialKernel->setArg(7, periodicBoxVecZF);
            pmeInducedPotentialKernel->setArg(8, recipBoxVecXF);
            pmeInducedPotentialKernel->setArg(9, recipBoxVecYF);
            pmeInducedPotentialKernel->setArg(10, recipBoxVecZF);
        }
        pmeInducedPotentialKernel->execute(numParticles);
    }

    // Compute reciprocal energy from multipole * phi (only for fixed field)
    double recipEnergy = 0.0;
    if (isFixedField) {
        // E_recip = 0.5 * ELECTRIC * sum_i(charge_i * phi[i,0] + dipole_i · grad_cphi[i])
        // where cphi is the Cartesian phi (transformed from fractional)
        ArrayInterface& posq = cc.getPosq();
        int paddedAtoms = cc.getPaddedNumAtoms();

        // Build transformation matrix a[i][j] = gridDim[j] * recipBox[i][j]
        // This transforms fractional phi gradients to Cartesian coordinates
        double a[3][3];
        if (cc.getUseDoublePrecision()) {
            a[0][0] = gridSizeX * recipBoxVecXD.x;
            a[0][1] = gridSizeY * recipBoxVecXD.y;
            a[0][2] = gridSizeZ * recipBoxVecXD.z;
            a[1][0] = gridSizeX * recipBoxVecYD.x;
            a[1][1] = gridSizeY * recipBoxVecYD.y;
            a[1][2] = gridSizeZ * recipBoxVecYD.z;
            a[2][0] = gridSizeX * recipBoxVecZD.x;
            a[2][1] = gridSizeY * recipBoxVecZD.y;
            a[2][2] = gridSizeZ * recipBoxVecZD.z;
        } else {
            a[0][0] = gridSizeX * recipBoxVecXF.x;
            a[0][1] = gridSizeY * recipBoxVecXF.y;
            a[0][2] = gridSizeZ * recipBoxVecXF.z;
            a[1][0] = gridSizeX * recipBoxVecYF.x;
            a[1][1] = gridSizeY * recipBoxVecYF.y;
            a[1][2] = gridSizeZ * recipBoxVecYF.z;
            a[2][0] = gridSizeX * recipBoxVecZF.x;
            a[2][1] = gridSizeY * recipBoxVecZF.y;
            a[2][2] = gridSizeZ * recipBoxVecZF.z;
        }

        if (cc.getUseDoublePrecision()) {
            vector<mm_double4> posqVec;
            vector<double> labDipoleVec;
            vector<double> phiVec;
            posq.download(posqVec);
            labDipoles.download(labDipoleVec);
            pmePhi.download(phiVec);

            for (int i = 0; i < numParticles; i++) {
                double charge = posqVec[i].w;
                double dipoleX = labDipoleVec[3*i];
                double dipoleY = labDipoleVec[3*i+1];
                double dipoleZ = labDipoleVec[3*i+2];

                double phiVal = phiVec[i];
                double fphiX = phiVec[i + paddedAtoms];
                double fphiY = phiVec[i + paddedAtoms*2];
                double fphiZ = phiVec[i + paddedAtoms*3];

                // Transform fractional phi gradients to Cartesian
                double cphiX = a[0][0]*fphiX + a[0][1]*fphiY + a[0][2]*fphiZ;
                double cphiY = a[1][0]*fphiX + a[1][1]*fphiY + a[1][2]*fphiZ;
                double cphiZ = a[2][0]*fphiX + a[2][1]*fphiY + a[2][2]*fphiZ;

                recipEnergy += charge * phiVal + dipoleX * cphiX + dipoleY * cphiY + dipoleZ * cphiZ;
            }
        } else {
            vector<mm_float4> posqVec;
            vector<float> labDipoleVec;
            vector<float> phiVec;
            posq.download(posqVec);
            labDipoles.download(labDipoleVec);
            pmePhi.download(phiVec);

            for (int i = 0; i < numParticles; i++) {
                double charge = posqVec[i].w;
                double dipoleX = labDipoleVec[3*i];
                double dipoleY = labDipoleVec[3*i+1];
                double dipoleZ = labDipoleVec[3*i+2];

                double phiVal = phiVec[i];
                double fphiX = phiVec[i + paddedAtoms];
                double fphiY = phiVec[i + paddedAtoms*2];
                double fphiZ = phiVec[i + paddedAtoms*3];

                // Transform fractional phi gradients to Cartesian
                double cphiX = a[0][0]*fphiX + a[0][1]*fphiY + a[0][2]*fphiZ;
                double cphiY = a[1][0]*fphiX + a[1][1]*fphiY + a[1][2]*fphiZ;
                double cphiZ = a[2][0]*fphiX + a[2][1]*fphiY + a[2][2]*fphiZ;

                recipEnergy += charge * phiVal + dipoleX * cphiX + dipoleY * cphiY + dipoleZ * cphiZ;
            }
        }
        recipEnergy *= 0.5 * ONE_4PI_EPS0;
    } else {
        // Induced reciprocal energy: 0.5 * ELECTRIC * sum(inducedDipole_frac · grad_phi_fixed_frac)
        // The induced dipole is in Cartesian, phi gradients are in fractional
        // Transform induced dipole to fractional: u_frac = cartToFrac * u_cart
        ArrayInterface& posq = cc.getPosq();
        int paddedAtoms = cc.getPaddedNumAtoms();

        // Build transformation matrix cartToFrac[i][j] = gridSize[j] * recipBoxVec[i][j]
        double cartToFrac[3][3];
        if (cc.getUseDoublePrecision()) {
            cartToFrac[0][0] = gridSizeX * recipBoxVecXD.x;
            cartToFrac[0][1] = gridSizeY * recipBoxVecXD.y;
            cartToFrac[0][2] = gridSizeZ * recipBoxVecXD.z;
            cartToFrac[1][0] = gridSizeX * recipBoxVecYD.x;
            cartToFrac[1][1] = gridSizeY * recipBoxVecYD.y;
            cartToFrac[1][2] = gridSizeZ * recipBoxVecYD.z;
            cartToFrac[2][0] = gridSizeX * recipBoxVecZD.x;
            cartToFrac[2][1] = gridSizeY * recipBoxVecZD.y;
            cartToFrac[2][2] = gridSizeZ * recipBoxVecZD.z;
        } else {
            cartToFrac[0][0] = gridSizeX * recipBoxVecXF.x;
            cartToFrac[0][1] = gridSizeY * recipBoxVecXF.y;
            cartToFrac[0][2] = gridSizeZ * recipBoxVecXF.z;
            cartToFrac[1][0] = gridSizeX * recipBoxVecYF.x;
            cartToFrac[1][1] = gridSizeY * recipBoxVecYF.y;
            cartToFrac[1][2] = gridSizeZ * recipBoxVecYF.z;
            cartToFrac[2][0] = gridSizeX * recipBoxVecZF.x;
            cartToFrac[2][1] = gridSizeY * recipBoxVecZF.y;
            cartToFrac[2][2] = gridSizeZ * recipBoxVecZF.z;
        }

        if (cc.getUseDoublePrecision()) {
            vector<double> inducedDipoleVec;
            vector<double> phiVec;
            inducedDipole.download(inducedDipoleVec);
            pmePhi.download(phiVec);  // Use fixed potential, not induced

            for (int i = 0; i < numParticles; i++) {
                double ux = inducedDipoleVec[3*i];
                double uy = inducedDipoleVec[3*i+1];
                double uz = inducedDipoleVec[3*i+2];

                // Transform to fractional
                double uf0 = cartToFrac[0][0]*ux + cartToFrac[0][1]*uy + cartToFrac[0][2]*uz;
                double uf1 = cartToFrac[1][0]*ux + cartToFrac[1][1]*uy + cartToFrac[1][2]*uz;
                double uf2 = cartToFrac[2][0]*ux + cartToFrac[2][1]*uy + cartToFrac[2][2]*uz;

                // Fractional phi gradients
                double fphiX = phiVec[i + paddedAtoms];
                double fphiY = phiVec[i + paddedAtoms*2];
                double fphiZ = phiVec[i + paddedAtoms*3];

                recipEnergy += uf0 * fphiX + uf1 * fphiY + uf2 * fphiZ;
            }
        } else {
            vector<float> inducedDipoleVec;
            vector<float> phiVec;
            inducedDipole.download(inducedDipoleVec);
            pmePhi.download(phiVec);

            for (int i = 0; i < numParticles; i++) {
                double ux = inducedDipoleVec[3*i];
                double uy = inducedDipoleVec[3*i+1];
                double uz = inducedDipoleVec[3*i+2];

                double uf0 = cartToFrac[0][0]*ux + cartToFrac[0][1]*uy + cartToFrac[0][2]*uz;
                double uf1 = cartToFrac[1][0]*ux + cartToFrac[1][1]*uy + cartToFrac[1][2]*uz;
                double uf2 = cartToFrac[2][0]*ux + cartToFrac[2][1]*uy + cartToFrac[2][2]*uz;

                double fphiX = phiVec[i + paddedAtoms];
                double fphiY = phiVec[i + paddedAtoms*2];
                double fphiZ = phiVec[i + paddedAtoms*3];

                recipEnergy += uf0 * fphiX + uf1 * fphiY + uf2 * fphiZ;
            }
        }
        recipEnergy *= 0.5 * ONE_4PI_EPS0;
    }

    return recipEnergy;
}

double CommonCalcTholeDipoleForceKernel::computePmeInducedRecipEnergy() {
    // Induced reciprocal energy: 0.5 * ELECTRIC * sum(inducedDipole_frac · grad_phi_fixed_frac)
    // Uses the already-computed fixed potential (pmePhi), no grid operations needed
    int paddedAtoms = cc.getPaddedNumAtoms();

    // Build transformation matrix cartToFrac[i][j] = gridSize[j] * recipBoxVec[i][j]
    double cartToFrac[3][3];
    if (cc.getUseDoublePrecision()) {
        cartToFrac[0][0] = gridSizeX * recipBoxVecXD.x;
        cartToFrac[0][1] = gridSizeY * recipBoxVecXD.y;
        cartToFrac[0][2] = gridSizeZ * recipBoxVecXD.z;
        cartToFrac[1][0] = gridSizeX * recipBoxVecYD.x;
        cartToFrac[1][1] = gridSizeY * recipBoxVecYD.y;
        cartToFrac[1][2] = gridSizeZ * recipBoxVecYD.z;
        cartToFrac[2][0] = gridSizeX * recipBoxVecZD.x;
        cartToFrac[2][1] = gridSizeY * recipBoxVecZD.y;
        cartToFrac[2][2] = gridSizeZ * recipBoxVecZD.z;
    } else {
        cartToFrac[0][0] = gridSizeX * recipBoxVecXF.x;
        cartToFrac[0][1] = gridSizeY * recipBoxVecXF.y;
        cartToFrac[0][2] = gridSizeZ * recipBoxVecXF.z;
        cartToFrac[1][0] = gridSizeX * recipBoxVecYF.x;
        cartToFrac[1][1] = gridSizeY * recipBoxVecYF.y;
        cartToFrac[1][2] = gridSizeZ * recipBoxVecYF.z;
        cartToFrac[2][0] = gridSizeX * recipBoxVecZF.x;
        cartToFrac[2][1] = gridSizeY * recipBoxVecZF.y;
        cartToFrac[2][2] = gridSizeZ * recipBoxVecZF.z;
    }

    double energy = 0.0;
    if (cc.getUseDoublePrecision()) {
        vector<double> inducedDipoleVec;
        vector<double> phiVec;
        inducedDipole.download(inducedDipoleVec);
        pmePhi.download(phiVec);

        for (int i = 0; i < numParticles; i++) {
            double ux = inducedDipoleVec[3*i];
            double uy = inducedDipoleVec[3*i+1];
            double uz = inducedDipoleVec[3*i+2];

            // Transform to fractional
            double uf0 = cartToFrac[0][0]*ux + cartToFrac[0][1]*uy + cartToFrac[0][2]*uz;
            double uf1 = cartToFrac[1][0]*ux + cartToFrac[1][1]*uy + cartToFrac[1][2]*uz;
            double uf2 = cartToFrac[2][0]*ux + cartToFrac[2][1]*uy + cartToFrac[2][2]*uz;

            // Fractional phi gradients
            double fphiX = phiVec[i + paddedAtoms];
            double fphiY = phiVec[i + paddedAtoms*2];
            double fphiZ = phiVec[i + paddedAtoms*3];

            energy += uf0 * fphiX + uf1 * fphiY + uf2 * fphiZ;
        }
    } else {
        vector<float> inducedDipoleVec;
        vector<float> phiVec;
        inducedDipole.download(inducedDipoleVec);
        pmePhi.download(phiVec);

        for (int i = 0; i < numParticles; i++) {
            double ux = inducedDipoleVec[3*i];
            double uy = inducedDipoleVec[3*i+1];
            double uz = inducedDipoleVec[3*i+2];

            double uf0 = cartToFrac[0][0]*ux + cartToFrac[0][1]*uy + cartToFrac[0][2]*uz;
            double uf1 = cartToFrac[1][0]*ux + cartToFrac[1][1]*uy + cartToFrac[1][2]*uz;
            double uf2 = cartToFrac[2][0]*ux + cartToFrac[2][1]*uy + cartToFrac[2][2]*uz;

            double fphiX = phiVec[i + paddedAtoms];
            double fphiY = phiVec[i + paddedAtoms*2];
            double fphiZ = phiVec[i + paddedAtoms*3];

            energy += uf0 * fphiX + uf1 * fphiY + uf2 * fphiZ;
        }
    }
    return 0.5 * ONE_4PI_EPS0 * energy;
}

double CommonCalcTholeDipoleForceKernel::computePmeSelfEnergy() {
    // Self-energy correction for PME - matches reference implementation
    // E_self = -alpha*electric/(sqrt(pi)) * [sum_i(q_i^2) + (2/3)*alpha^2*sum_i(mu_i^2) + (2/3)*alpha^2*sum_i(mu_i·mu_ind_i)]
    //        - pi*electric/(2*volume*alpha^2) * (totalCharge)^2  (plasma term)

    double cii = 0.0;           // sum of q^2
    double dii_perm = 0.0;      // sum of |mu_perm|^2
    double dii_cross = 0.0;     // sum of mu_perm · mu_ind
    double totalCharge = 0.0;

    ArrayInterface& posq = cc.getPosq();
    if (cc.getUseDoublePrecision()) {
        vector<mm_double4> posqVec;
        vector<double> labDipoleVec, inducedDipoleVec;
        posq.download(posqVec);
        labDipoles.download(labDipoleVec);
        inducedDipole.download(inducedDipoleVec);

        for (int i = 0; i < numParticles; i++) {
            double q = posqVec[i].w;
            totalCharge += q;
            cii += q * q;

            double mx = labDipoleVec[3*i];
            double my = labDipoleVec[3*i+1];
            double mz = labDipoleVec[3*i+2];
            dii_perm += mx*mx + my*my + mz*mz;

            double ux = inducedDipoleVec[3*i];
            double uy = inducedDipoleVec[3*i+1];
            double uz = inducedDipoleVec[3*i+2];
            dii_cross += mx*ux + my*uy + mz*uz;
        }
    } else {
        vector<mm_float4> posqVec;
        vector<float> labDipoleVec, inducedDipoleVec;
        posq.download(posqVec);
        labDipoles.download(labDipoleVec);
        inducedDipole.download(inducedDipoleVec);

        for (int i = 0; i < numParticles; i++) {
            double q = posqVec[i].w;
            totalCharge += q;
            cii += q * q;

            double mx = labDipoleVec[3*i];
            double my = labDipoleVec[3*i+1];
            double mz = labDipoleVec[3*i+2];
            dii_perm += mx*mx + my*my + mz*mz;

            double ux = inducedDipoleVec[3*i];
            double uy = inducedDipoleVec[3*i+1];
            double uz = inducedDipoleVec[3*i+2];
            dii_cross += mx*ux + my*uy + mz*uz;
        }
    }

    // Compute prefactor: -alpha*electric/sqrt(pi)
    double prefac = -pmeAlpha * ONE_4PI_EPS0 / sqrt(M_PI);
    double a2 = pmeAlpha * pmeAlpha;

    double chargeTerm = prefac * cii;
    double permDipoleTerm = prefac * (2.0/3.0) * a2 * dii_perm;
    double crossTerm = prefac * (2.0/3.0) * a2 * dii_cross;

    // Plasma term (for charged systems): -pi*electric/(2*volume*alpha^2) * totalCharge^2
    Vec3 box[3];
    cc.getPeriodicBoxVectors(box[0], box[1], box[2]);
    double volume = box[0][0] * box[1][1] * box[2][2];  // Assumes orthorhombic box
    double plasmaTerm = totalCharge * totalCharge * M_PI * ONE_4PI_EPS0 / (2.0 * volume * a2);

    return chargeTerm + permDipoleTerm + crossTerm - plasmaTerm;
}

bool CommonCalcTholeDipoleForceKernel::iterateDipolesByDIIS(int iteration) {
    // Record the current dipoles and compute the error
    // Use cc.getNumThreadBlocks() groups with diisBlockSize threads per block for proper reduction
    int numThreadBlocks = cc.getNumThreadBlocks();
    recordDIISDipolesKernel->setArg(8, iteration);
    recordDIISDipolesKernel->execute(numThreadBlocks * diisBlockSize, diisBlockSize);

    // Build the DIIS matrix
    buildMatrixKernel->setArg(1, iteration);
    buildMatrixKernel->execute(1);

    // Solve for coefficients
    int numPrev = min(iteration+1, MaxPrevDIISDipoles);
    solveMatrixKernel->setArg(0, numPrev);
    solveMatrixKernel->execute(32, 32);

    // Check convergence - sum errors from all thread blocks
    vector<mm_float2> errors;
    inducedDipoleErrors.download(errors);
    double total = 0;
    for (int i = 0; i < numThreadBlocks; i++)
        total += errors[i].x + errors[i].y;
    double rms = sqrt(total/(3*numParticles));

    // If converged, don't update dipoles (matches Reference PCG behavior)
    if (rms < inducedEpsilon)
        return true;

    // Update dipoles using DIIS for next iteration
    updateInducedFieldKernel->setArg(3, numPrev);
    updateInducedFieldKernel->execute(numParticles);

    return false;
}

void CommonCalcTholeDipoleForceKernel::ensureMultipolesValid(ContextImpl& context) {
    if (multipolesAreValid) {
        int numAtoms = cc.getNumAtoms();
        if (cc.getUseDoublePrecision()) {
            vector<mm_double4> pos1, pos2;
            cc.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numAtoms; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
        else {
            vector<mm_float4> pos1, pos2;
            cc.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numAtoms; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
    }
    if (!multipolesAreValid)
        context.calcForcesAndEnergy(false, false, context.getIntegrator().getIntegrationForceGroups());
}

void CommonCalcTholeDipoleForceKernel::copyParametersToContext(ContextImpl& context, const TholeDipoleForce& force) {
    ContextSelector selector(cc);
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    vector<mm_float2> dampingAndTholeVec(cc.getPaddedNumAtoms());
    vector<float> polarizabilityVec(cc.getPaddedNumAtoms());
    vector<float> localDipolesVec(3*cc.getPaddedNumAtoms());

    for (int i = 0; i < numParticles; i++) {
        double charge, polarity;
        int axisType, atomZ, atomX, atomY;
        vector<double> dipole;
        force.getParticleParameters(i, charge, dipole, polarity, axisType, atomZ, atomX, atomY);
        double damp = (polarity > 0 ? pow(polarity, 1.0/6.0) : 0.0);
        dampingAndTholeVec[i] = mm_float2((float) damp, (float) force.getTholeDampingParameter());
        polarizabilityVec[i] = (float) polarity;
        for (int j = 0; j < 3; j++)
            localDipolesVec[3*i+j] = (float) dipole[j];
    }

    dampingAndThole.upload(dampingAndTholeVec);
    polarizability.upload(polarizabilityVec);
    localDipoles.upload(localDipolesVec);
    cc.invalidateMolecules();
    multipolesAreValid = false;
}

void CommonCalcTholeDipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePME)
        throw OpenMMException("getPMEParametersInContext: This Force does not use PME");
    alpha = pmeAlpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

void CommonCalcTholeDipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numAtoms = cc.getNumAtoms();
    dipoles.resize(numAtoms);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<double> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numAtoms; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
    else {
        vector<float> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numAtoms; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
}

void CommonCalcTholeDipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numAtoms = cc.getNumAtoms();
    dipoles.resize(numAtoms);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<double> d;
        inducedDipole.download(d);
        for (int i = 0; i < numAtoms; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
    else {
        vector<float> d;
        inducedDipole.download(d);
        for (int i = 0; i < numAtoms; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
}

void CommonCalcTholeDipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numAtoms = cc.getNumAtoms();
    dipoles.resize(numAtoms);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<double> labDipoleVec, inducedDipoleVec;
        labDipoles.download(labDipoleVec);
        inducedDipole.download(inducedDipoleVec);
        for (int i = 0; i < numAtoms; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i] + inducedDipoleVec[3*i],
                                     labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1],
                                     labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2]);
    }
    else {
        vector<float> labDipoleVec, inducedDipoleVec;
        labDipoles.download(labDipoleVec);
        inducedDipole.download(inducedDipoleVec);
        for (int i = 0; i < numAtoms; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i] + inducedDipoleVec[3*i],
                                     labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1],
                                     labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2]);
    }
}

void CommonCalcTholeDipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const vector<Vec3>& inputGrid, vector<double>& outputElectrostaticPotential) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    int numPoints = inputGrid.size();
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    ComputeArray points, potential;
    points.initialize(cc, numPoints, 4*elementSize, "points");
    potential.initialize(cc, numPoints, elementSize, "potential");

    if (cc.getUseDoublePrecision()) {
        vector<mm_double4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = mm_double4(inputGrid[i][0], inputGrid[i][1], inputGrid[i][2], 0);
        points.upload(p);
    }
    else {
        vector<mm_float4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = mm_float4((float) inputGrid[i][0], (float) inputGrid[i][1], (float) inputGrid[i][2], 0);
        points.upload(p);
    }

    computePotentialKernel->setArg(3, points);
    computePotentialKernel->setArg(4, potential);
    computePotentialKernel->setArg(5, numPoints);
    setPeriodicBoxArgs(context, computePotentialKernel, 6);
    computePotentialKernel->execute(numPoints, 128);

    outputElectrostaticPotential.resize(numPoints);
    if (cc.getUseDoublePrecision())
        potential.download(outputElectrostaticPotential);
    else {
        vector<float> p(numPoints);
        potential.download(p);
        for (int i = 0; i < numPoints; i++)
            outputElectrostaticPotential[i] = p[i];
    }
}

template <class T, class T4>
void CommonCalcTholeDipoleForceKernel::computeSystemMultipoleMomentsImpl(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    int numAtoms = cc.getNumAtoms();
    vector<T4> posqVec;
    vector<T> labDipoleVec;
    vector<T> inducedDipoleVec;
    cc.getPosq().download(posqVec);
    labDipoles.download(labDipoleVec);
    inducedDipole.download(inducedDipoleVec);

    // Compute center of mass
    double totalMass = 0.0;
    Vec3 centerOfMass(0, 0, 0);
    for (int i = 0; i < numAtoms; i++) {
        double mass = system.getParticleMass(i);
        totalMass += mass;
        centerOfMass[0] += mass*posqVec[i].x;
        centerOfMass[1] += mass*posqVec[i].y;
        centerOfMass[2] += mass*posqVec[i].z;
    }
    if (totalMass > 0.0) {
        centerOfMass[0] /= totalMass;
        centerOfMass[1] /= totalMass;
        centerOfMass[2] /= totalMass;
    }

    // Compute total charge and dipole
    double totalCharge = 0.0;
    Vec3 totalDipole(0, 0, 0);
    const vector<int>& order = cc.getAtomIndex();
    for (int i = 0; i < numAtoms; i++) {
        double q = posqVec[i].w;
        totalCharge += q;
        double dx = posqVec[i].x - centerOfMass[0];
        double dy = posqVec[i].y - centerOfMass[1];
        double dz = posqVec[i].z - centerOfMass[2];
        totalDipole[0] += q*dx + labDipoleVec[3*i] + inducedDipoleVec[3*i];
        totalDipole[1] += q*dy + labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1];
        totalDipole[2] += q*dz + labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2];
    }

    // Convert dipole to Debye
    const double DEBYE = 48.033324;
    outputMultipoleMoments.resize(4);
    outputMultipoleMoments[0] = totalCharge;
    outputMultipoleMoments[1] = totalDipole[0] * DEBYE;
    outputMultipoleMoments[2] = totalDipole[1] * DEBYE;
    outputMultipoleMoments[3] = totalDipole[2] * DEBYE;
}

void CommonCalcTholeDipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    ContextSelector selector(cc);
    ensureMultipolesValid(context);
    if (cc.getUseDoublePrecision())
        computeSystemMultipoleMomentsImpl<double, mm_double4>(context, outputMultipoleMoments);
    else
        computeSystemMultipoleMomentsImpl<float, mm_float4>(context, outputMultipoleMoments);
}
