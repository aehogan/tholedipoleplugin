#ifndef OPENMM_THOLEDIPOLEFORCE_H_
#define OPENMM_THOLEDIPOLEFORCE_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "internal/windowsExportTholeDipole.h"
#include "openmm/Vec3.h"

#include <sstream>
#include <vector>

namespace TholeDipolePlugin {

/**
 * This class implements the Thole/Applequist polarizable dipole interaction.
 * It is a simplified version of the AMOEBA multipole force with only
 * charge and dipole moments (no quadrupoles), and a single induced dipole per atom.
 *
 * To use it, create a TholeDipoleForce object then call addParticle() once for each atom.  After
 * an entry has been added, you can modify its force field parameters by calling setParticleParameters().
 * This will have no effect on Contexts that already exist unless you call updateParametersInContext().
 */
class OPENMM_EXPORT_THOLEDIPOLE TholeDipoleForce : public OpenMM::Force {

public:

    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,

        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 1
    };

    enum PolarizationType {
        /**
         * Direct polarization approximation.  The induced dipoles depend only on the fixed multipoles, not on other
         * induced dipoles.
         */
        Direct = 0,

        /**
         * Full mutually induced polarization.  The dipoles are iterated until they converge to the accuracy specified
         * by getMutualInducedTargetEpsilon().
         */
        Mutual = 1,

        /**
         * Extrapolated perturbation theory approximation.  The dipoles are iterated a few times, and then an analytic
         * approximation is used to extrapolate to the fully converged values.  Call setExtrapolationCoefficients()
         * to set the coefficients used for the extrapolation.
         */
        Extrapolated = 2
    };

    enum MultipoleAxisTypes { 
        ZThenX = 0, 
        Bisector = 1, 
        ZBisect = 2, 
        ThreeFold = 3, 
        ZOnly = 4, 
        NoAxisType = 5, 
        LastAxisTypeIndex = 6 
    };

    enum CovalentType {
        Covalent12 = 0, 
        Covalent13 = 1, 
        Covalent14 = 2, 
        Covalent15 = 3,
        CovalentEnd = 4
    };

    /**
     * Create a TholeDipoleForce.
     */
    TholeDipoleForce();

    /**
     * Get the number of particles in the potential function
     */
    int getNumParticles() const {
        return particles.size();
    }

    /**
     * Get the method used for handling long-range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;

    /**
     * Set the method used for handling long-range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);

    /**
     * Get polarization type
     */
    PolarizationType getPolarizationType() const;

    /**
     * Set the polarization type
     */
    void setPolarizationType(PolarizationType type);

    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;

    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);

    /**
     * Get the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

    /**
     * Set the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);

    /**
     * Get the B-spline order to use for PME charge spreading
     *
     * @return the B-spline order
     */
    int getPmeBSplineOrder() const;

    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions
     * on the allowed grid sizes, the values that are actually used may be slightly different from those
     * specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;

    /**
     * Add a particle to the force.
     *
     * @param charge               the particle's charge
     * @param molecularDipole      the particle's molecular dipole (vector of size 3)
     * @param polarizability       the particle's isotropic polarizability
     * @param tholeDamping         the particle's Thole damping parameter
     * @param axisType             the particle's axis type
     * @param multipoleAtomZ       index of first atom used in constructing lab<->molecular frames (-1 for none)
     * @param multipoleAtomX       index of second atom used in constructing lab<->molecular frames (-1 for none)
     * @param multipoleAtomY       index of third atom used in constructing lab<->molecular frames (-1 for none)
     *
     * @return the index of the particle that was added
     */
    int addParticle(double charge, const std::vector<double>& molecularDipole, double polarizability, 
                    double tholeDamping, int axisType = NoAxisType, int multipoleAtomZ = -1, 
                    int multipoleAtomX = -1, int multipoleAtomY = -1);

    /**
     * Get the particle parameters.
     *
     * @param index                     the index of the particle for which to get parameters
     * @param[out] charge               the particle's charge
     * @param[out] molecularDipole      the particle's molecular dipole (vector of size 3)
     * @param[out] polarizability       the particle's isotropic polarizability
     * @param[out] tholeDamping         the particle's Thole damping parameter
     * @param[out] axisType             the particle's axis type
     * @param[out] multipoleAtomZ       index of first atom used in constructing lab<->molecular frames
     * @param[out] multipoleAtomX       index of second atom used in constructing lab<->molecular frames
     * @param[out] multipoleAtomY       index of third atom used in constructing lab<->molecular frames
     */
    void getParticleParameters(int index, double& charge, std::vector<double>& molecularDipole, 
                               double& polarizability, double& tholeDamping, int& axisType, 
                               int& multipoleAtomZ, int& multipoleAtomX, int& multipoleAtomY) const;

    /**
     * Set the particle parameters.
     *
     * @param index                the index of the particle for which to set parameters
     * @param charge               the particle's charge
     * @param molecularDipole      the particle's molecular dipole (vector of size 3)
     * @param polarizability       the particle's isotropic polarizability
     * @param tholeDamping         the particle's Thole damping parameter
     * @param axisType             the particle's axis type
     * @param multipoleAtomZ       index of first atom used in constructing lab<->molecular frames
     * @param multipoleAtomX       index of second atom used in constructing lab<->molecular frames
     * @param multipoleAtomY       index of third atom used in constructing lab<->molecular frames
     */
    void setParticleParameters(int index, double charge, const std::vector<double>& molecularDipole, 
                               double polarizability, double tholeDamping, int axisType, 
                               int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY);

    /**
     * Set the CovalentMap for an atom
     *
     * @param index                the index of the atom for which to set parameters
     * @param typeId               CovalentType type
     * @param covalentAtoms        vector of covalent atoms associated w/ the specified CovalentType
     */
    void setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms);

    /**
     * Get the CovalentMap for an atom
     *
     * @param index                the index of the atom for which to get parameters
     * @param typeId               CovalentType type
     * @param[out] covalentAtoms   output vector of covalent atoms associated w/ the specified CovalentType
     */
    void getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const;

    /**
     * Get all CovalentMaps for an atom
     *
     * @param index                the index of the atom for which to get parameters
     * @param[out] covalentLists   output vector of covalent lists of atoms
     */
    void getCovalentMaps(int index, std::vector<std::vector<int> >& covalentLists) const;

    /**
     * Get the max number of iterations to be used in calculating the mutual induced dipoles
     *
     * @return max number of iterations
     */
    int getMutualInducedMaxIterations(void) const;

    /**
     * Set the max number of iterations to be used in calculating the mutual induced dipoles
     *
     * @param inputMutualInducedMaxIterations   number of iterations
     */
    void setMutualInducedMaxIterations(int inputMutualInducedMaxIterations);

    /**
     * Get the target epsilon to be used to test for convergence of iterative method used in calculating the mutual induced dipoles
     *
     * @return target epsilon
     */
    double getMutualInducedTargetEpsilon(void) const;

    /**
     * Set the target epsilon to be used to test for convergence of iterative method used in calculating the mutual induced dipoles
     *
     * @param inputMutualInducedTargetEpsilon   target epsilon
     */
    void setMutualInducedTargetEpsilon(double inputMutualInducedTargetEpsilon);

    /**
     * Set the coefficients for the mu_0, mu_1, mu_2, ..., mu_n terms in the extrapolation
     * algorithm for induced dipoles.
     *
     * @param coefficients      a vector whose mth entry specifies the coefficient for mu_m.  The length of this
     *                          vector determines how many iterations are performed.
     */
    void setExtrapolationCoefficients(const std::vector<double> &coefficients);

    /**
     * Get the coefficients for the mu_0, mu_1, mu_2, ..., mu_n terms in the extrapolation
     * algorithm for induced dipoles.
     */
    const std::vector<double>& getExtrapolationCoefficients() const;

    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the grid dimensions and separation (alpha)
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * This can be overridden by explicitly setting an alpha parameter and grid dimensions to use.
     */
    double getEwaldErrorTolerance() const;

    /**
     * Set the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the grid dimensions and separation (alpha)
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * This can be overridden by explicitly setting an alpha parameter and grid dimensions to use.
     */
    void setEwaldErrorTolerance(double tol);

    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     *
     * @param context         the Context for which to get the fixed dipoles
     * @param[out] dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);

    /**
     * Get the induced dipole moments of all particles.
     *
     * @param context         the Context for which to get the induced dipoles
     * @param[out] dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);

    /**
     * Get the total dipole moments (fixed plus induced) of all particles.
     *
     * @param context         the Context for which to get the total dipoles
     * @param[out] dipoles    the total dipole moment of particle i is stored into the i'th element
     */
    void getTotalDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);

    /**
     * Get the electrostatic potential.
     *
     * @param inputGrid    input grid points over which the potential is to be evaluated
     * @param context      context
     * @param[out] outputElectrostaticPotential output potential
     */
    void getElectrostaticPotential(const std::vector<OpenMM::Vec3>& inputGrid,
                                    OpenMM::Context& context, std::vector<double>& outputElectrostaticPotential);

    /**
     * Get the system multipole moments.
     *
     * This method is most useful for non-periodic systems.  When called for a periodic system, only the
     * <i>lowest nonvanishing moment</i> has a well defined value.  This means that if the system has a net
     * nonzero charge, the dipole moment is not well defined and should be ignored.
     *
     * @param context      context
     * @param[out] outputMultipoleMoments (charge, dipole_x, dipole_y, dipole_z)
     */
    void getSystemMultipoleMoments(OpenMM::Context& context, std::vector<double>& outputMultipoleMoments);

    /**
     * Update the particle parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setParticleParameters() to modify this object's parameters, then call updateParametersInContext() to
     * copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of particles.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the Context.  Furthermore, this method cannot be used to add new particles,
     * only to change the parameters of existing ones.
     */
    void updateParametersInContext(OpenMM::Context& context);

    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if nonbondedMethod uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == TholeDipoleForce::PME;
    }

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    NonbondedMethod nonbondedMethod;
    PolarizationType polarizationType;
    double cutoffDistance;
    double alpha;
    int pmeBSplineOrder, nx, ny, nz;
    int mutualInducedMaxIterations;
    std::vector<double> extrapolationCoefficients;
    double mutualInducedTargetEpsilon;
    double ewaldErrorTol;
    
    class ParticleInfo;
    std::vector<ParticleInfo> particles;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class TholeDipoleForce::ParticleInfo {
public:
    int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
    double charge, polarizability, tholeDamping;
    std::vector<double> molecularDipole;
    std::vector<std::vector<int> > covalentInfo;

    ParticleInfo() {
        axisType = multipoleAtomZ = multipoleAtomX = multipoleAtomY = -1;
        charge = polarizability = tholeDamping = 0.0;
        molecularDipole.resize(3);
    }

    ParticleInfo(double charge, const std::vector<double>& inputMolecularDipole, double polarizability,
                 double tholeDamping, int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY) :
        axisType(axisType), multipoleAtomZ(multipoleAtomZ), multipoleAtomX(multipoleAtomX), multipoleAtomY(multipoleAtomY),
        charge(charge), polarizability(polarizability), tholeDamping(tholeDamping) {

        covalentInfo.resize(CovalentEnd);

        molecularDipole.resize(3);
        molecularDipole[0] = inputMolecularDipole[0];
        molecularDipole[1] = inputMolecularDipole[1];
        molecularDipole[2] = inputMolecularDipole[2];
    }
};

} // namespace TholeDipolePlugin

#endif /*OPENMM_THOLEDIPOLEFORCE_H_*/
