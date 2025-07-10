/* -------------------------------------------------------------------------- *
 *                             OpenMMTholeDipole                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
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

#include "TholeDipoleForceProxy.h"
#include "TholeDipoleForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>

using namespace TholeDipolePlugin;
using namespace OpenMM;
using namespace std;

TholeDipoleForceProxy::TholeDipoleForceProxy() : SerializationProxy("TholeDipoleForce") {
}

void TholeDipoleForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const TholeDipoleForce& force = *reinterpret_cast<const TholeDipoleForce*>(object);
    
    // Serialize force parameters
    node.setIntProperty("nonbondedMethod", force.getNonbondedMethod());
    node.setIntProperty("polarizationType", force.getPolarizationType());
    node.setDoubleProperty("cutoffDistance", force.getCutoffDistance());
    node.setDoubleProperty("ewaldErrorTolerance", force.getEwaldErrorTolerance());
    node.setIntProperty("mutualInducedMaxIterations", force.getMutualInducedMaxIterations());
    node.setDoubleProperty("mutualInducedTargetEpsilon", force.getMutualInducedTargetEpsilon());
    node.setIntProperty("pmeBSplineOrder", force.getPmeBSplineOrder());
    
    // Serialize PME parameters
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("alpha", alpha);
    node.setIntProperty("nx", nx);
    node.setIntProperty("ny", ny);
    node.setIntProperty("nz", nz);
    
    // Serialize extrapolation coefficients
    const std::vector<double>& extrapolationCoefficients = force.getExtrapolationCoefficients();
    SerializationNode& extrapCoeffs = node.createChildNode("ExtrapolationCoefficients");
    for (int i = 0; i < (int) extrapolationCoefficients.size(); i++) {
        extrapCoeffs.createChildNode("Coefficient").setDoubleProperty("c", extrapolationCoefficients[i]);
    }
    
    // Serialize particles
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, polarizability, tholeDamping;
        std::vector<double> molecularDipole;
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        force.getParticleParameters(i, charge, molecularDipole, polarizability, tholeDamping, 
                                    axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY);
        
        SerializationNode& particle = particles.createChildNode("Particle");
        particle.setDoubleProperty("charge", charge);
        particle.setDoubleProperty("polarizability", polarizability);
        particle.setDoubleProperty("tholeDamping", tholeDamping);
        particle.setIntProperty("axisType", axisType);
        particle.setIntProperty("multipoleAtomZ", multipoleAtomZ);
        particle.setIntProperty("multipoleAtomX", multipoleAtomX);
        particle.setIntProperty("multipoleAtomY", multipoleAtomY);
        
        // Serialize dipole components
        SerializationNode& dipole = particle.createChildNode("Dipole");
        dipole.setDoubleProperty("x", molecularDipole[0]);
        dipole.setDoubleProperty("y", molecularDipole[1]);
        dipole.setDoubleProperty("z", molecularDipole[2]);
        
        // Serialize covalent maps
        SerializationNode& covalentMaps = particle.createChildNode("CovalentMaps");
        for (int j = 0; j < TholeDipoleForce::CovalentEnd; j++) {
            std::vector<int> covalentAtoms;
            force.getCovalentMap(i, static_cast<TholeDipoleForce::CovalentType>(j), covalentAtoms);
            if (!covalentAtoms.empty()) {
                SerializationNode& covalentMap = covalentMaps.createChildNode("CovalentMap");
                covalentMap.setIntProperty("type", j);
                for (int k = 0; k < (int) covalentAtoms.size(); k++) {
                    covalentMap.createChildNode("Atom").setIntProperty("index", covalentAtoms[k]);
                }
            }
        }
    }
}

void* TholeDipoleForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    TholeDipoleForce* force = new TholeDipoleForce();
    try {
        // Deserialize force parameters
        force->setNonbondedMethod(static_cast<TholeDipoleForce::NonbondedMethod>(node.getIntProperty("nonbondedMethod")));
        force->setPolarizationType(static_cast<TholeDipoleForce::PolarizationType>(node.getIntProperty("polarizationType")));
        force->setCutoffDistance(node.getDoubleProperty("cutoffDistance"));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldErrorTolerance"));
        force->setMutualInducedMaxIterations(node.getIntProperty("mutualInducedMaxIterations"));
        force->setMutualInducedTargetEpsilon(node.getDoubleProperty("mutualInducedTargetEpsilon"));
        
        // Deserialize PME parameters
        force->setPMEParameters(node.getDoubleProperty("alpha"), node.getIntProperty("nx"), 
                               node.getIntProperty("ny"), node.getIntProperty("nz"));
        
        // Deserialize extrapolation coefficients
        const SerializationNode& extrapCoeffs = node.getChildNode("ExtrapolationCoefficients");
        std::vector<double> extrapolationCoefficients;
        for (int i = 0; i < (int) extrapCoeffs.getChildren().size(); i++) {
            const SerializationNode& coeff = extrapCoeffs.getChildren()[i];
            extrapolationCoefficients.push_back(coeff.getDoubleProperty("c"));
        }
        force->setExtrapolationCoefficients(extrapolationCoefficients);
        
        // Deserialize particles
        const SerializationNode& particles = node.getChildNode("Particles");
        for (int i = 0; i < (int) particles.getChildren().size(); i++) {
            const SerializationNode& particle = particles.getChildren()[i];
            
            double charge = particle.getDoubleProperty("charge");
            double polarizability = particle.getDoubleProperty("polarizability");
            double tholeDamping = particle.getDoubleProperty("tholeDamping");
            int axisType = particle.getIntProperty("axisType");
            int multipoleAtomZ = particle.getIntProperty("multipoleAtomZ");
            int multipoleAtomX = particle.getIntProperty("multipoleAtomX");
            int multipoleAtomY = particle.getIntProperty("multipoleAtomY");
            
            // Deserialize dipole components
            const SerializationNode& dipole = particle.getChildNode("Dipole");
            std::vector<double> molecularDipole(3);
            molecularDipole[0] = dipole.getDoubleProperty("x");
            molecularDipole[1] = dipole.getDoubleProperty("y");
            molecularDipole[2] = dipole.getDoubleProperty("z");
            
            int particleIndex = force->addParticle(charge, molecularDipole, polarizability, tholeDamping, 
                                                  axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY);
            
            // Deserialize covalent maps
            const SerializationNode& covalentMaps = particle.getChildNode("CovalentMaps");
            for (int j = 0; j < (int) covalentMaps.getChildren().size(); j++) {
                const SerializationNode& covalentMap = covalentMaps.getChildren()[j];
                int type = covalentMap.getIntProperty("type");
                std::vector<int> covalentAtoms;
                for (int k = 0; k < (int) covalentMap.getChildren().size(); k++) {
                    const SerializationNode& atom = covalentMap.getChildren()[k];
                    covalentAtoms.push_back(atom.getIntProperty("index"));
                }
                force->setCovalentMap(particleIndex, static_cast<TholeDipoleForce::CovalentType>(type), covalentAtoms);
            }
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
