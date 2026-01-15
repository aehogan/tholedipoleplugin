%module tholedipoleplugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 */
%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
  %template(vectorveci) vector<vector<int> >;
};

%{
#include "TholeDipoleForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

/*
 * Unit handling for Python wrapper.
 * Getters return Quantity objects, setters accept Quantity and strip units.
 */

/* getCutoffDistance returns nm */
%pythonappend TholeDipolePlugin::TholeDipoleForce::getCutoffDistance() const %{
    val = unit.Quantity(val, unit.nanometer)
%}

/* setCutoffDistance accepts nm */
%pythonprepend TholeDipolePlugin::TholeDipoleForce::setCutoffDistance(double distance) %{
    if unit.is_quantity(distance):
        distance = distance.value_in_unit(unit.nanometer)
%}

/* addParticle: use shadow to handle units since SWIG uses *args for default parameters */
%feature("shadow") TholeDipolePlugin::TholeDipoleForce::addParticle %{
def addParticle(self, charge, molecularDipole, polarizability, axisType=5, multipoleAtomZ=-1, multipoleAtomX=-1, multipoleAtomY=-1):
    """Add a particle to the force.

    Parameters
    ----------
    charge : float or Quantity
        The charge in elementary charge units
    molecularDipole : list
        The permanent dipole moment [x, y, z] in e*nm units
    polarizability : float or Quantity
        The polarizability in nm^3 units
    axisType : int
        The axis type for the local frame (default NoAxisType=5)
    multipoleAtomZ, multipoleAtomX, multipoleAtomY : int
        Indices of atoms defining the local frame (default -1 for none)
    """
    if unit.is_quantity(charge):
        charge = charge.value_in_unit(unit.elementary_charge)
    if molecularDipole is not None and len(molecularDipole) > 0 and unit.is_quantity(molecularDipole[0]):
        molecularDipole = [d.value_in_unit(unit.elementary_charge*unit.nanometer) for d in molecularDipole]
    if unit.is_quantity(polarizability):
        polarizability = polarizability.value_in_unit(unit.nanometer**3)
    return _tholedipoleplugin.TholeDipoleForce_addParticle(self, charge, molecularDipole, polarizability, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY)
%}

/* setParticleParameters: use shadow for consistent handling */
%feature("shadow") TholeDipolePlugin::TholeDipoleForce::setParticleParameters %{
def setParticleParameters(self, index, charge, molecularDipole, polarizability, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY):
    """Set the parameters for a particle.

    Parameters
    ----------
    index : int
        The index of the particle
    charge : float or Quantity
        The charge in elementary charge units
    molecularDipole : list
        The permanent dipole moment [x, y, z] in e*nm units
    polarizability : float or Quantity
        The polarizability in nm^3 units
    axisType : int
        The axis type for the local frame
    multipoleAtomZ, multipoleAtomX, multipoleAtomY : int
        Indices of atoms defining the local frame
    """
    if unit.is_quantity(charge):
        charge = charge.value_in_unit(unit.elementary_charge)
    if molecularDipole is not None and len(molecularDipole) > 0 and unit.is_quantity(molecularDipole[0]):
        molecularDipole = [d.value_in_unit(unit.elementary_charge*unit.nanometer) for d in molecularDipole]
    if unit.is_quantity(polarizability):
        polarizability = polarizability.value_in_unit(unit.nanometer**3)
    return _tholedipoleplugin.TholeDipoleForce_setParticleParameters(self, index, charge, molecularDipole, polarizability, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY)
%}

/* getParticleParameters returns (charge, dipole, polarizability, axisType, z, x, y) with units */
%pythonappend TholeDipolePlugin::TholeDipoleForce::getParticleParameters(int index, double& charge, std::vector<double>& molecularDipole, double& polarizability, int& axisType, int& multipoleAtomZ, int& multipoleAtomX, int& multipoleAtomY) const %{
    # val is a tuple: (charge, molecularDipole, polarizability, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY)
    charge = unit.Quantity(val[0], unit.elementary_charge)
    dipole = tuple(unit.Quantity(d, unit.elementary_charge*unit.nanometer) for d in val[1])
    polarizability = unit.Quantity(val[2], unit.nanometer**3)
    val = (charge, dipole, polarizability, val[3], val[4], val[5], val[6])
%}

/* getInducedDipoles returns list of Vec3 in e*nm */
%pythonappend TholeDipolePlugin::TholeDipoleForce::getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles) %{
    val = [mm.Vec3(v[0], v[1], v[2])*unit.elementary_charge*unit.nanometer for v in val]
%}

/* getLabFramePermanentDipoles returns list of Vec3 in e*nm */
%pythonappend TholeDipolePlugin::TholeDipoleForce::getLabFramePermanentDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles) %{
    val = [mm.Vec3(v[0], v[1], v[2])*unit.elementary_charge*unit.nanometer for v in val]
%}

/* getTotalDipoles returns list of Vec3 in e*nm */
%pythonappend TholeDipolePlugin::TholeDipoleForce::getTotalDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles) %{
    val = [mm.Vec3(v[0], v[1], v[2])*unit.elementary_charge*unit.nanometer for v in val]
%}

/* getElectrostaticPotential returns list in kJ/(mol*e) */
%pythonappend TholeDipolePlugin::TholeDipoleForce::getElectrostaticPotential(const std::vector<OpenMM::Vec3>& inputGrid, OpenMM::Context& context, std::vector<double>& outputElectrostaticPotential) %{
    val = [unit.Quantity(v, unit.kilojoule_per_mole/unit.elementary_charge) for v in val]
%}

/* getSystemMultipoleMoments returns list with units (total charge, dipole x/y/z, quadrupole components) */
%pythonappend TholeDipolePlugin::TholeDipoleForce::getSystemMultipoleMoments(OpenMM::Context& context, std::vector<double>& outputMultipoleMoments) %{
    # First element is charge (e), next 3 are dipole (Debye), rest are quadrupole (Debye*nm)
    if len(val) >= 4:
        result = [unit.Quantity(val[0], unit.elementary_charge)]
        result.extend([unit.Quantity(v, unit.debye) for v in val[1:4]])
        if len(val) > 4:
            result.extend([unit.Quantity(v, unit.debye*unit.nanometer) for v in val[4:]])
        val = result
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

/*
 * Typemaps for std::vector<OpenMM::Vec3>& as output parameter.
 * These mirror OpenMM's typemaps.i but with the OpenMM:: namespace prefix.
 */
%typemap(in, numinputs=0) std::vector<OpenMM::Vec3>& (std::vector<OpenMM::Vec3> temp) {
    $1 = &temp;
}

%typemap(argout, fragment="Vec3_to_PyVec3") std::vector<OpenMM::Vec3>& {
    int n = (*$1).size();
    PyObject * pyList = PyList_New(n);
    for (int i=0; i<n; i++) {
        OpenMM::Vec3& v = (*$1).at(i);
        PyObject* pyVec = Vec3_to_PyVec3(v);
        PyList_SET_ITEM(pyList, i, pyVec);
    }
    $result = pyList;
}


namespace TholeDipolePlugin {

class TholeDipoleForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        PME = 1
    };

    enum PolarizationType {
        Direct = 0,
        Mutual = 1,
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

    enum TholeDampingType {
        NoDamping = 0,
        Exponential = 1,
        Amoeba = 2,
        Linear = 3
    };

    TholeDipoleForce();

    int getNumParticles() const;

    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);

    PolarizationType getPolarizationType() const;
    void setPolarizationType(PolarizationType type);

    TholeDampingType getTholeDampingType() const;
    void setTholeDampingType(TholeDampingType type);

    double getTholeDampingParameter() const;
    void setTholeDampingParameter(double parameter);

    bool getDampPermanentInducedField() const;
    void setDampPermanentInducedField(bool damp);

    double getCutoffDistance() const;
    void setCutoffDistance(double distance);

    int getPmeBSplineOrder() const;

    int getMutualInducedMaxIterations() const;
    void setMutualInducedMaxIterations(int inputMutualInducedMaxIterations);

    double getMutualInducedTargetEpsilon() const;
    void setMutualInducedTargetEpsilon(double inputMutualInducedTargetEpsilon);

    void setExtrapolationCoefficients(const std::vector<double>& coefficients);
    const std::vector<double>& getExtrapolationCoefficients() const;

    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);

    bool usesPeriodicBoundaryConditions() const;

    int addParticle(double charge, const std::vector<double>& molecularDipole, double polarizability,
                    int axisType = NoAxisType, int multipoleAtomZ = -1,
                    int multipoleAtomX = -1, int multipoleAtomY = -1);

    /*
     * Output parameters for getParticleParameters
    */
    %apply double& OUTPUT {double& charge};
    %apply double& OUTPUT {double& polarizability};
    %apply int& OUTPUT {int& axisType};
    %apply int& OUTPUT {int& multipoleAtomZ};
    %apply int& OUTPUT {int& multipoleAtomX};
    %apply int& OUTPUT {int& multipoleAtomY};
    %apply std::vector<double>& OUTPUT {std::vector<double>& molecularDipole};
    void getParticleParameters(int index, double& charge, std::vector<double>& molecularDipole,
                               double& polarizability, int& axisType,
                               int& multipoleAtomZ, int& multipoleAtomX, int& multipoleAtomY) const;
    %clear double& charge;
    %clear double& polarizability;
    %clear int& axisType;
    %clear int& multipoleAtomZ;
    %clear int& multipoleAtomX;
    %clear int& multipoleAtomY;
    %clear std::vector<double>& molecularDipole;

    void setParticleParameters(int index, double charge, const std::vector<double>& molecularDipole,
                               double polarizability, int axisType,
                               int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY);

    void setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms);

    %apply std::vector<int>& OUTPUT {std::vector<int>& covalentAtoms};
    void getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const;
    %clear std::vector<int>& covalentAtoms;

    %apply std::vector<std::vector<int> >& OUTPUT {std::vector<std::vector<int> >& covalentLists};
    void getCovalentMaps(int index, std::vector<std::vector<int> >& covalentLists) const;
    %clear std::vector<std::vector<int> >& covalentLists;

    /*
     * PME parameters
    */
    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    void setPMEParameters(double alpha, int nx, int ny, int nz);

    /*
     * Dipole getters - std::vector<Vec3>& is automatically an output parameter
     * due to typemaps in OpenMM's typemaps.i (numinputs=0, argout)
    */
    void getLabFramePermanentDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    void getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    void getTotalDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);

    %apply std::vector<double>& OUTPUT {std::vector<double>& outputElectrostaticPotential};
    void getElectrostaticPotential(const std::vector<OpenMM::Vec3>& inputGrid,
                                    OpenMM::Context& context, std::vector<double>& outputElectrostaticPotential);
    %clear std::vector<double>& outputElectrostaticPotential;

    %apply std::vector<double>& OUTPUT {std::vector<double>& outputMultipoleMoments};
    void getSystemMultipoleMoments(OpenMM::Context& context, std::vector<double>& outputMultipoleMoments);
    %clear std::vector<double>& outputMultipoleMoments;

    void updateParametersInContext(OpenMM::Context& context);

    /*
     * Add methods for casting a Force to a TholeDipoleForce.
    */
    %extend {
        static TholeDipolePlugin::TholeDipoleForce& cast(OpenMM::Force& force) {
            return dynamic_cast<TholeDipolePlugin::TholeDipoleForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<TholeDipolePlugin::TholeDipoleForce*>(&force) != NULL);
        }
    }
};

}
