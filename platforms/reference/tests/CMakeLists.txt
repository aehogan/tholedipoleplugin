#
# Testing
#

# Create individual tests for each test function, each with its own source file
SET(INDIVIDUAL_TESTS
    TholeDipoleAmmoniaDirectPolarization
    TholeDipoleAmmoniaMutualPolarization
    TholeDipoleWaterPMEDirectPolarization
    TholeDipoleWaterPMEMutualPolarization
    ParticleInducedDipoles
    ParticleLabFramePermanentDipoles
    ParticleTotalDipoles
    Triclinic
    ZBisect
    ZOnly3Particle
    ZOnlyNoDipNoPol
    ZOnlyNoDip
    ZOnlyNoPol
    ZOnly
    ZOnlyMutual
    ZeroCharges
    ZeroDipoles
    ZeroPolarizabilities
    AllZeros
    SingleParticle
    SingleWater
    ZThenX
    TwoPointCharges
    1DChainSizeConsistency
    PMECutoffConvergence
    Bisector
    NoAxisType
    ThreeFold
)

FOREACH(TEST_NAME ${INDIVIDUAL_TESTS})
    ADD_EXECUTABLE(Test${TEST_NAME} Test${TEST_NAME}.cpp TholeDipoleTestCommon.cpp)
    TARGET_LINK_LIBRARIES(Test${TEST_NAME} ${SHARED_TARGET} ${OPENMM_LIBRARY} /home/aehogan2/miniforge3/envs/openff_jun2025/lib/libOpenMMAmoeba.so)
    SET_TARGET_PROPERTIES(Test${TEST_NAME} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
    ADD_TEST(${TEST_NAME} ${EXECUTABLE_OUTPUT_PATH}/Test${TEST_NAME})
ENDFOREACH(TEST_NAME ${INDIVIDUAL_TESTS})

