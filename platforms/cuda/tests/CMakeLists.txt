#
# Testing
#

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})

# Create individual tests for each test file
SET(INDIVIDUAL_TESTS
    TholeDipoleAmmonia
    TholeDipoleWaterPME
    Triclinic
    ZOnly
    AxisTypes
    EdgeCases
    TwoPointCharges
    TwoDipoles
    ParticleAccessors
    1DChainSizeConsistency
    CustomPointCharges
    PMECutoffConvergence
)

# Debug executable
ADD_EXECUTABLE(debug_induced_iter0 debug_induced_iter0.cpp CudaTestCommon.cpp)
TARGET_LINK_LIBRARIES(debug_induced_iter0 ${SHARED_THOLEDIPOLE_TARGET} ${SHARED_TARGET} TholeDipolePluginReference)

# Compare induced dipoles utility
ADD_EXECUTABLE(CompareInducedDipoles CompareInducedDipoles.cpp CudaTestCommon.cpp)
TARGET_LINK_LIBRARIES(CompareInducedDipoles ${SHARED_THOLEDIPOLE_TARGET} ${SHARED_TARGET} TholeDipolePluginReference)

FOREACH(TEST_NAME ${INDIVIDUAL_TESTS})
    ADD_EXECUTABLE(TestCuda${TEST_NAME} TestCuda${TEST_NAME}.cpp CudaTestCommon.cpp)
    TARGET_LINK_LIBRARIES(TestCuda${TEST_NAME} ${SHARED_THOLEDIPOLE_TARGET} ${SHARED_TARGET} TholeDipolePluginReference)
    IF (APPLE)
        SET_TARGET_PROPERTIES(TestCuda${TEST_NAME} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
    ELSE (APPLE)
        SET_TARGET_PROPERTIES(TestCuda${TEST_NAME} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
    ENDIF (APPLE)
    ADD_TEST(TestCuda${TEST_NAME}Single ${EXECUTABLE_OUTPUT_PATH}/TestCuda${TEST_NAME} single)
    ADD_TEST(TestCuda${TEST_NAME}Mixed ${EXECUTABLE_OUTPUT_PATH}/TestCuda${TEST_NAME} mixed)
    ADD_TEST(TestCuda${TEST_NAME}Double ${EXECUTABLE_OUTPUT_PATH}/TestCuda${TEST_NAME} double)
ENDFOREACH(TEST_NAME ${INDIVIDUAL_TESTS})
