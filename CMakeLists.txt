CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
IF(COMMAND CMAKE_POLICY)
    CMAKE_POLICY(SET CMP0003 OLD)
ENDIF(COMMAND CMAKE_POLICY)

PROJECT( nn )

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
SET(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE INTERNAL "Prefix prepended to install directories" FORCE)

# color definitions
IF(NOT WIN32)
    STRING(ASCII 27 Esc)
    SET(ColourReset "${Esc}[m")
    SET(BoldRed     "${Esc}[1;31m")
    SET(BoldGreen   "${Esc}[1;32m")
    SET(BoldMagenta "${Esc}[1;35m")
ENDIF()

FIND_PACKAGE(OpenMP QUIET)

FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

ADD_LIBRARY(nn nn.cpp)

ADD_EXECUTABLE(tutorial tutorial.cpp)
TARGET_LINK_LIBRARIES(tutorial nn)

ADD_EXECUTABLE(mnist mnist.cpp)
TARGET_LINK_LIBRARIES(mnist nn)

MESSAGE("")

# ----- OpenMP -----
IF (OPENMP_FOUND)

    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    MESSAGE("${BoldGreen}nn will be compiled with OpenMP support. ${ColourReset}" )

ELSE()

    MESSAGE("${BoldMagenta}nn will run in single core mode since OpenMP was not found.${ColourReset}" )

ENDIF()

# ----- Testing -----
OPTION(WITH_GTEST "Download and compile unit tests using googletest. " OFF) 

IF (${WITH_GTEST})

    INCLUDE(ExternalProject)

    # ----- Download and build gtest -----
    ExternalProject_Add(googletest
        SVN_REPOSITORY "http://googletest.googlecode.com/svn/tags/release-1.7.0"
        UPDATE_COMMAND ""
        INSTALL_COMMAND ""
        )

    INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest/include)
    LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build)

    ADD_EXECUTABLE(nntest nntest.cpp)
    TARGET_LINK_LIBRARIES(nntest nn gtest gtest_main pthread)

    MESSAGE("${BoldGreen}Compiling tests using googletest.${ColourReset}" )

ELSE()

    MESSAGE("${BoldMagenta}Tests will not be compiled. Run cmake with -DWITH_GTEST=ON to compile unit tests.${ColourReset}" )

ENDIF()

MESSAGE("")
