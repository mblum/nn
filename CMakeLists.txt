CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
IF(COMMAND CMAKE_POLICY)
	CMAKE_POLICY(SET CMP0003 OLD)
ENDIF(COMMAND CMAKE_POLICY)

PROJECT( nn )

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
SET(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE INTERNAL "Prefix prepended to install directories" FORCE)

FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

ADD_LIBRARY(nn nn.cpp)

# ----- Testing -----
INCLUDE(ExternalProject)

# ----- Download and build gtest -----
ExternalProject_Add(googletest
  SVN_REPOSITORY "http://googletest.googlecode.com/svn/tags/release-1.6.0"
  INSTALL_COMMAND ""
)

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest/include)
LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build)

ADD_EXECUTABLE(nn_test nn_test.cpp)
TARGET_LINK_LIBRARIES(nn_test nn gtest gtest_main pthread)

ADD_EXECUTABLE(nn_example nn_example.cpp)
TARGET_LINK_LIBRARIES(nn_example nn)

ADD_EXECUTABLE(mnist mnist.cpp)
TARGET_LINK_LIBRARIES(mnist nn)
