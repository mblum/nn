CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
IF(COMMAND CMAKE_POLICY)
	CMAKE_POLICY(SET CMP0003 OLD)
ENDIF(COMMAND CMAKE_POLICY)

PROJECT( nn )

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
SET(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE INTERNAL "Prefix prepended to install directories" FORCE)

FIND_PACKAGE(Eigen3 REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

ADD_LIBRARY(nn nn.cpp)

ADD_EXECUTABLE(tutorial tutorial.cpp)
TARGET_LINK_LIBRARIES(tutorial nn)

ADD_EXECUTABLE(mnist mnist.cpp)
TARGET_LINK_LIBRARIES(mnist nn)

OPTION(WITH_GTEST "Download and compile unit tests using googletest. " OFF) 

# ----- Testing -----
IF (${WITH_GTEST})

  INCLUDE(ExternalProject)

  # ----- Download and build gtest -----
  ExternalProject_Add(googletest
    SVN_REPOSITORY "http://googletest.googlecode.com/svn/tags/release-1.6.0"
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
  )

  INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest/include)
  LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/googletest-prefix/src/googletest-build)

  ADD_EXECUTABLE(nntest nntest.cpp)
  TARGET_LINK_LIBRARIES(nntest nn gtest gtest_main pthread)

ENDIF()
