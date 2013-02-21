Multi-layer perceptrons using RPROP
===================================

nn is a lightweight neural network library using resilient propagation for training the weights

Installation
------------

nn was tested on Ubuntu Linux and MacOS

* install [CMake](http://http://www.cmake.org/)
* download [Eigen3](http://eigen.tuxfamily.org/) and put it somewhere cmake can find it
* clone the nn repository or download it [here](https://bitbucket.org/mblum/nn/get/master.tar.gz)
* change to the nn directory and create a build folder 
  ~~~
  mkdir build
  ~~~
* run cmake from within the build folder and compile the library using make
  ~~~
  cd build
  cmake ..
  make
  ~~~
* if cmake was able to download [googletest](http://code.google.com/p/googletest/) you can run unit tests now
  ~~~
  ./nn_test
  ~~~

Usage of the library
--------------------


