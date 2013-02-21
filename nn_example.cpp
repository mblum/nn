//
//  nn_example.cpp
//
//  Created by Manuel Blum on 21.02.13.
//  Copyright (c) 2012 Uni Freiburg. All rights reserved.
//

#include <Eigen/Dense>
#include <iostream>

#include "nn.h"

int main (int argc, const char* argv[]) {

  // input dimensionality
  int n_input = 2;
  // output dimensionality
  int n_output = 1;
  // number of training samples
  int m = 4;
  // number of layers
  int k = 3;
  // number of optimization steps
  int max_steps = 50;
  // regularization parameter
  int lambda = 0.01;

  // training inputs
  matrix_t X(m, n_input);
  matrix_t Y(m, n_output);

  // XOR problem
  X << 0, 0, 0, 1, 1, 0, 1, 1;
  Y << 0, 1, 1, 0;
  std::cout << "training input: " << std::endl << X << std::endl;
  std::cout << "training output: " << std::endl << Y << std::endl;

  // specify network topology
  Eigen::VectorXi topo(k);
  topo << n_input, 5, n_output;
  std::cout << "topology: " << std::endl << topo << std::endl;

  // initialize a neural network with given topology
  NeuralNet nn(topo);
  
  // train the network
  std::cout << "starting training" << std::endl;
  double err;
  for (int i = 0; i < max_steps; ++i) {
    err = nn.loss(X, Y, lambda);
    nn.rprop();
    printf("%3i   %4.4f\n", i, err*err);
  }

  // write model to disk
  nn.write("example.net");

  // read model from disk
  NeuralNet nn2("example.net");

  // testing 
  nn2.forward_pass(X);
  matrix_t Y_test = nn2.layer.back().a;

  std::cout << "test input:" << std::endl << X << std::endl;
  std::cout << "test output:" << std::endl << Y_test << std::endl;

  return 0;
}

