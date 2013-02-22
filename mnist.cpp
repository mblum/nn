// Copyright (c) 2013, Manuel Blum
// All rights reserved.

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <cstdio>

#include "nn.h"

inline void swap(int &val)
{
	val = (val<<24) | ((val<<8) & 0x00ff0000) | ((val>>8) & 0x0000ff00) | (val>>24);
}

matrix_t read_mnist_images(const char* filename)
{
  matrix_t X;
  std::ifstream fs(filename, std::ios::binary);
  if(fs) {
    int magic_number, num_images, num_rows, num_columns;
    fs.read((char*)&magic_number, sizeof(magic_number));
    fs.read((char*)&num_images, sizeof(num_images));
    fs.read((char*)&num_rows, sizeof(num_rows));
    fs.read((char*)&num_columns, sizeof(num_columns));
    if (magic_number != 2051) {
      swap(magic_number);
      swap(num_images);
      swap(num_rows);
      swap(num_columns);
    }

    X = matrix_t::Zero(num_images, num_rows*num_columns);

    for (size_t i=0; i<num_images; ++i) {
      for (size_t j=0; j<num_rows*num_columns; ++j) {
        unsigned char temp=0;
        fs.read((char*)&temp,sizeof(temp));
        X(i,j) = (double) temp;        
      }
    }
    fs.close();
  }
  return X;
}

matrix_t read_mnist_labels(const char* filename)
{
  matrix_t Y;
  std::ifstream fs(filename, std::ios::binary);
  if(fs) {
    int magic_number, num_images, num_rows, num_columns;
    fs.read((char*)&magic_number, sizeof(magic_number));
    fs.read((char*)&num_images, sizeof(num_images));
    if (magic_number != 2049) {
      swap(magic_number);
      swap(num_images);
    }

    Y = matrix_t::Zero(num_images, 10);

    for (size_t i=0; i<num_images; ++i) {
      unsigned char temp=0;
      fs.read((char*)&temp,sizeof(temp));
      Y(i,(int) temp) = 1.0;        
    }
    fs.close();
  }
  return Y;

}

int main (int argc, const char* argv[]) {

  if (argc != 2) {
    std::cout << "please provide path to mnist data ..." << std::endl;
    std::cout << "you can download the dataset at http://yann.lecun.com/exdb/mnist/" << std::endl;
    std::cout << std::endl << "usage: " << argv[1] << " path_to_data" << std::endl;
    return 1;
  }

  std::cout << "reading data ..." << std::endl;

  matrix_t X_train = read_mnist_images("../train-images-idx3-ubyte");
  matrix_t Y_train = read_mnist_labels("../train-labels-idx1-ubyte");

  matrix_t X_test = read_mnist_images("../t10k-images-idx3-ubyte");
  matrix_t Y_test = read_mnist_labels("../t10k-labels-idx1-ubyte");

  // number of optimization steps
  int max_steps = 300;
  // regularization parameter
  int lambda = 0.01;

  // specify network topology
  Eigen::VectorXi topo(4);
  topo << X_train.cols(), 300, 100, Y_test.cols();
  std::cout << "topology: " << topo.transpose() << std::endl;

  // initialize a neural network with given topology
  std::cout << "initializing network ..." << std::endl;
  NeuralNet nn(topo);

  std::cout << "scaling the data ..." << std::endl;
  nn.autoscale(X_train, Y_train);
  
  // train the network
  std::cout << "starting training" << std::endl;
  std::cout << "iter        error" << std::endl;
  double err;
  for (int i = 0; i < max_steps; ++i) {
    err = nn.loss(X_train, Y_train, lambda);
    nn.rprop();
    printf("%4i   %10.7f\n", i, err);
  }

  // test accuracy
  nn.forward_pass(X_test);
  matrix_t prediction = nn.get_activation();
  int correct = 0;
  int k;
  for (size_t i=0; i<Y_test.rows(); ++i) {
    prediction.row(i).maxCoeff(&k);
    correct += Y_test(i, k);
  }

  std::cout << "test accuracy: " << correct*1.0/Y_test.rows() << std::endl;

  return 0;
}



