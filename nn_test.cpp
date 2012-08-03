//
//  nn_test.cpp
//
//  Created by Manuel Blum on 02.07.12.
//  Copyright (c) 2012 Uni Freiburg. All rights reserved.
//

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

#include "nn.h"

TEST(nn, sigmoid1) {
  matrix_t X(3,3);
  matrix_t s(3,3);
  X << -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2;
  s <<  0.1192, 0.1824, 0.2689, 0.3775, 0.5, 0.6225, 0.7311, 0.8176, 0.8808;
  matrix_t Y = NeuralNet::sigmoid(X);
  ASSERT_NEAR((Y.array()-s.array()).maxCoeff(), 0.0, 1e-4);
}

TEST(nn, sigmoid2) {
  matrix_t X = matrix_t::Random(100,200) * 2;
  matrix_t Y = NeuralNet::sigmoid(X);
  ASSERT_EQ(X.rows(), Y.rows());
  ASSERT_EQ(X.cols(), Y.cols());
  for (int i=0; i<X.rows(); ++i) {
    for (int j=0; j<X.cols(); ++j) {
      ASSERT_NEAR(Y(i,j), 1/(1+exp(-X(i,j))), 1e-7);
    }
  }
}

TEST(nn, sigmoid_gradient1) {
  matrix_t X(3,3);
  matrix_t s(3,3);
  X << -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2;
  s <<  0.105, 0.1491, 0.1966, 0.2350, 0.25, 0.2350, 0.1966, 0.1491, 0.1050;
  matrix_t sig = NeuralNet::sigmoid(X);
  matrix_t Y = NeuralNet::sigmoid_gradient(sig);
  ASSERT_NEAR((Y.array()-s.array()).maxCoeff(), 0.0, 1e-4);
}

TEST(nn, sigmoid_gradient2) {
  matrix_t X = matrix_t::Random(100,200) * 2;
  matrix_t sig = NeuralNet::sigmoid(X);
  matrix_t Y = NeuralNet::sigmoid_gradient(sig);
  ASSERT_EQ(X.rows(), Y.rows());
  ASSERT_EQ(X.cols(), Y.cols());
  for (int i=0; i<X.rows(); ++i) {
    for (int j=0; j<X.cols(); ++j) {
      double sigmoid = 1/(1+exp(-X(i,j)));
      ASSERT_NEAR(Y(i,j), sigmoid*(1-sigmoid), 1e-7);
    }
  }
}

// compare analytical with numerical gradients
TEST(nn, gradient) {
  Eigen::VectorXi topo(4);
  topo << 3, 10, 10, 2;
  NeuralNet nn(topo);
  nn.init_weights(0.5);
  int m = 100;
  matrix_t X = matrix_t::Random(m,3);
  matrix_t Y = matrix_t::Random(m,2);
  Y *= 0.5+0.5;
  double lambda = 0.01;
  double e = 1e-4;
  ASSERT_EQ(nn.layer[0].size, topo(0));
  for (int i=1; i<nn.layer.size(); ++i) {
    ASSERT_EQ(nn.layer[i].W.rows(), nn.layer[i].size);
    ASSERT_EQ(nn.layer[i].W.cols(), nn.layer[i-1].size);
    ASSERT_EQ(nn.layer[i].b.rows(), nn.layer[i].size);
    ASSERT_EQ(nn.layer[i].size, topo(i));
    for (int j=0; j<nn.layer[i].W.rows(); ++j) {
      for (int k=0; k<nn.layer[i].W.cols(); ++k) {
        double w = nn.layer[i].W(j,k);
        nn.layer[i].W(j,k) = w - e;
        double j1 = nn.loss(X, Y, lambda);
        nn.layer[i].W(j,k) = w + e;
        double j2 = nn.loss(X, Y, lambda);
        nn.layer[i].W(j,k) = w;
        nn.loss(X, Y, lambda);
        ASSERT_NEAR((j2-j1)/(2*e), nn.layer[i].dEdW(j,k), 1e-9);
      }
    }
    for (int j=0; j<nn.layer[i].b.rows(); ++j) {
      double b = nn.layer[i].b(j);
      nn.layer[i].b(j) = b - e;
      double j1 = nn.loss(X, Y, lambda);
      nn.layer[i].b(j) = b + e;
      double j2 = nn.loss(X, Y, lambda);
      nn.layer[i].b(j) = b;
      nn.loss(X, Y, lambda);
      ASSERT_NEAR((j2-j1)/(2*e), nn.layer[i].dEdb(j), 1e-9);
    }
  }
}

TEST(nn, readwrite) {
  Eigen::VectorXi topo(4);
  topo << 3, 10, 10, 2;
  NeuralNet nn(topo);
  nn.init_weights(0.5);
  matrix_t X = matrix_t::Random(100,3);
  matrix_t Y = matrix_t::Random(100,2);
  Y *= 0.5+0.5;
  nn.write("/tmp/nn.txt");
  NeuralNet nnclone("/tmp/nn.txt");
  ASSERT_EQ(topo.size(), nnclone.layer.size());
  for (int i=0; i<topo.size(); ++i) ASSERT_EQ(topo(i), nnclone.layer[i].size);
  for (int i=1; i<nn.layer.size(); ++i) {
    for (int j=0; j<nn.layer[i].W.rows(); ++j) {
      for (int k=0; k<nn.layer[i].W.cols(); ++k) {
        ASSERT_NEAR(nn.layer[i].W(j,k), nnclone.layer[i].W(j,k), 1e-6);
      }
    }
    for (int j=0; j<nn.layer[i].b.rows(); ++j) {
      ASSERT_NEAR(nn.layer[i].b(j), nnclone.layer[i].b(j), 1e-6);
    }
  }
}

TEST(nn, testfunction) {
  srand((unsigned int) time(NULL));
  Eigen::VectorXi topo(4);
  topo << 2, 10, 10, 1;
  NeuralNet nn(topo);
  matrix_t X = matrix_t::Random(2000,2);
  matrix_t Y = matrix_t::Random(2000,1);
  for (int i=0;i<X.rows(); ++i) {
    Y(i,0) = (X(i,0) * X(i,1) + 1.2)*0.4;
  }
  double lambda = 0.01, err;
  for (int i=0;i<300;++i) {
    err = nn.loss(X, Y, lambda);
    nn.rprop();
  }
  ASSERT_LE(err, 0.001);
  nn.write("/tmp/nn.txt");
  NeuralNet nnclone("/tmp/nn.txt");
  ASSERT_NEAR(nn.loss(X, Y, lambda), nnclone.loss(X, Y, lambda), 1e-8);
}
