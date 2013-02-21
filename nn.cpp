//
//  nn.cpp
//
//  Created by Manuel Blum on 02.07.12.
//  Copyright (c) 2012 Uni Freiburg. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <assert.h>

#include "nn.h"


// default parameters for Rprop
const RpropParams NeuralNet::p = {0.1, 50, 1e-6, 0.5, 1.2};

NeuralNet::NeuralNet(Eigen::VectorXi &topology) {
  init_layer(topology);
  init_weights(0.5);
  autoscale_reset();
}

NeuralNet::NeuralNet(const char *filename) {
  std::ifstream fs(filename, std::ios::in | std::ios::binary);
  if (fs) {
    // number of layers
    int num_layers;
    fs.read((char *)&num_layers, sizeof(int));
    Eigen::VectorXi topology(num_layers);
    // topology
    fs.read((char *) topology.data(), topology.rows() * sizeof(int) );
    init_layer(topology);
    autoscale_reset();
    // scaling parameters
    fs.read((char *) Xscale.data(), Xscale.size() * sizeof(F_TYPE));
    fs.read((char *) Xshift.data(), Xshift.size() * sizeof(F_TYPE));
    fs.read((char *) Yscale.data(), Yscale.size() * sizeof(F_TYPE));
    fs.read((char *) Yshift.data(), Yshift.size() * sizeof(F_TYPE));
    // weights
    for (int i=1; i<layer.size(); ++i) {
      fs.read((char *) layer[i].W.data(), layer[i].W.rows() * layer[i].W.cols() * sizeof(F_TYPE));
      fs.read((char *) layer[i].b.data(), layer[i].b.size() * sizeof(F_TYPE));
    }
  }
}

void NeuralNet::init_layer(Eigen::VectorXi &topology) {
  // init input layer
  Layer l;
  l.size = topology(0);
  layer.push_back(l);
  // init hidden and output layer
  for (int i=1; i<topology.size(); ++i) {
    Layer l;
    l.size  = topology(i);    
    l.W.setZero(l.size, layer[i-1].size);    
    l.b.setZero(l.size);
    l.dEdW.setZero(l.size, layer[i-1].size);
    l.dEdb.setZero(l.size);
    // set initial Delta
    l.DeltaW.resize(l.size, layer[i-1].size);
    l.Deltab.resize(l.size);    
    l.DeltaW.setConstant(p.Delta_0);
    l.Deltab.setConstant(p.Delta_0);
    layer.push_back(l);
  }  
}

void NeuralNet::init_weights(F_TYPE range) {
  for (int i=1; i<layer.size(); ++i) {
    layer[i].W.setRandom();
    layer[i].b.setRandom();
    layer[i].W *= range;
    layer[i].b *= range;
  }
}

NeuralNet::~NeuralNet() {
}

F_TYPE NeuralNet::loss(const matrix_t &X, const matrix_t &Y, F_TYPE lambda) {
  assert(layer.front().size == X.cols());
  assert(layer.back().size == Y.cols());
  // number of samples
  size_t m = X.rows();
  // forward pass
  forward_pass(X);
  // compute error
  array_t error = layer.back().a.array() - ((Y.rowwise()-Yshift.transpose()) * Yscale.asDiagonal()).array();
  // compute cost
  F_TYPE J = 0.5*error.square().sum()/m;
  // compute delta  
  layer.back().delta = (error * sigmoid_gradient(layer.back().a).array()).matrix();
  for (size_t i=layer.size()-2; i>0; --i) {
    matrix_t g = sigmoid_gradient(layer[i].a);
    layer[i].delta = (layer[i+1].delta * layer[i+1].W).cwiseProduct(g);
  }
  // compute partial derivatives and RPROP parameters
  for (int i=1; i<layer.size(); ++i) {
    // add regularization
    J += 0.5 * lambda * layer[i].W.array().square().sum() / m;
    matrix_t dEdW = (layer[i].delta.transpose() * layer[i-1].a + lambda*layer[i].W) / m;
    vector_t dEdb = layer[i].delta.colwise().sum().transpose() / m;
    layer[i].directionW = layer[i].dEdW.cwiseProduct(dEdW);
    layer[i].directionb = layer[i].dEdb.cwiseProduct(dEdb);
    layer[i].dEdW = dEdW;
    layer[i].dEdb = dEdb;
  }
  return J;
}

void NeuralNet::rprop() {
  for (int i=1; i<layer.size(); ++i) {
    for (int j=0; j<layer[i].size; ++j) {
      for (int k=0; k<layer[i-1].size; ++k) {
        F_TYPE u = rprop_update(layer[i].directionW(j,k), layer[i].DeltaW(j,k), layer[i].dEdW(j,k));
        layer[i].W(j,k) += u;
      }
      layer[i].b(j) += rprop_update(layer[i].directionb(j), layer[i].Deltab(j), layer[i].dEdb(j));
    }
  }
}

F_TYPE NeuralNet::rprop_update(F_TYPE &direction, F_TYPE &Delta, F_TYPE grad) {
  if (direction > 0) {
    Delta = std::min(Delta * p.eta_plus, p.Delta_max);
    direction = grad;
    if (grad > 0) return -Delta;
    else return Delta;
  } else if (direction < 0) {
    Delta = std::max(Delta * p.eta_minus, p.Delta_min);
    direction = 0;
    return 0;
  } else {
    direction = grad;
    if (grad > 0) return -Delta;
    else return Delta;
  }
}

void NeuralNet::rprop_reset() {
  for (int i=1; i<layer.size(); ++i) {
    layer[i].dEdW.setZero();
    layer[i].dEdb.setZero();
    layer[i].DeltaW.setConstant(p.Delta_0);
    layer[i].Deltab.setConstant(p.Delta_0);
  }
}

void NeuralNet::forward_pass(const matrix_t &X) {
  assert(layer.front().size == X.cols());
  // copy and scale data matrix
  layer[0].a = (X.rowwise()-Xshift.transpose()) * Xscale.asDiagonal();
  for (int i=1; i<layer.size(); ++i) {
    // compute input for current layer
    layer[i].z = layer[i-1].a * layer[i].W.transpose();
    // add bias
    layer[i].z.rowwise() += layer[i].b.transpose(); 
    // apply activation function
    layer[i].a = sigmoid(layer[i].z);
  }
}

matrix_t NeuralNet::get_activation() {
  return (layer.back().a * Yscale.asDiagonal().inverse()).rowwise() + Yshift.transpose();
}

matrix_t NeuralNet::sigmoid(const matrix_t &x) {
  return ((-x).array().exp() + 1.0).inverse().matrix();
}

matrix_t NeuralNet::sigmoid_gradient(const matrix_t &x) {
  return x.cwiseProduct((1.0-x.array()).matrix());
}

void NeuralNet::gradient_descent(F_TYPE alpha) {
  for (int i=1; i<layer.size(); ++i) {
    layer[i].W -= alpha*layer[i].dEdW;
    layer[i].b -= alpha*layer[i].dEdb;
  }
}

bool NeuralNet::write(const char *filename) {
  // open file
  std::ofstream fs(filename, std::ios::out | std::ios::binary);
  // write everything to disk
  if (fs) {
    // number of layers
    int num_layers = layer.size();
    fs.write((char *)&num_layers, sizeof(int));
    // topology
    for (int i=0; i<layer.size(); ++i) {
      int num_units = layer[i].size;
      fs.write((char *) &num_units, sizeof(int));
    }
    // scaling parameters
    fs.write((char *) Xscale.data(), Xscale.size() * sizeof(F_TYPE));
    fs.write((char *) Xshift.data(), Xshift.size() * sizeof(F_TYPE));
    fs.write((char *) Yscale.data(), Yscale.size() * sizeof(F_TYPE));
    fs.write((char *) Yshift.data(), Yshift.size() * sizeof(F_TYPE));
    // weights
    for (int i=1; i<layer.size(); ++i) {
      fs.write((char *) layer[i].W.data(), layer[i].W.rows() * layer[i].W.cols() * sizeof(F_TYPE));
      fs.write((char *) layer[i].b.data(), layer[i].b.size() * sizeof(F_TYPE));
    }
  }
  fs.close();
  return true;
}

void NeuralNet::autoscale(const matrix_t &X, const matrix_t &Y) {
  assert(layer.front().size == X.cols());
  assert(layer.back().size == Y.cols());
  // compute the mean of the input data
  Xshift = X.colwise().mean();
  // compute the standard deviation of the input data
  Xscale = (X.rowwise()-Xshift.transpose()).array().square().colwise().mean().array().sqrt().inverse();
  // compute the minimum target values
  Yshift = Y.colwise().minCoeff();
  // compute the maximum shifted target values
  Yscale = (Y.colwise().maxCoeff() - Yshift).array().inverse();
}

void NeuralNet::autoscale_reset() {
  Xscale = vector_t::Ones(layer.front().size);
  Xshift = vector_t::Zero(layer.front().size);
  Yscale = vector_t::Ones(layer.back().size);
  Yshift = vector_t::Zero(layer.back().size);
}
