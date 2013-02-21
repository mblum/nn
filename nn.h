//
//  nn.h
//
//  Created by Manuel Blum on 02.07.12.
//  Copyright (c) 2012 Uni Freiburg. All rights reserved.
//

#ifndef __NN_H__
#define __NN_H__

#include <Eigen/Core>
#include <vector>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> array_t;

struct Layer {
  size_t size;
  matrix_t a, z, delta;
  matrix_t W, dEdW, DeltaW, directionW;
  vector_t b, dEdb, Deltab, directionb;
};

struct RpropParams {
  double Delta_0, Delta_max, Delta_min, eta_minus, eta_plus;
};

class NeuralNet {
public:
  /** Init neural net with given topology. */
  NeuralNet(Eigen::VectorXi &topology);
  /** Read neural net from file. */
  NeuralNet(const char * filename);
  /** Destructor. */
  ~NeuralNet();
  /** Initial weights randomly in range. */
  void init_weights(double range);
  /** Compute the loss function and its gradient. 
   *  Rows of X are instances, columns are features. 
   *  Lambda is a regularization parameter penalizing large weights. */
  double loss(const matrix_t &X, const matrix_t &Y, double lambda);
  /** Propagate data through the net.
   *  Rows of X are instances, columns are features. */
  void forward_pass(const matrix_t &X);
  /** Return activation of output layer. */
  matrix_t get_activation();
  /** Perform one iteration of RPROP using the default parameters. */
  void rprop();
  void rprop_reset();
  /** Perform one iteration of gradient descent using learning rate alpha. */
  void gradient_descent(double alpha);
  /** Write net parameter to file. */
  bool write(const char * filename);
  /** Returns the logistic function values f(x) given x. */
  static matrix_t sigmoid(const matrix_t &x);
  /** Returns the gradient f'(x) of the logistic function given f(x). */
  static matrix_t sigmoid_gradient(const matrix_t &x);
  /** Holds the layers of the neural net. */
  std::vector<Layer> layer;
  /** Compute autoscale parameters. */
  void autoscale(const matrix_t &X, const matrix_t &Y);
  void autoscale_reset();  
protected:
  /** Allocate memory and initialize default values. */
  void init_layer(Eigen::VectorXi &topology);
  /** Return delta w for given arguments. */ 
  double rprop_update(double &direction, double &Delta, double grad);
  /** Default parameters for RPROP. */ 
  static const RpropParams p;
  /** Scaling parameters. */
  vector_t Xshift;
  vector_t Xscale;
  vector_t Yshift;
  vector_t Yscale;
};

#endif
