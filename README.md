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
  
          cd path/to/nn
          mkdir build

* run cmake from within the build folder and compile the library using make
      
          cd build
          cmake ..
          make

* run the example code

          ./nn_example

* if cmake was able to download [googletest](http://code.google.com/p/googletest/) you can run unit tests now
    
          ./nn_test

Usage of the library
--------------------

### Preparing your data

Organize your training data into a *(m x n_input)* matrix containing the training inputs. Each row of this matrix correspods to a training sample and each column corresponds to a feature. Prepare a matrix of size *(m x n_output)* containing the target values, where *n_output* is the number of dimensions of the output. 

          matrix_t X(m, n_input);
          matrix_t Y(m, n_output);

          // fill with data

### Initializing the neural network

This neural network implementation only supports fully connected feed forward networks. The neurons are organized into *k* layers. There is at least one input layer and one output layer and an arbitrary number of hidden layers. Each neuron has outgoing connections to all neurons in the subsequent layer. The number of neurons in the input and the output layer is given by the dimensionality of the training data. After specifying the network topology you can create the neural network.The weights will be initialized randomly.

          Eigen::VectorXi topo(k);
          topo << n_input, n1, n2, ..., n_output;

          // initialize a neural network with given topology
          NeuralNet nn(topo);

### Training the network

Alternate between computing the quadratic loss of the neural network model and adapting the parameters until the loss converges. You can also specify a regularization parameter *lambda*, which adds an additional error for large weights and thereby avoiding overfitting.

          for (int i = 0; i < max_steps; ++i) {
            err = nn.loss(X, Y, lambda);
            nn.rprop();
          }

### Making predictions

If you trained a model using training data, you normally want to use this model for predicting new data. With neural networks you would use the forward pass for this. Afterwards the network output can be read from the activation of the output layer. 

          nn.forward_pass(X_test);
          matrix_t Y_test = nn.layer.back().a;

### Reading and writing models to disk

You can read and write neural network model to textfiles.

          // write model to disk
          nn.write(filename);

          // read model from disk
          NeuralNet nn(filename);

### Scaling the data

### Changing the floating number precision

