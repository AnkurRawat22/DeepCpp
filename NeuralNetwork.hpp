// NeuralNetwork.hpp
#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Eigen/Eigen>
#include <iostream>
#include <vector>
typedef unsigned int uint;


typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

    void propagateForward(RowVector& input);
    void propagateBackward(RowVector& output);
    void calcErrors(RowVector& output);
    void updateWeights();
    void train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data);

    std::vector<RowVector*> neuronLayers;
    std::vector<RowVector*> cacheLayers;
    std::vector<RowVector*> deltas;
    std::vector<Matrix*> weights;
    Scalar learningRate;

private:
    std::vector<uint> topology;
};

Scalar activationFunction(Scalar x);
Scalar activationFunctionDerivative(Scalar x);

void ReadCSV(std::string filename, std::vector<RowVector*>& data);
void genData(std::string filename);

#endif // NEURALNETWORK_HPP
