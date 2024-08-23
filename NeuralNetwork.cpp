// NeuralNetwork.cpp
#include "NeuralNetwork.hpp"
#include <fstream>
#include <string>
typedef unsigned int uint;

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate) {
    this->topology = topology;
    this->learningRate = learningRate;

    for (uint i = 0; i < topology.size(); i++) {
        if (i == topology.size() - 1)
            neuronLayers.push_back(new RowVector(topology[i]));
        else
            neuronLayers.push_back(new RowVector(topology[i] + 1));

        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        if (i != topology.size() - 1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        if (i > 0) {
            if (i != topology.size() - 1) {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
            } else {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
}

void NeuralNetwork::propagateForward(RowVector& input) {
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

    for (uint i = 1; i < topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));
    }
}

void NeuralNetwork::calcErrors(RowVector& output) {
    (*deltas.back()) = output - (*neuronLayers.back());

    for (uint i = topology.size() - 2; i > 0; i--) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    }
}

void NeuralNetwork::updateWeights() {
    for (uint i = 0; i < topology.size() - 1; i++) {
        if (i != topology.size() - 2) {
            for (uint c = 0; c < weights[i]->cols() - 1; c++) {
                for (uint r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * 
                                                   activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * 
                                                   neuronLayers[i]->coeffRef(r);
                }
            }
        } else {
            for (uint c = 0; c < weights[i]->cols(); c++) {
                for (uint r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * 
                                                   activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * 
                                                   neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

void NeuralNetwork::propagateBackward(RowVector& output) {
    calcErrors(output);
    updateWeights();
}

void NeuralNetwork::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data) {
    for (uint i = 0; i < input_data.size(); i++) {
        std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
        propagateForward(*input_data[i]);
        std::cout << "Expected output is : " << *output_data[i] << std::endl;
        std::cout << "Output produced is : " << *neuronLayers.back() << std::endl;
        propagateBackward(*output_data[i]);
        std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
    }
}

Scalar activationFunction(Scalar x) {
    return tanhf(x);
}

Scalar activationFunctionDerivative(Scalar x) {
    return 1 - tanhf(x) * tanhf(x);
}

void ReadCSV(std::string filename, std::vector<RowVector*>& data) {
    data.clear();
    std::ifstream file(filename);
    std::string line, word;

    getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<Scalar> parsed_vec;

    while (getline(ss, word, ', ')) {
        parsed_vec.push_back(Scalar(std::stof(&word[0])));
    }
    uint cols = parsed_vec.size();
    data.push_back(new RowVector(cols));

    for (uint i = 0; i < cols; i++) {
        data.back()->coeffRef(1, i) = parsed_vec[i];
    }

    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            std::stringstream ss(line);
            data.push_back(new RowVector(1, cols));
            uint i = 0;
            while (getline(ss, word, ', ')) {
                data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
                i++;
            }
        }
    }
}

void genData(std::string filename) {
    std::ofstream file1(filename + "-in");
    std::ofstream file2(filename + "-out");

    for (uint r = 0; r < 1000; r++) {
        Scalar x = rand() / Scalar(RAND_MAX);
        Scalar y = rand() / Scalar(RAND_MAX);
        file1<< x << ", " << y << std::endl;
        file2<< 2 * x + 10 + y << std::endl;
    }
    file1.close();
    file2.close();
}
