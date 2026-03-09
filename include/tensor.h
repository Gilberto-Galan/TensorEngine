#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <random>
#include <iostream>

class Tensor {
public:
    std::vector<float> data;
    int rows, cols;

    Tensor(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

    // Inicialización con valores aleatorios (simulando pesos de una red)
    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &val : data) val = dist(gen);
    }

    // Helper para depuración (solo usar con matrices pequeñas)
    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << "\n";
        }
    }
};

#endif