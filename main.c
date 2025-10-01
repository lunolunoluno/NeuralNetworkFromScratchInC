#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "array_utils.h"

int main(){
    srand((unsigned int)time(NULL));

    int n_inputs = 4;
    int n_batch = 3;
    int n_neurons_l1 = 3;
    int n_neurons_l2 = 3;

    // initialize batch of inputs
    ndarray inputs;
    int inputs_shape[2] = {n_batch, n_inputs};
    init_ndarray(&inputs, 2, inputs_shape, 0.0);
    float inputs_values[12] = {1.0, 2.0, 3.0, 2.5, 
                               2.0, 5.0, -1.0, 2.0, 
                               -1.5, 2.7, 3.3, -0.8};
    set_data_ndarray(&inputs, inputs_values);

    // initialize weights + biases for layer 1
    ndarray weights1;
    int weights1_shape[2] = {n_neurons_l1, n_inputs};
    init_ndarray(&weights1, 2, weights1_shape, 0.0);
    float weights1_values[12] = {0.2, 0.8, -0.5, 1.0,
                                0.5, -0.91, 0.26, -0.5,
                                -0.26, -0.27, 0.17, 0.87};
    set_data_ndarray(&weights1, weights1_values);
    float biases1[3] = {2.0, 3.0, 0.5};

    // initialize weights + biases for layer 2
    ndarray weights2;
    int weights2_shape[2] = {n_neurons_l2, n_neurons_l1};
    init_ndarray(&weights2, 2, weights2_shape, 0.0);
    float weights2_values[9] = {0.1, -0.14, 0.5,
                                -0.5, 0.12, -0.33,
                                -0.44, 0.73, -0.13};
    set_data_ndarray(&weights2, weights2_values);
    float biases2[3] = {-1.0, 2.0, -0.5};

    // calculate output for layer 1
    ndarray weights1_T = transpose_copy_ndarray(weights1);
    ndarray layer1_outputs = matrix_product(inputs, weights1_T);//X * W
    add_vec_to_matrix(&layer1_outputs, biases1);//+ biases

    printf("Layer 1 output:\n");
    print_ndarray(layer1_outputs);

    // calculate output for layer 2
    ndarray weights2_T = transpose_copy_ndarray(weights2);
    ndarray layer2_outputs = matrix_product(layer1_outputs, weights2_T);//X * W
    add_vec_to_matrix(&layer2_outputs, biases2);//+ biases

    printf("Layer 2 output:\n");
    print_ndarray(layer2_outputs);


    // destroy allocated variables
    destroy_ndarray(&weights1);
    destroy_ndarray(&weights1_T);
    destroy_ndarray(&weights2);
    destroy_ndarray(&weights2_T);
    destroy_ndarray(&inputs);
    destroy_ndarray(&layer1_outputs);
    destroy_ndarray(&layer2_outputs);

    return 0;
}