#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "array_utils.h"
#include "layer.h"

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

    ndarray bias1;
    int bias1_shape[1] = {n_neurons_l1};
    init_ndarray(&bias1, 1, bias1_shape, 0.0);
    float bias1_values[3] = {2.0, 3.0, 0.5};
    set_data_ndarray(&bias1, bias1_values);

    // initialize weights + biases for layer 2
    ndarray weights2;
    int weights2_shape[2] = {n_neurons_l2, n_neurons_l1};
    init_ndarray(&weights2, 2, weights2_shape, 0.0);
    float weights2_values[9] = {0.1, -0.14, 0.5,
                                -0.5, 0.12, -0.33,
                                -0.44, 0.73, -0.13};
    set_data_ndarray(&weights2, weights2_values);

    ndarray bias2;
    int bias2_shape[1] = {n_neurons_l2};
    init_ndarray(&bias2, 1, bias2_shape, 0.0);
    float bias2_values[3] = {-1.0, 2.0, -0.5};
    set_data_ndarray(&bias2, bias2_values);
    

    // calculate output for layer 1
    layer_dense layer1;
    layer_init(&layer1, n_neurons_l1, n_inputs);    
    set_layer_weights(&layer1, weights1);
    set_layer_bias(&layer1, bias1);
    
    layer_forward(&layer1, inputs);

    printf("Layer 1 output:\n");
    print_ndarray(layer1.outputs);

    // // calculate output for layer 2
    layer_dense layer2;
    layer_init(&layer2, n_neurons_l2, n_neurons_l1);    
    set_layer_weights(&layer2, weights2);
    set_layer_bias(&layer2, bias2);
    
    layer_forward(&layer2, copy_ndarray(layer1.outputs)); // important to make a copy of layer1.outputs to avoid double-free

    printf("Layer 2 output:\n");
    print_ndarray(layer2.outputs);


    // destroy allocated variables
    destroy_layer(&layer1);
    destroy_layer(&layer2);

    return 0;
}