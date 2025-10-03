#include <stdlib.h>
#include <time.h>
#include "layer.h"

void layer_init(layer_dense *layer, int n_neurons, int n_inputs){
    // init the weights with random values between -1 and 1
    srand((unsigned int)time(NULL));
    ndarray weights;
    int weights_shape[2] = {n_neurons, n_inputs};
    init_ndarray(&weights, 2, weights_shape, 0.0);
    float *weights_values = get_random_array(n_neurons*n_inputs, -1, 1);
    set_data_ndarray(&weights, weights_values);
    layer->weights = weights;
    free(weights_values);

    // init the bias with just zeros (for now)
    ndarray bias;
    int bias_shape[1] = {n_neurons};
    init_ndarray(&bias, 1, bias_shape, 0.0);
    layer->bias = bias;

    // init inputs and outputs with default values
    ndarray inputs;
    int inputs_shape[1] = {n_inputs};
    init_ndarray(&inputs, 1, inputs_shape, 0.0);
    layer->inputs = inputs;

    ndarray outputs;
    int outputs_shape[1] = {n_inputs};
    init_ndarray(&outputs, 1, outputs_shape, 0.0);
    layer->outputs = outputs;
}

void set_layer_weights(layer_dense *layer, ndarray weights){
    destroy_ndarray(&layer->weights);
    layer->weights = weights;
}

void set_layer_bias(layer_dense *layer, ndarray bias){
    destroy_ndarray(&layer->bias);
    layer->bias = bias;
}

void layer_forward(layer_dense *layer, ndarray inputs){
    destroy_ndarray(&layer->inputs);
    layer->inputs = inputs;

    ndarray weights_T = transpose_copy_ndarray(layer->weights);
    ndarray layer_outputs = matrix_product(inputs, weights_T);//X * W
    add_vec_to_matrix(&layer_outputs, layer->bias.data);//+ biases

    destroy_ndarray(&weights_T);
    destroy_ndarray(&layer->outputs);
    layer->outputs = layer_outputs;
}

// void layer_backward(layer_dense *layer);

void destroy_layer(layer_dense *layer){
    destroy_ndarray(&layer->bias);
    destroy_ndarray(&layer->weights);
    destroy_ndarray(&layer->inputs);
    destroy_ndarray(&layer->outputs);
}

