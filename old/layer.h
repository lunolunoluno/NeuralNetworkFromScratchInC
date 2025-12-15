#ifndef _LAYER_
#define _LAYER_

#include "array_utils.h"

typedef struct layer_dense
{
    int n_neurons;
    ndarray inputs;
    ndarray weights;
    ndarray bias;
    ndarray outputs; 
}layer_dense;

void layer_init(layer_dense *layer, int n_neurons, int n_inputs);
void set_layer_weights(layer_dense *layer, ndarray weights);
void set_layer_bias(layer_dense *layer, ndarray bias);
void layer_forward(layer_dense *layer, ndarray inputs);
void layer_backward(layer_dense *layer);
void destroy_layer(layer_dense *layer);

#endif
