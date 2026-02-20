#ifndef _LAYER_
#define _LAYER_

typedef struct layer_params
{
    int nb_inputs;
    int nb_neurons;
    float *biases;
    float *dbiases;
    float **weights;
    float **dweights;
    float *inputs;
    float *dinputs;
    float *outputs;
} layer_params;

void init_layer(int nb_inputs, int nb_neurons, layer_params *params);
void destroy_layer(layer_params *params);

typedef struct activation_params
{
    int nb_neurons;
    float *inputs;
    float *dinputs;
    float *outputs;
} activation_params;

void init_activation(int nb_neurons, activation_params *params);
void destroy_activation(activation_params *params);

#endif