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

void init_layer(int nb_inputs, int nb_neurons, layer_params *layer);
void destroy_layer(layer_params *layer);

typedef struct activation_params
{
    int nb_neurons;
    float *inputs;
    float *dinputs;
    float *outputs;
} activation_params;

void init_activation(int nb_neurons, activation_params *activation);
void destroy_activation(activation_params *activation);

#endif