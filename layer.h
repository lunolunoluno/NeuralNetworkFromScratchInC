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
    float *dinputs;
    float *outputs;
} layer_params;

void init_layer(int nb_inputs, int nb_neurons, layer_params *layer);
void destroy_layer(layer_params *layer);
void layer_forward(layer_params *layer, float* inputs);
void layer_backward(layer_params *layer, float* layer_inputs, float *dvalues);

typedef struct activation_params
{
    int nb_neurons;
    float *dinputs;
    float *outputs;
} activation_params;

void init_activation(int nb_neurons, activation_params *activation);
void destroy_activation(activation_params *activation);
void relu_forward(activation_params *relu, float *inputs);
void relu_backward(activation_params *relu, float* relu_inputs, float *dvalues);
void softmax_forward(activation_params *softmax, float *inputs);
void softmax_crossentropy_backward(activation_params *softmax, float* label_one_hot);

float calculate_crossentropy_loss(activation_params *softmax, int label_index);

#endif