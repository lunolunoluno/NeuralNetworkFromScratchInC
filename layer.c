#include <stdlib.h>
#include "utils.h"
#include "layer.h"

void init_layer(int nb_inputs, int nb_neurons, layer_params *layer)
{
    layer->nb_inputs = nb_inputs;
    layer->nb_neurons = nb_neurons;

    layer->inputs = calloc(nb_inputs, sizeof(float));
    layer->dinputs = calloc(nb_inputs, sizeof(float));

    layer->biases = calloc(nb_neurons, sizeof(float));
    layer->dbiases = calloc(nb_neurons, sizeof(float));
    layer->weights = malloc(nb_neurons * sizeof(float *));
    layer->dweights = malloc(nb_neurons * sizeof(float *));
    for (int i = 0; i < nb_neurons; i++)
    {
        layer->weights[i] = calloc(nb_inputs, sizeof(float));
        layer->dweights[i] = calloc(nb_inputs, sizeof(float));
        
        // init biases with random values
        layer->biases[i] = get_random_float(-1.0, 1.0);
    }

    // init weights with random values
    float *weight_values = get_random_array(nb_inputs * nb_neurons, -1.0, 1.0);
    set_2darray_value(layer->weights, nb_inputs, nb_neurons, weight_values);
    free(weight_values);

    layer->outputs = calloc(nb_neurons, sizeof(float));
}

void destroy_layer(layer_params *layer)
{
    free(layer->inputs);
    free(layer->dinputs);
    free(layer->biases);
    free(layer->dbiases);
    for (int i = 0; i < layer->nb_neurons; i++)
    {
        free(layer->weights[i]);
        free(layer->dweights[i]);
    }
    free(layer->weights);
    free(layer->dweights);
    free(layer->outputs);
}

void init_activation(int nb_neurons, activation_params *activation)
{
    activation->nb_neurons = nb_neurons;
    activation->inputs = calloc(nb_neurons, sizeof(float));
    activation->dinputs = calloc(nb_neurons, sizeof(float));
    activation->outputs = calloc(nb_neurons, sizeof(float));
}

void destroy_activation(activation_params *activation)
{
    free(activation->inputs);
    free(activation->dinputs);
    free(activation->outputs);
}