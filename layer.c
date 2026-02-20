#include <stdlib.h>
#include "utils.h"
#include "layer.h"

void init_layer(int nb_inputs, int nb_neurons, layer_params *params)
{
    params->nb_inputs = nb_inputs;
    params->nb_neurons = nb_neurons;

    params->inputs = calloc(nb_inputs, sizeof(float));
    params->dinputs = calloc(nb_inputs, sizeof(float));

    params->biases = calloc(nb_neurons, sizeof(float));
    params->dbiases = calloc(nb_neurons, sizeof(float));
    params->weights = malloc(nb_neurons * sizeof(float *));
    params->dweights = malloc(nb_neurons * sizeof(float *));
    for (int i = 0; i < nb_neurons; i++)
    {
        params->weights[i] = calloc(nb_inputs, sizeof(float));
        params->dweights[i] = calloc(nb_inputs, sizeof(float));
    }

    // init weights with random values
    float *weight_values = get_random_array(nb_inputs * nb_neurons, -1.0, 1.0);
    set_2darray_value(params->weights, nb_inputs, nb_neurons, weight_values);
    free(weight_values);

    params->outputs = calloc(nb_neurons, sizeof(float));
}

void destroy_layer(layer_params *params)
{
    free(params->inputs);
    free(params->dinputs);
    free(params->biases);
    free(params->dbiases);
    for (int i = 0; i < params->nb_neurons; i++)
    {
        free(params->weights[i]);
        free(params->dweights[i]);
    }
    free(params->weights);
    free(params->dweights);
    free(params->outputs);
}

void init_activation(int nb_neurons, activation_params *params)
{
    params->nb_neurons = nb_neurons;
    params->inputs = calloc(nb_neurons, sizeof(float));
    params->dinputs = calloc(nb_neurons, sizeof(float));
    params->outputs = calloc(nb_neurons, sizeof(float));
}

void destroy_activation(activation_params *params)
{
    free(params->inputs);
    free(params->dinputs);
    free(params->outputs);
}