#include <stdlib.h>
#include <math.h>
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

void layer_forward(layer_params *layer)
{
    for (int i = 0; i < layer->nb_neurons; i++)
    {
        float neuron_output = 0.0;
        for (int j = 0; j < layer->nb_inputs; j++)
        {
            neuron_output += layer->inputs[j] * layer->weights[i][j];
        }
        neuron_output += layer->biases[i];
        layer->outputs[i] = neuron_output;
    }
}

void layer_backward(layer_params *layer, float *dvalues)
{
    // gradients on parameters
    for (int i = 0; i < layer->nb_neurons; i++)
    {
        layer->dbiases[i] = dvalues[i]; // this is because there is no batch implemented yet
        for (int j = 0; j < layer->nb_inputs; j++)
        {
            layer->dweights[i][j] = layer->inputs[j] * dvalues[i];

            // gradient on values
            layer->dinputs[j] += dvalues[i] * layer->weights[i][j];
        }
    }
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

void relu_forward(activation_params *relu)
{
    for (int i = 0; i < relu->nb_neurons; i++)
    {
        relu->outputs[i] = (relu->inputs[i] > 0.0) ? relu->inputs[i] : 0.0;
    }
}

void relu_backward(activation_params *relu, float *dvalues)
{
    for (int i = 0; i < relu->nb_neurons; i++)
    {
        relu->dinputs[i] = (relu->inputs[i] <= 0) ? 0 : dvalues[i];
    }
}

void softmax_forward(activation_params *softmax)
{
    // get max value
    float softmax_input_max = 0;
    for (int i = 0; i < softmax->nb_neurons; i++)
    {
        if (softmax->inputs[i] > softmax_input_max)
        {
            softmax_input_max = softmax->inputs[i];
        }
    }
    // get unnormalized probabilities
    float exp_sum = 0;
    for (int i = 0; i < softmax->nb_neurons; i++)
    {
        softmax->outputs[i] = exp(softmax->inputs[i] - softmax_input_max);
        exp_sum += softmax->outputs[i];
    }
    // normalize the probabilities
    for (int i = 0; i < softmax->nb_neurons; i++)
    {
        softmax->outputs[i] = softmax->outputs[i] / exp_sum;
    }
}

void softmax_crossentropy_backward(activation_params *softmax, float *label_one_hot)
{
    for (int i = 0; i < softmax->nb_neurons; i++)
    {
        softmax->dinputs[i] = softmax->outputs[i] - label_one_hot[i];
    }
}

float calculate_crossentropy_loss(activation_params *softmax, int label_index)
{
    float prediction = (softmax->outputs[label_index] <= 0) ? 0.0000001 : softmax->outputs[label_index];
    return -log(prediction);
}