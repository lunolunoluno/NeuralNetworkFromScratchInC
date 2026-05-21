#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#include "layer.h"

void init_layer(int nb_inputs, int nb_neurons, int batch_size, layer_params *layer)
{
    srand((unsigned int)time(NULL));

    layer->nb_inputs = nb_inputs;
    layer->nb_neurons = nb_neurons;
    layer->batch_size = batch_size;

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

    layer->dinputs = malloc(batch_size * sizeof(float *));
    layer->outputs = malloc(batch_size * sizeof(float *));

    for (int b = 0; b < batch_size; b++)
    {
        layer->outputs[b] = calloc(nb_neurons, sizeof(float));
        layer->dinputs[b] = calloc(nb_inputs, sizeof(float));
    }
}

void destroy_layer(layer_params *layer)
{
    free(layer->biases);
    free(layer->dbiases);
    for (int i = 0; i < layer->nb_neurons; i++)
    {
        free(layer->weights[i]);
        free(layer->dweights[i]);
    }
    free(layer->weights);
    free(layer->dweights);
    for (int i = 0; i < layer->batch_size; i++)
    {
        free(layer->dinputs[i]);
        free(layer->outputs[i]);
    }
    free(layer->dinputs);
    free(layer->outputs);
}

void layer_forward(layer_params *layer, float **inputs)
{
    for (int b = 0; b < layer->batch_size; b++)
    {
        for (int i = 0; i < layer->nb_neurons; i++)
        {
            float neuron_output = 0.0;
            for (int j = 0; j < layer->nb_inputs; j++)
            {
                neuron_output += inputs[b][j] * layer->weights[i][j];
            }
            neuron_output += layer->biases[i];
            layer->outputs[b][i] = neuron_output;
        }
    }
}

void layer_backward(layer_params *layer, float **layer_inputs, float **dvalues)
{
    // reset dinputs to 0
    for (int i = 0; i < layer->batch_size; i++)
    {
        for (int j = 0; j < layer->nb_inputs; j++)
        {
            layer->dinputs[i][j] = 0.0;
        }
    }

    // reset dweights to 0
    for (int i = 0; i < layer->nb_neurons; i++)
    {
        for (int j = 0; j < layer->nb_inputs; j++)
        {
            layer->dweights[i][j] = 0.0;
        }
    }

    // reset dbiases to 0
    for (int i = 0; i < layer->nb_neurons; i++)
    {
        layer->dbiases[i] = 0.0;
    }

    for (int b = 0; b < layer->batch_size; b++)
    {
        // gradients on parameters
        for (int i = 0; i < layer->nb_neurons; i++)
        {
            layer->dbiases[i] += dvalues[b][i];
            for (int j = 0; j < layer->nb_inputs; j++)
            {
                layer->dweights[i][j] += layer_inputs[b][j] * dvalues[b][i];

                // gradient on values
                layer->dinputs[b][j] += dvalues[b][i] * layer->weights[i][j];
            }
        }
    }
}

void update_layer_params(layer_params *layer, float learning_rate)
{
    for (int i = 0; i < layer->nb_neurons; i++)
    {
        for (int j = 0; j < layer->nb_inputs; j++)
        {
            layer->weights[i][j] += -learning_rate * layer->dweights[i][j];
        }
        layer->biases[i] += -learning_rate * layer->dbiases[i];
    }
}

void init_activation(int nb_neurons, int batch_size, activation_params *activation)
{
    activation->nb_neurons = nb_neurons;
    activation->batch_size = batch_size;

    activation->dinputs = malloc(batch_size * sizeof(float *));
    activation->outputs = malloc(batch_size * sizeof(float *));
    for (int i = 0; i < batch_size; i++)
    {
        activation->dinputs[i] = calloc(nb_neurons, sizeof(float));
        activation->outputs[i] = calloc(nb_neurons, sizeof(float));
    }
}

void destroy_activation(activation_params *activation)
{
    for (int i = 0; i < activation->batch_size; i++)
    {
        free(activation->dinputs[i]);
        free(activation->outputs[i]);
    }
    free(activation->dinputs);
    free(activation->outputs);
}

void relu_forward(activation_params *relu, float **inputs)
{
    for (int b = 0; b < relu->batch_size; b++)
    {
        for (int i = 0; i < relu->nb_neurons; i++)
        {
            relu->outputs[b][i] = (inputs[b][i] > 0.0) ? inputs[b][i] : 0.0;
        }
    }
}

void relu_backward(activation_params *relu, float **relu_inputs, float **dvalues)
{
    for (int b = 0; b < relu->batch_size; b++)
    {
        for (int i = 0; i < relu->nb_neurons; i++)
        {
            relu->dinputs[b][i] = (relu_inputs[b][i] <= 0) ? 0 : dvalues[b][i];
        }
    }
}

void softmax_forward(activation_params *softmax, float **inputs)
{
    for (int b = 0; b < softmax->batch_size; b++)
    {
        // get max value
        float softmax_input_max = inputs[b][0];
        for (int i = 0; i < softmax->nb_neurons; i++)
        {
            if (inputs[b][i] > softmax_input_max)
            {
                softmax_input_max = inputs[b][i];
            }
        }
        // get unnormalized probabilities
        float exp_sum = 0;
        for (int i = 0; i < softmax->nb_neurons; i++)
        {
            softmax->outputs[b][i] = exp(inputs[b][i] - softmax_input_max);
            exp_sum += softmax->outputs[b][i];
        }
        // normalize the probabilities
        for (int i = 0; i < softmax->nb_neurons; i++)
        {
            softmax->outputs[b][i] = softmax->outputs[b][i] / exp_sum;
        }
    }
}

void softmax_crossentropy_backward(activation_params *softmax, int **label_one_hot)
{
    for (int b = 0; b < softmax->batch_size; b++)
    {
        for (int i = 0; i < softmax->nb_neurons; i++)
        {
            softmax->dinputs[b][i] = softmax->outputs[b][i] - label_one_hot[b][i];
        }
    }
}

float calculate_crossentropy_loss(activation_params *softmax, int *label_index)
{
    if (softmax->batch_size == 0) return 0.0f;
    float sum_loss = 0;
    for (int i = 0; i < softmax->batch_size; i++)
    {
        if (label_index[i] >= 0) // if label_index == -1, then ignore
        {
            float o = softmax->outputs[i][label_index[i]];
            sum_loss += -log(clip_value(o, 1e-7, 1 - 1e-7));
        }
    }
    return sum_loss / softmax->batch_size;
}