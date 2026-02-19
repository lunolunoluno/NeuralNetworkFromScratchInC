#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "layer.h"
#include "utils.h"

#define INPUT_SIZE 2
#define LAYER1_NB_NEURONS 3
#define LAYER2_NB_NEURONS 3 // this is also the output


int main()
{
    // INIT VARIABLES

    float *inputs = calloc(INPUT_SIZE, sizeof(float));
    float *label_one_hot = calloc(LAYER2_NB_NEURONS, sizeof(float));

    layer_params layer1;
    activation_params layer1_relu;

    layer_params layer2;
    activation_params layer2_softmax;

    // GIVE VARIABLES INITIAL VALUES
    
    float learning_rate = 1.0;
    inputs[0] = -0.8326189893369458;
    inputs[1] = -0.5538462048218106;
    int label = 0;
    label_one_hot[label] = 1.0;

    init_layer(INPUT_SIZE, LAYER1_NB_NEURONS, &layer1);
    layer1.inputs[0] = inputs[0];
    layer1.inputs[1] = inputs[1];

    float layer1_weights_values[INPUT_SIZE * LAYER1_NB_NEURONS] = {0.01764052, 0.02240893,
                                                                   0.00400157, 0.01867558,
                                                                   0.00978738, -0.00977278};
    set_2darray_value(layer1.weights, INPUT_SIZE, LAYER1_NB_NEURONS, layer1_weights_values);
    layer1.biases[0] = 0.2;
    layer1.biases[1] = 0.003;
    layer1.biases[2] = 0.0005;
    init_activation(LAYER1_NB_NEURONS, &layer1_relu);

    init_layer(LAYER1_NB_NEURONS, LAYER2_NB_NEURONS, &layer2);
    float layer2_weights_values[LAYER1_NB_NEURONS * LAYER2_NB_NEURONS] = {0.00950088, -0.00151357, -0.00103219,
                                                                          0.00410599, 0.00144044, 0.01454273,
                                                                          0.00761038, 0.00121675, 0.00443863};
    set_2darray_value(layer2.weights, LAYER1_NB_NEURONS, LAYER2_NB_NEURONS, layer2_weights_values);
    layer2.biases[0] = -0.1;
    layer2.biases[1] = 0.002;
    layer2.biases[2] = -0.0005;
    init_activation(LAYER2_NB_NEURONS, &layer2_softmax);

    printf("INPUTS: ");
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        printf("%f, ", inputs[i]);
    }
    printf("\n");
    printf("ONE-HOT LABEL: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f, ", label_one_hot[i]);
    }
    printf("\n");


    // FEED FORWARD LAYER 1
    layer_forward(&layer1);

    printf("LAYER 1 OUTPUT: ");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer1.outputs[i]);
        // transfer output to activation function
        layer1_relu.inputs[i] = layer1.outputs[i];
    }
    printf("\n");

    // LAYER 1 ReLU
    relu_forward(&layer1_relu);

    printf("LAYER 1 RELU: ");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer1_relu.outputs[i]);
        // transfer value to next layer
        layer2.inputs[i] = layer1_relu.outputs[i];
    }
    printf("\n");

    // FEED FORWARD LAYER 2
    layer_forward(&layer2);

    printf("LAYER 2 OUTPUT: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2.outputs[i]);
        // transfer output to activation function
        layer2_softmax.inputs[i] = layer2.outputs[i];
    }
    printf("\n");

    // LAYER 2 SOFTMAX
    softmax_forward(&layer2_softmax);

    printf("LAYER 2 SOFTMAX: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2_softmax.outputs[i]);
    }
    printf("\n");

    // CALCULATE CATEGORICAL CROSS-ENTROPY LOSS
    float loss = calculate_crossentropy_loss(&layer2_softmax, label);
    printf("CATEGORICAL CROSS-ENTROPY LOSS %f\n", loss);

    // BACKPROPAGATION OF SOFTMAX + CROSS-ENTROPY LOSS (easier to implement)
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        layer2_softmax.dinputs[i] = layer2_softmax.outputs[i] - label_one_hot[i];
    }

    printf("BACKPROPAGATION OF SOFTMAX + CROSS-ENTROPY LOSS: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2_softmax.dinputs[i]);
    }
    printf("\n");

    // BACKPROPAGATION LAYER 2
    // gradients on parameters
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        layer2.dbiases[i] = layer2_softmax.dinputs[i]; // this is because there is no batch implemented yet
        for (int j = 0; j < LAYER1_NB_NEURONS; j++)
        {
            layer2.dweights[i][j] = layer2.inputs[j] * layer2_softmax.dinputs[i];

            // gradient on values
            layer2.dinputs[j] += layer2_softmax.dinputs[i] * layer2.weights[i][j];
        }
    }

    printf("BACKPROPAGATION OF LAYER 2 inputs gradient:");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer2.dinputs[i]);
    }
    printf("\n");
    printf("BACKPROPAGATION OF LAYER 2 weights gradient: \n");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        for (int j = 0; j < LAYER1_NB_NEURONS; j++)
        {
            printf("%f,", layer2.dweights[i][j]);
        }
        printf("\n");
    }
    printf("BACKPROPAGATION OF LAYER 2 biases gradient:");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2.dbiases[i]);
    }
    printf("\n");

    // BACKPROPAGATION LAYER 1 RELU
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        layer1_relu.dinputs[i] = (layer1_relu.inputs[i] <= 0) ? 0 : layer2.dinputs[i];
    }

    printf("LAYER 1 RELU BACKWARD: ");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer1_relu.dinputs[i]);
    }
    printf("\n");

    // BACKPROPAGATION LAYER 1
    // gradients on parameters
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        layer1.dbiases[i] = layer1_relu.dinputs[i]; // this is because there is no batch implemented yet
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            layer1.dweights[i][j] = layer1.inputs[j] * layer1_relu.dinputs[i];

            // gradient on values
            // no need to calculate it for the first layer
        }
    }
    
    printf("BACKPROPAGATION OF LAYER 1 weights gradient: \n");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            printf("%f,", layer1.dweights[i][j]);
        }
        printf("\n");
    }
    printf("BACKPROPAGATION OF LAYER 1 biases gradient:");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer1.dbiases[i]);
    }
    printf("\n");


    // UPDATE PARAMETERS WITH SGD
    // Layer 1
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            layer1.weights[i][j] += -learning_rate * layer1.dweights[i][j];
        }
        layer1.biases[i] += -learning_rate * layer1.dbiases[i];
    }
    printf("UPDATED LAYER 1 weights: \n");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            printf("%f,", layer1.weights[i][j]);
        }
        printf("\n");
    }
    printf("UPDATED LAYER 1 biases:");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer1.biases[i]);
    }
    printf("\n");

    // Layer 2
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        for (int j = 0; j < LAYER1_NB_NEURONS; j++)
        {
            layer2.weights[i][j] += -learning_rate * layer2.dweights[i][j];
        }
        layer2.biases[i] += -learning_rate * layer2.dbiases[i];
    }
    printf("UPDATED LAYER 2 weights: \n");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        for (int j = 0; j < LAYER1_NB_NEURONS; j++)
        {
            printf("%f,", layer2.weights[i][j]);
        }
        printf("\n");
    }
    printf("UPDATED LAYER 2 biases:");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2.biases[i]);
    }
    printf("\n");

    // FREE VARIABLES

    destroy_activation(&layer2_softmax);
    destroy_layer(&layer2);

    destroy_activation(&layer1_relu);
    destroy_layer(&layer1);

    free(label_one_hot);
    free(inputs);
    return 0;
}