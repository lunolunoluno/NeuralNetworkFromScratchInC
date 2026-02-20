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

    int nb_epochs = 1000;
    float learning_rate = 1.0;
    inputs[0] = -0.8326189893369458;
    inputs[1] = -0.5538462048218106;
    int label = 0;
    label_one_hot[label] = 1.0;

    init_layer(INPUT_SIZE, LAYER1_NB_NEURONS, &layer1);
    init_activation(LAYER1_NB_NEURONS, &layer1_relu); 

    init_layer(LAYER1_NB_NEURONS, LAYER2_NB_NEURONS, &layer2);
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

    for (int epoch = 0; epoch < nb_epochs; epoch++)
    {
        // FEED FORWARD LAYER 1
        layer_forward(&layer1, inputs);

        // LAYER 1 ReLU
        relu_forward(&layer1_relu, layer1.outputs);

        // FEED FORWARD LAYER 2
        layer_forward(&layer2, layer1_relu.outputs);

        // LAYER 2 SOFTMAX
        softmax_forward(&layer2_softmax, layer2.outputs);

        // CALCULATE CATEGORICAL CROSS-ENTROPY LOSS
        float loss = calculate_crossentropy_loss(&layer2_softmax, label);
        if (epoch % 100 == 0)
        {
            printf("epoch: %d, loss: %f\n", epoch, loss);
        }

        if (loss <= 0.0000001)
        {
            printf("LOSS <= 0.0000001 at epoch %d!\n", epoch);
            break;
        }

        // BACKPROPAGATION OF SOFTMAX + CROSS-ENTROPY LOSS (easier to implement)
        softmax_crossentropy_backward(&layer2_softmax, label_one_hot);

        // BACKPROPAGATION LAYER 2
        layer_backward(&layer2, layer1_relu.outputs, layer2_softmax.dinputs);

        // BACKPROPAGATION LAYER 1 RELU
        relu_backward(&layer1_relu, layer1.outputs, layer2.dinputs);

        // BACKPROPAGATION LAYER 1
        layer_backward(&layer1, inputs, layer1_relu.dinputs);

        // UPDATE PARAMETERS WITH SGD
        // Layer 1
        update_layer_params(&layer1, learning_rate);
        // Layer 2
        update_layer_params(&layer2, learning_rate);
    }

    layer_forward(&layer1, inputs);
    relu_forward(&layer1_relu, layer1.outputs);
    layer_forward(&layer2, layer1_relu.outputs);
    softmax_forward(&layer2_softmax, layer2.outputs);
    printf("NETWORK PREDICTION: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2_softmax.outputs[i]);
    }
    printf("\n");
    float loss = calculate_crossentropy_loss(&layer2_softmax, label);
    printf("FINAL LOSS: %f\n", loss);

    // FREE VARIABLES

    destroy_activation(&layer2_softmax);
    destroy_layer(&layer2);

    destroy_activation(&layer1_relu);
    destroy_layer(&layer1);

    free(label_one_hot);
    free(inputs);
    return 0;
}