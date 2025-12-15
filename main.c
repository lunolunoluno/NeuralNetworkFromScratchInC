#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 2
#define LAYER1_NB_NEURONS 3
#define LAYER2_NB_NEURONS 3 // this is also the output

// temp function to set weights to a specific value
void set_weights_value(float **weights, int nb_input, int nb_output, float values[])
{
    for (int i = 0; i < nb_output; i++)
    {
        for (int j = 0; j < nb_input; j++)
        {
            weights[i][j] = values[(i * nb_input) + j];
        }
    }
}

int main()
{
    // INIT VARIABLES
    float *inputs = calloc(INPUT_SIZE, sizeof(float));

    float *layer1_biases = calloc(LAYER1_NB_NEURONS, sizeof(float));
    float **layer1_weights = malloc(LAYER1_NB_NEURONS * sizeof(float *));
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        layer1_weights[i] = calloc(INPUT_SIZE, sizeof(float));
    }
    float *layer1_output = calloc(LAYER1_NB_NEURONS, sizeof(float));
    float *layer1_relu = calloc(LAYER1_NB_NEURONS, sizeof(float));

    float *layer2_biases = calloc(LAYER2_NB_NEURONS, sizeof(float));
    float **layer2_weights = malloc(LAYER2_NB_NEURONS * sizeof(float *));
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        layer2_weights[i] = calloc(LAYER1_NB_NEURONS, sizeof(float));
    }
    float *layer2_output = calloc(LAYER2_NB_NEURONS, sizeof(float));

    // GIVE VARIABLES INITIAL VALUES
    inputs[0] = -0.8326189893369458;
    inputs[1] = -0.5538462048218106;
    int label = 0;

    float layer1_weights_values[INPUT_SIZE * LAYER1_NB_NEURONS] = {0.01764052, 0.02240893,
                                                                   0.00400157, 0.01867558,
                                                                   0.00978738, -0.00977278};
    set_weights_value(layer1_weights, INPUT_SIZE, LAYER1_NB_NEURONS, layer1_weights_values);
    layer1_biases[0] = 0.2;
    layer1_biases[1] = 0.003;
    layer1_biases[2] = 0.0005;

    float layer2_weights_values[LAYER1_NB_NEURONS * LAYER2_NB_NEURONS] = {0.1, -0.14, 0.5,
                                                                          -0.5, 0.12, -0.33,
                                                                          -0.44, 0.73, -0.13};
    set_weights_value(layer2_weights, LAYER1_NB_NEURONS, LAYER2_NB_NEURONS, layer2_weights_values);
    layer2_biases[0] = -1.0;
    layer2_biases[1] = 2.0;
    layer2_biases[2] = -0.5;

    // FEED FORWARD LAYER 1
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        float neuron_output = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            neuron_output += inputs[j] * layer1_weights[i][j];
            // printf("%f * %f = %f\n", inputs[j], layer1_weights[i][j], neuron_output);
        }
        neuron_output += layer1_biases[i];
        // printf("+ %f = %f\n", layer1_biases[i], neuron_output);
        layer1_output[i] = neuron_output;
    }

    printf("LAYER 1 OUTPUT: ");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer1_output[i]);
    }
    printf("\n");

    // LAYER 1 ReLU
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        layer1_relu[i] = (layer1_output[i] > 0.0) ? layer1_output[i] : 0.0;
    }

    printf("LAYER 1 RELU: ");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer1_relu[i]);
    }
    printf("\n");  

    // FEED FORWARD LAYER 2
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        float neuron_output = 0.0;
        for (int j = 0; j < LAYER1_NB_NEURONS; j++)
        {
            neuron_output += layer1_output[j] * layer2_weights[i][j];
        }
        neuron_output += layer2_biases[i];
        layer2_output[i] = neuron_output;
    }

    printf("LAYER 2 OUTPUT: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2_output[i]);
    }
    printf("\n");

    // FREE VARIABLES
    free(layer2_output);
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        free(layer2_weights[i]);
    }
    free(layer2_weights);
    free(layer2_biases);

    free(layer1_relu);
    free(layer1_output);
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        free(layer1_weights[i]);
    }
    free(layer1_weights);
    free(layer1_biases);

    free(inputs);
    return 0;
}