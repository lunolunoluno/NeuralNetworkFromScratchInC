#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    float *label_one_hot = calloc(LAYER2_NB_NEURONS, sizeof(float));

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
    float *layer2_softmax = calloc(LAYER2_NB_NEURONS, sizeof(float));
    float *layer2_d_softmax = calloc(LAYER2_NB_NEURONS, sizeof(float));

    // GIVE VARIABLES INITIAL VALUES
    inputs[0] = -0.8326189893369458;
    inputs[1] = -0.5538462048218106;
    int label = 0;
    label_one_hot[label] = 1.0;

    float layer1_weights_values[INPUT_SIZE * LAYER1_NB_NEURONS] = {0.01764052, 0.02240893,
                                                                   0.00400157, 0.01867558,
                                                                   0.00978738, -0.00977278};
    set_weights_value(layer1_weights, INPUT_SIZE, LAYER1_NB_NEURONS, layer1_weights_values);
    layer1_biases[0] = 0.2;
    layer1_biases[1] = 0.003;
    layer1_biases[2] = 0.0005;

    float layer2_weights_values[LAYER1_NB_NEURONS * LAYER2_NB_NEURONS] = {0.00950088, -0.00151357, -0.00103219,
                                                                          0.00410599, 0.00144044, 0.01454273,
                                                                          0.00761038, 0.00121675, 0.00443863};
    set_weights_value(layer2_weights, LAYER1_NB_NEURONS, LAYER2_NB_NEURONS, layer2_weights_values);
    layer2_biases[0] = -0.1;
    layer2_biases[1] = 0.002;
    layer2_biases[2] = -0.0005;
    
    printf("INPUTS: ");
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        printf("%f, ",inputs[i]);
    }
    printf("\n");
    printf("ONE-HOT LABEL: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f, ",label_one_hot[i]);
    }
    printf("\n");

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
            neuron_output += layer1_relu[j] * layer2_weights[i][j];
            // printf("%f * %f = %f\n", layer1_relu[j], layer2_weights[i][j], neuron_output);
        }
        neuron_output += layer2_biases[i];
        // printf("+ %f = %f\n", layer2_biases[i], neuron_output);
        layer2_output[i] = neuron_output;
    }

    printf("LAYER 2 OUTPUT: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2_output[i]);
    }
    printf("\n");

    // LAYER 2 SOFTMAX
    // get max value
    float layer2_output_max = 0;
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        if (layer2_output[i] > layer2_output_max)
        {
            layer2_output_max = layer2_output[i];
        }
    }
    // get unnormalized probabilities
    float exp_sum = 0;
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        layer2_softmax[i] = exp(layer2_output[i] - layer2_output_max);
        exp_sum += layer2_softmax[i];
    }
    // normalize the probabilities
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        layer2_softmax[i] = layer2_softmax[i] / exp_sum;
    }

    printf("LAYER 2 SOFTMAX: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2_softmax[i]);
    }
    printf("\n");

    // CALCULATE CATEGORICAL CROSS-ENTROPY LOSS
    layer2_softmax[label] = (layer2_softmax[label] <= 0) ? 0.0000001 : layer2_softmax[label];
    float loss = -log(layer2_softmax[label]);
    printf("CATEGORICAL CROSS-ENTROPY LOSS %f\n", loss);

    // BACKPROPAGATION OF SOFTMAX + CROSS-ENTROPY (easier to implement)
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        layer2_d_softmax[i] = layer2_softmax[i] - label_one_hot[i];
    }

    printf("BACKPROPAGATION OF SOFTMAX + CROSS-ENTROPY: ");
    for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    {
        printf("%f,", layer2_d_softmax[i]);
    }
    printf("\n");

    // FREE VARIABLES
    free(layer2_d_softmax);
    free(layer2_softmax);
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

    free(label_one_hot);
    free(inputs);
    return 0;
}