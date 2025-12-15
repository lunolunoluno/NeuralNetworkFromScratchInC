#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 4
#define LAYER1_NB_NEURONS 3 // this is also the output

int main(){
    // INIT VARIABLES
    float* inputs = calloc(INPUT_SIZE, sizeof(float));
    float* layer1_biases = calloc(LAYER1_NB_NEURONS, sizeof(float));
    float** layer1_weights = malloc(LAYER1_NB_NEURONS * sizeof(float*));
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        layer1_weights[i] = calloc(INPUT_SIZE, sizeof(float));
    }
    float* layer1_output = calloc(LAYER1_NB_NEURONS, sizeof(float));

    // GIVE VARIABLES INITIAL VALUES
    inputs[0] = 1.0;
    inputs[1] = 2.0;
    inputs[2] = 3.0;
    inputs[3] = 2.5;

    layer1_weights[0][0] = 0.2;
    layer1_weights[0][1] = 0.8;
    layer1_weights[0][2] = -0.5;
    layer1_weights[0][3] = 1.0;
    layer1_weights[1][0] = 0.5;
    layer1_weights[1][1] = -0.91;
    layer1_weights[1][2] = 0.26;
    layer1_weights[1][3] = -0.5;
    layer1_weights[2][0] = -0.26;
    layer1_weights[2][1] = -0.27;
    layer1_weights[2][2] = 0.17;
    layer1_weights[2][3] = 0.87;
    layer1_biases[0] = 2.0;
    layer1_biases[1] = 3.0;
    layer1_biases[2] = 0.5;

    // FEED FORWARD LAYER 1
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        float neuron_output = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            neuron_output += inputs[j] * layer1_weights[i][j];
        }
        neuron_output += layer1_biases[i];
        layer1_output[i] = neuron_output;
    }
    
    printf("LAYER 1 OUTPUT: ");
    for (int i = 0; i < LAYER1_NB_NEURONS; i++)
    {
        printf("%f,", layer1_output[i]);
    }
    printf("\n");
    
    // FREE VARIABLES
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